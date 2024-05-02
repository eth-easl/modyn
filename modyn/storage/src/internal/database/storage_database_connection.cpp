#include "internal/database/storage_database_connection.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "modyn/utils/utils.hpp"
#include "soci/postgresql/soci-postgresql.h"
#include "soci/sqlite3/soci-sqlite3.h"

using namespace modyn::storage;

soci::session StorageDatabaseConnection::get_session() const {
  const std::string connection_string =
      fmt::format("dbname={} user={} password={} host={} port={}", database_, username_, password_, host_, port_);
  soci::connection_parameters parameters;

  switch (drivername_) {
    case DatabaseDriver::POSTGRESQL:
      parameters = soci::connection_parameters(soci::postgresql, connection_string);
      break;
    case DatabaseDriver::SQLITE3:
      parameters = soci::connection_parameters(soci::sqlite3, connection_string);
      break;
    default:
      FAIL("Unsupported database driver");
  }
  return soci::session(parameters);
}

void StorageDatabaseConnection::create_tables() const {
  soci::session session = get_session();

  std::string dataset_table_sql;
  std::string file_table_sql;
  std::string sample_table_sql;
  switch (drivername_) {
    case DatabaseDriver::POSTGRESQL:
      dataset_table_sql =
#include "sql/PostgreSQLDataset.sql"
          ;
      file_table_sql =
#include "sql/PostgreSQLFile.sql"
          ;
      sample_table_sql =
#include "sql/PostgreSQLSample.sql"
          ;
      break;
    case DatabaseDriver::SQLITE3:
      dataset_table_sql =
#include "sql/SQLiteDataset.sql"
          ;
      file_table_sql =
#include "sql/SQLiteFile.sql"
          ;
      sample_table_sql =
#include "sql/SQLiteSample.sql"
          ;
      break;
    default:
      FAIL("Unsupported database driver");
  }
  session << dataset_table_sql;

  session << file_table_sql;

  session << sample_table_sql;

  if (drivername_ == DatabaseDriver::POSTGRESQL && sample_table_unlogged_) {
    session << "ALTER TABLE samples SET UNLOGGED";
  }

  session.close();
}

bool StorageDatabaseConnection::add_dataset(const std::string& name, const std::string& base_path,
                                            const FilesystemWrapperType& filesystem_wrapper_type,
                                            const FileWrapperType& file_wrapper_type, const std::string& description,
                                            const std::string& version, const std::string& file_wrapper_config,
                                            const bool ignore_last_timestamp, const int64_t file_watcher_interval,
                                            const bool upsert) const {
  soci::session session = get_session();

  auto filesystem_wrapper_type_int = static_cast<int64_t>(filesystem_wrapper_type);
  auto file_wrapper_type_int = static_cast<int64_t>(file_wrapper_type);
  std::string boolean_string = ignore_last_timestamp ? "true" : "false";

  const int64_t dataset_id = get_dataset_id(name);

  if (dataset_id != -1 && !upsert) {
    // Dataset already exists
    SPDLOG_ERROR("Cannot insert dataset {} as it already exists", name);
    return false;
  }

  if (dataset_id != -1) {
    SPDLOG_INFO("Dataset {} already exists, updating...", name);

    const std::string update_query =
#include "sql/SQLUpdateDataset.sql"
        ;

    session << update_query, soci::use(base_path), soci::use(filesystem_wrapper_type_int),
        soci::use(file_wrapper_type_int), soci::use(description), soci::use(version), soci::use(file_wrapper_config),
        soci::use(boolean_string), soci::use(file_watcher_interval), soci::use(dataset_id);
    return true;
  }

  // Dataset does not exist
  const std::string insertion_query =
#include "sql/SQLInsertDataset.sql"
      ;

  switch (drivername_) {
    // same logic for both dbms > fallthrough
    case DatabaseDriver::POSTGRESQL:
    case DatabaseDriver::SQLITE3:
      session << insertion_query, soci::use(name), soci::use(base_path), soci::use(filesystem_wrapper_type_int),
          soci::use(file_wrapper_type_int), soci::use(description), soci::use(version), soci::use(file_wrapper_config),
          soci::use(boolean_string), soci::use(file_watcher_interval);
      break;
    default:
      SPDLOG_ERROR("Error adding dataset: Unsupported database driver.");
      return false;
  }

  // Create partition table for samples
  if (!add_sample_dataset_partition(name)) {
    FAIL("Partition creation failed.");
  }
  return true;
}

int64_t StorageDatabaseConnection::get_dataset_id(const std::string& name) const {
  soci::session session = get_session();

  int64_t dataset_id = -1;
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(name);
  session.close();

  return dataset_id;
}

DatabaseDriver StorageDatabaseConnection::get_drivername(const YAML::Node& config) {
  ASSERT(config["storage"]["database"], "No database configuration found");

  const auto drivername = config["storage"]["database"]["drivername"].as<std::string>();
  if (drivername == "postgresql") {
    return DatabaseDriver::POSTGRESQL;
  }
  if (drivername == "sqlite3") {
    return DatabaseDriver::SQLITE3;
  }

  FAIL("Unsupported database driver: " + drivername);
}

bool StorageDatabaseConnection::delete_dataset(const std::string& name, const int64_t dataset_id) const {
  soci::session session = get_session();

  // Delete all samples for this dataset
  try {
    session << "DELETE FROM samples WHERE dataset_id = :dataset_id", soci::use(dataset_id);
  } catch (const soci::soci_error& e) {
    SPDLOG_ERROR("Error deleting samples for dataset {}: {}", name, e.what());
    return false;
  }

  // Delete all files for this dataset
  try {
    session << "DELETE FROM files WHERE dataset_id = :dataset_id", soci::use(dataset_id);
  } catch (const soci::soci_error& e) {
    SPDLOG_ERROR("Error deleting files for dataset {}: {}", name, e.what());
    return false;
  }

  // Delete the dataset
  try {
    session << "DELETE FROM datasets WHERE name = :name", soci::use(name);
  } catch (const soci::soci_error& e) {
    SPDLOG_ERROR("Error deleting dataset {}: {}", name, e.what());
    return false;
  }

  session.close();

  return true;
}

bool StorageDatabaseConnection::add_sample_dataset_partition(const std::string& dataset_name) const {
  soci::session session = get_session();
  int64_t dataset_id = get_dataset_id(dataset_name);
  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} not found", dataset_name);
    return false;
  }
  switch (drivername_) {
    case DatabaseDriver::POSTGRESQL: {
      std::string dataset_partition_table_name = "samples__did" + std::to_string(dataset_id);
      try {
        session << fmt::format(
            "CREATE TABLE IF NOT EXISTS {} "
            "PARTITION OF samples "
            "FOR VALUES IN ({}) "
            "PARTITION BY HASH (sample_id)",
            dataset_partition_table_name, dataset_id);
      } catch (const soci::soci_error& e) {
        SPDLOG_ERROR("Error creating partition table for dataset {}: {}", dataset_name, e.what());
        session.close();
        return false;
      }

      try {
        for (int64_t i = 0; i < hash_partition_modulus_; i++) {
          std::string hash_partition_name = dataset_partition_table_name + "_part" + std::to_string(i);
          session << fmt::format(
              "CREATE TABLE IF NOT EXISTS {} "
              "PARTITION OF {} "
              "FOR VALUES WITH (modulus {}, REMAINDER {})",
              hash_partition_name, dataset_partition_table_name, hash_partition_modulus_, i);
        }
      } catch (const soci::soci_error& e) {
        SPDLOG_ERROR("Error creating hash partitions for dataset {}: {}", dataset_name, e.what());
        session.close();
        return false;
      }
      break;
    }
    case DatabaseDriver::SQLITE3: {
      SPDLOG_INFO(
          "Skipping partition creation for dataset {}, not supported for "
          "driver.",
          dataset_name);
      break;
    }
    default:
      FAIL("Unsupported database driver.");
  }

  session.close();
  return true;
}
