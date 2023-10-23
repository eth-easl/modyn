#include "internal/database/storage_database_connection.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "internal/utils/utils.hpp"
#include "soci/postgresql/soci-postgresql.h"
#include "soci/sqlite3/soci-sqlite3.h"

using namespace storage::database;

soci::session StorageDatabaseConnection::get_session() const {
  const std::string connection_string = "dbname='" + database_ + "' user='" + username_ + "' password='" + password_ +
                                        "' host='" + host_ + "' port=" + port_;
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

  const char* dataset_table_sql;
  const char* file_table_sql;
  const char* sample_table_sql;
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
}

bool StorageDatabaseConnection::add_dataset(
    const std::string& name, const std::string& base_path,
    const storage::filesystem_wrapper::FilesystemWrapperType& filesystem_wrapper_type,
    const storage::file_wrapper::FileWrapperType& file_wrapper_type, const std::string& description,
    const std::string& version, const std::string& file_wrapper_config, const bool& ignore_last_timestamp,
    const int& file_watcher_interval) const {
    soci::session session = get_session();

    SPDLOG_INFO("Adding dataset {} to database", name);
    SPDLOG_INFO("Filesystem wrapper type: {}", filesystem_wrapper_type);
    SPDLOG_INFO("File wrapper type: {}", file_wrapper_type);
    auto filesystem_wrapper_type_int = static_cast<int64_t>(filesystem_wrapper_type);
    auto file_wrapper_type_int = static_cast<int64_t>(file_wrapper_type);
    std::string boolean_string = ignore_last_timestamp ? "true" : "false";

    SPDLOG_INFO("Adding dataset {} to database", name);
    SPDLOG_INFO("Filesystem wrapper type: {}", filesystem_wrapper_type_int);
    SPDLOG_INFO("File wrapper type: {}", file_wrapper_type_int);
    if (get_dataset_id(name) != -1) {
      SPDLOG_ERROR("Dataset {} already exists", name);
      return false;
    }
    switch (drivername_) {
      case DatabaseDriver::POSTGRESQL:
        session << "INSERT INTO datasets (name, base_path, filesystem_wrapper_type, "
                   "file_wrapper_type, description, version, file_wrapper_config, "
                   "ignore_last_timestamp, file_watcher_interval, last_timestamp) "
                   "VALUES (:name, "
                   ":base_path, :filesystem_wrapper_type, :file_wrapper_type, "
                   ":description, :version, :file_wrapper_config, "
                   ":ignore_last_timestamp, :file_watcher_interval, 0) "
                   "ON DUPLICATE KEY UPDATE base_path = :base_path, "
                   "filesystem_wrapper_type = :filesystem_wrapper_type, "
                   "file_wrapper_type = :file_wrapper_type, description = "
                   ":description, version = :version, file_wrapper_config = "
                   ":file_wrapper_config, ignore_last_timestamp = "
                   ":ignore_last_timestamp, file_watcher_interval = "
                   ":file_watcher_interval, last_timestamp=0",
            soci::use(name), soci::use(base_path), soci::use(filesystem_wrapper_type_int),
            soci::use(file_wrapper_type_int), soci::use(description), soci::use(version),
            soci::use(file_wrapper_config), soci::use(boolean_string), soci::use(file_watcher_interval);
        break;
      case DatabaseDriver::SQLITE3:
        session << "INSERT INTO datasets (name, base_path, filesystem_wrapper_type, "
                   "file_wrapper_type, description, version, file_wrapper_config, "
                   "ignore_last_timestamp, file_watcher_interval, last_timestamp) "
                   "VALUES (:name, "
                   ":base_path, :filesystem_wrapper_type, :file_wrapper_type, "
                   ":description, :version, :file_wrapper_config, "
                   ":ignore_last_timestamp, :file_watcher_interval, 0)",
            soci::use(name), soci::use(base_path), soci::use(filesystem_wrapper_type_int),
            soci::use(file_wrapper_type_int), soci::use(description), soci::use(version),
            soci::use(file_wrapper_config), soci::use(boolean_string), soci::use(file_watcher_interval);
        break;
      default:
        SPDLOG_ERROR("Error adding dataset: Unsupported database driver.");
        return false;
    }

    // Create partition table for samples
    add_sample_dataset_partition(name);
    return true;
}

int64_t StorageDatabaseConnection::get_dataset_id(const std::string& name) const {
  soci::session session = get_session();

  int64_t dataset_id = -1;
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(name);

  return dataset_id;
}

DatabaseDriver StorageDatabaseConnection::get_drivername(const YAML::Node& config) {
  if (!config["storage"]["database"]) {
    FAIL("No database configuration found");
  }
  const auto drivername = config["storage"]["database"]["drivername"].as<std::string>();
  if (drivername == "postgresql") {
    return DatabaseDriver::POSTGRESQL;
  }
  if (drivername == "sqlite3") {
    return DatabaseDriver::SQLITE3;
  }
  FAIL("Unsupported database driver: " + drivername);
}

bool StorageDatabaseConnection::delete_dataset(const std::string& name) const {
  int64_t dataset_id = get_dataset_id(name);
  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} not found", name);
    return false;
  }

  soci::session session = get_session();

  // Delete all samples for this dataset
  session << "DELETE FROM samples WHERE dataset_id = :dataset_id", soci::use(dataset_id);

  // Delete all files for this dataset
  session << "DELETE FROM files WHERE dataset_id = :dataset_id", soci::use(dataset_id);

  // Delete the dataset
  session << "DELETE FROM datasets WHERE name = :name", soci::use(name);

  return true;
}

void StorageDatabaseConnection::add_sample_dataset_partition(const std::string& dataset_name) const {
  soci::session session = get_session();
  int64_t dataset_id = get_dataset_id(dataset_name);
  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} not found", dataset_name);
    return;
  }
  switch (drivername_) {
    case DatabaseDriver::POSTGRESQL: {
      std::string dataset_partition_table_name = "samples__did" + std::to_string(dataset_id);
      session << "CREATE TABLE IF NOT EXISTS :dataset_partition_table_name "
                 "PARTITION OF samples "
                 "FOR VALUES IN (:dataset_id) "
                 "PARTITION BY HASH (sample_id)",
          soci::use(dataset_partition_table_name), soci::use(dataset_id);

      for (int64_t i = 0; i < hash_partition_modulus_; i++) {
        std::string hash_partition_name = dataset_partition_table_name + "_part" + std::to_string(i);
        session << "CREATE TABLE IF NOT EXISTS :hash_partition_name PARTITION "
                   "OF :dataset_partition_table_name "
                   "FOR VALUES WITH (modulus :hash_partition_modulus, "
                   "REMAINDER :i)",
            soci::use(hash_partition_name), soci::use(dataset_partition_table_name), soci::use(hash_partition_modulus_),
            soci::use(i);
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
}
