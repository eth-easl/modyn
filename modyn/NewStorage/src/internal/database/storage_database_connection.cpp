#include "internal/database/storage_database_connection.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "soci/postgresql/soci-postgresql.h"
#include "soci/sqlite3/soci-sqlite3.h"

using namespace storage;

soci::session* StorageDatabaseConnection::get_session() const {
  std::string connection_string = "dbname='" + this->database_ + "' user='" + this->username_ + "' password='" +
                                  this->password_ + "' host='" + this->host_ + "' port=" + this->port_;
  soci::connection_parameters parameters;
  if (this->drivername == "postgresql") {
    parameters = soci::connection_parameters(soci::postgresql, connection_string);
  } else if (this->drivername == "sqlite3") {
    parameters = soci::connection_parameters(soci::sqlite3, connection_string);
  } else {
    throw std::runtime_error("Error getting session: Unsupported database driver: " + this->drivername);
  }
  std::unique_ptr<soci::session> sql(new soci::session(parameters));
  return sql.release();
}

void StorageDatabaseConnection::create_tables() const {
  soci::session* session = this->get_session();

  const char* dataset_table_sql =
#include "sql/Dataset.sql"
      ;

  *session << dataset_table_sql;

  const char* file_table_sql;
  const char* sample_table_sql;
  if (this->drivername == "postgresql") {
    file_table_sql =
#include "sql/File.sql"
        ;
    sample_table_sql =
#include "sql/Sample.sql"
        ;
  } else if (this->drivername == "sqlite3") {
    file_table_sql =
#include "sql/SQLiteFile.sql"
        ;
    sample_table_sql =
#include "sql/SQLiteSample.sql"
        ;
  } else {
    throw std::runtime_error("Error creating tables: Unsupported database driver: " + this->drivername);
  }

  *session << file_table_sql;

  *session << sample_table_sql;
}

bool StorageDatabaseConnection::add_dataset(const std::string& name, const std::string& base_path,
                                            const std::string& filesystem_wrapper_type,
                                            const std::string& file_wrapper_type, const std::string& description,
                                            const std::string& version, const std::string& file_wrapper_config,
                                            const bool& ignore_last_timestamp, const int& file_watcher_interval) const {
  try {
    soci::session* session = this->get_session();

    std::string boolean_string = ignore_last_timestamp ? "true" : "false";
    if (this->drivername == "postgresql") {
      *session << "INSERT INTO datasets (name, base_path, filesystem_wrapper_type, "
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
          soci::use(name), soci::use(base_path), soci::use(filesystem_wrapper_type), soci::use(file_wrapper_type),
          soci::use(description), soci::use(version), soci::use(file_wrapper_config), soci::use(boolean_string),
          soci::use(file_watcher_interval);
    } else if (this->drivername == "sqlite3") {
      *session << "INSERT INTO datasets (name, base_path, filesystem_wrapper_type, "
                  "file_wrapper_type, description, version, file_wrapper_config, "
                  "ignore_last_timestamp, file_watcher_interval, last_timestamp) "
                  "VALUES (:name, "
                  ":base_path, :filesystem_wrapper_type, :file_wrapper_type, "
                  ":description, :version, :file_wrapper_config, "
                  ":ignore_last_timestamp, :file_watcher_interval, 0)",
          soci::use(name), soci::use(base_path), soci::use(filesystem_wrapper_type), soci::use(file_wrapper_type),
          soci::use(description), soci::use(version), soci::use(file_wrapper_config), soci::use(boolean_string),
          soci::use(file_watcher_interval);
    } else {
      throw std::runtime_error("Error adding dataset: Unsupported database driver: " + this->drivername);
    }

    // Create partition table for samples
    add_sample_dataset_partition(name, session);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error adding dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

bool StorageDatabaseConnection::delete_dataset(const std::string& name) const {
  try {
    soci::session* session = this->get_session();

    int64_t dataset_id;
    *session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(name);

    // Delete all samples for this dataset
    *session << "DELETE FROM samples WHERE dataset_id = :dataset_id", soci::use(dataset_id);

    // Delete all files for this dataset
    *session << "DELETE FROM files WHERE dataset_id = :dataset_id", soci::use(dataset_id);

    // Delete the dataset
    *session << "DELETE FROM datasets WHERE name = :name", soci::use(name);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error deleting dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

void StorageDatabaseConnection::add_sample_dataset_partition(const std::string& dataset_name,
                                                             soci::session* session) const {
  if (this->drivername == "postgresql") {
    int64_t dataset_id;
    *session << "SELECT dataset_id FROM datasets WHERE name = :dataset_name", soci::into(dataset_id),
        soci::use(dataset_name);
    if (dataset_id == 0) {
      throw std::runtime_error("Dataset " + dataset_name + " not found");
    }
    std::string dataset_partition_table_name = "samples__did" + std::to_string(dataset_id);
    *session << "CREATE TABLE IF NOT EXISTS :dataset_partition_table_name "
                "PARTITION OF samples "
                "FOR VALUES IN (:dataset_id) "
                "PARTITION BY HASH (sample_id)",
        soci::use(dataset_partition_table_name), soci::use(dataset_id);

    for (int64_t i = 0; i < this->hash_partition_modulus_; i++) {
      std::string hash_partition_name = dataset_partition_table_name + "_part" + std::to_string(i);
      *session << "CREATE TABLE IF NOT EXISTS :hash_partition_name PARTITION "
                  "OF :dataset_partition_table_name "
                  "FOR VALUES WITH (modulus :hash_partition_modulus, "
                  "REMAINDER :i)",
          soci::use(hash_partition_name), soci::use(dataset_partition_table_name),
          soci::use(this->hash_partition_modulus_), soci::use(i);
    }
  } else {
    SPDLOG_INFO(
        "Skipping partition creation for dataset {}, not supported for "
        "driver {}",
        dataset_name, this->drivername);
  }
}
