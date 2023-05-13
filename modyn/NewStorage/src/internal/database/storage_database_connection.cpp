#include "internal/database/storage_database_connection.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>

#include "soci/postgresql/soci-postgresql.h"
#include "soci/sqlite3/soci-sqlite3.h"

using namespace storage;

soci::session* StorageDatabaseConnection::get_session() {
  std::string connection_string = "dbname='" + this->database + "' user='" + this->username + "' password='" +
                                  this->password + "' host='" + this->host + "' port=" + this->port;
  if (this->drivername == "postgresql") {
    soci::connection_parameters parameters(soci::postgresql, connection_string);
    std::unique_ptr<soci::session> sql(new soci::session(parameters));
    return sql.release();
  } else if (this->drivername == "sqlite3") {
    soci::connection_parameters parameters(soci::sqlite3, connection_string);
    std::unique_ptr<soci::session> sql(new soci::session(parameters));
    return sql.release();
  } else {
    throw std::runtime_error("Unsupported database driver: " + this->drivername);
  }
}

void StorageDatabaseConnection::create_tables() {
  soci::session* session = this->get_session();

  std::string input_file_path = std::filesystem::path(__FILE__).parent_path() / "sql/Dataset.sql";
  std::ifstream dataset_input_file(input_file_path);
  if (dataset_input_file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(dataset_input_file)), std::istreambuf_iterator<char>());
    dataset_input_file.close();
    *session << content;
  } else {
    SPDLOG_ERROR("Unable to open Dataset.sql file");
  }

  std::string file_input_file_path;
  std::string sample_input_file_path;
  if (this->drivername == "postgresql") {
    sample_input_file_path = std::filesystem::path(__FILE__).parent_path() / "sql/Sample.sql";
    file_input_file_path = std::filesystem::path(__FILE__).parent_path() / "sql/File.sql";
  } else if (this->drivername == "sqlite3") {
    sample_input_file_path = std::filesystem::path(__FILE__).parent_path() / "sql/SQLiteSample.sql";
    file_input_file_path = std::filesystem::path(__FILE__).parent_path() / "sql/SQLiteFile.sql";
  } else {
    throw std::runtime_error("Unsupported database driver: " + this->drivername);
  }

  std::ifstream file_input_file(file_input_file_path);
  if (file_input_file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(file_input_file)), std::istreambuf_iterator<char>());
    file_input_file.close();
    *session << content;
  } else {
    SPDLOG_ERROR("Unable to open File.sql file");
  }

  std::ifstream sample_input_file(sample_input_file_path);
  if (sample_input_file.is_open()) {
    std::string content((std::istreambuf_iterator<char>(sample_input_file)), std::istreambuf_iterator<char>());
    sample_input_file.close();
    *session << content;
  } else {
    SPDLOG_ERROR("Unable to open Sample.sql file");
  }

  delete session;
}

bool StorageDatabaseConnection::add_dataset(std::string name, std::string base_path,
                                            std::string filesystem_wrapper_type, std::string file_wrapper_type,
                                            std::string description, std::string version,
                                            std::string file_wrapper_config, bool ignore_last_timestamp,
                                            int file_watcher_interval) {
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
      throw std::runtime_error("Unsupported database driver: " + this->drivername);
    }

    // Create partition table for samples
    add_sample_dataset_partition(name, session);

    delete session;
  } catch (const std::exception e&) {
    SPDLOG_ERROR("Error adding dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

bool StorageDatabaseConnection::delete_dataset(std::string name) {
  try {
    soci::session* session = this->get_session();

    long long dataset_id;
    *session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(name);

    // Delete all samples for this dataset
    *session << "DELETE FROM samples WHERE dataset_id = :dataset_id", soci::use(dataset_id);

    // Delete all files for this dataset
    *session << "DELETE FROM files WHERE dataset_id = :dataset_id", soci::use(dataset_id);

    // Delete the dataset
    *session << "DELETE FROM datasets WHERE name = :name", soci::use(name);

    delete session;

  } catch (std::exception e) {
    SPDLOG_ERROR("Error deleting dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

void StorageDatabaseConnection::add_sample_dataset_partition(std::string dataset_name, soci::session* session) {
  if (this->drivername == "postgresql") {
    long long dataset_id;
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

    for (long long i = 0; i < this->hash_partition_modulus; i++) {
      std::string hash_partition_name = dataset_partition_table_name + "_part" + std::to_string(i);
      *session << "CREATE TABLE IF NOT EXISTS :hash_partition_name PARTITION "
                  "OF :dataset_partition_table_name "
                  "FOR VALUES WITH (modulus :hash_partition_modulus, "
                  "REMAINDER :i)",
          soci::use(hash_partition_name), soci::use(dataset_partition_table_name),
          soci::use(this->hash_partition_modulus), soci::use(i);
    }
  } else {
    SPDLOG_INFO(
        "Skipping partition creation for dataset {}, not supported for "
        "driver {}",
        dataset_name, this->drivername);
  }
}
