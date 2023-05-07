#include "StorageDatabaseConnection.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

using namespace storage;

soci::session *StorageDatabaseConnection::get_session() {
  std::string connection_string = "dbname=" + this->database +
                                  " user=" + this->username +
                                  " password=" + this->password +
                                  " host=" + this->host + " port=" + this->port;
  if (this->drivername == "postgresql") {
    return new soci::session(*soci::factory_postgresql(), connection_string);
  } else if (this->drivername == "sqlite") {
    return new soci::session(*soci::factory_sqlite3(), connection_string);
  } else {
    throw std::runtime_error("Unsupported database driver: " +
                             this->drivername);
  }
}

void StorageDatabaseConnection::create_tables() {
  soci::session *session = this->get_session();

  std::ifstream ifs("sql/Dataset.sql");
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));
  session->prepare << content;

  ifs = std::ifstream("sql/File.sql");
  content = std::string((std::istreambuf_iterator<char>(ifs)),
                        (std::istreambuf_iterator<char>()));
  session->prepare << content;

  if (this->drivername == "postgresql") {
    ifs = std::ifstream("sql/SamplePartition.sql");
  } else if (this->drivername == "sqlite") {
    ifs = std::ifstream("sql/Sample.sql");
  } else {
    throw std::runtime_error("Unsupported database driver: " +
                             this->drivername);
  }
  content = std::string((std::istreambuf_iterator<char>(ifs)),
                        (std::istreambuf_iterator<char>()));
  session->prepare << content;

  session->commit();
  delete session;
}

bool StorageDatabaseConnection::add_dataset(
    std::string name, std::string base_path,
    std::string filesystem_wrapper_type, std::string file_wrapper_type,
    std::string description, std::string version,
    std::string file_wrapper_config, bool ignore_last_timestamp,
    int file_watcher_interval) {
  try {
    soci::session *session = this->get_session();

    std::string boolean_string = ignore_last_timestamp ? "true" : "false";
    // Insert dataset
    *session
        << "INSERT INTO datasets (name, base_path, filesystem_wrapper_type, "
           "file_wrapper_type, description, version, file_wrapper_config, "
           "ignore_last_timestamp, file_watcher_interval) VALUES (:name, "
           ":base_path, :filesystem_wrapper_type, :file_wrapper_type, "
           ":description, :version, :file_wrapper_config, "
           ":ignore_last_timestamp, :file_watcher_interval) "
           "ON DUPLICATE KEY UPDATE base_path = :base_path, "
           "filesystem_wrapper_type = :filesystem_wrapper_type, "
           "file_wrapper_type = :file_wrapper_type, description = "
           ":description, version = :version, file_wrapper_config = "
           ":file_wrapper_config, ignore_last_timestamp = "
           ":ignore_last_timestamp, file_watcher_interval = "
           ":file_watcher_interval",
        soci::use(name), soci::use(base_path),
        soci::use(filesystem_wrapper_type), soci::use(file_wrapper_type),
        soci::use(description), soci::use(version),
        soci::use(file_wrapper_config), soci::use(boolean_string),
        soci::use(file_watcher_interval);

    // Create partition table for samples
    add_sample_dataset_partition(name, session);

    session->commit();
    delete session;

  } catch (std::exception e) {
    SPDLOG_ERROR("Error adding dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

bool StorageDatabaseConnection::delete_dataset(std::string name) {
  try {
    soci::session *session = this->get_session();

    int dataset_id;
    *session << "SELECT id FROM dataset WHERE name = :name",
        soci::into(dataset_id), soci::use(name);

    // Delete all samples for this dataset
    *session
        << "DELETE FROM samples s WHERE s.dataset_id IN (SELECT d.dataset_id "
           "FROM dataset d WHERE d.name = :name)",
        soci::use(name);

    // Delete all files for this dataset
    *session
        << "DELETE FROM files f WHERE f.dataset_id IN (SELECT d.dataset_id "
           "FROM dataset d WHERE d.name = :name)",
        soci::use(name);

    // Delete the dataset
    *session << "DELETE FROM datasets WHERE name = :name", soci::use(name);

    session->commit();
    delete session;

  } catch (std::exception e) {
    SPDLOG_ERROR("Error deleting dataset {}: {}", name, e.what());
    return false;
  }
  return true;
}

void StorageDatabaseConnection::add_sample_dataset_partition(
    std::string dataset_name, soci::session *session) {
  if (this->drivername == "postgresql") {
    long long dataset_id;
    *session << "SELECT dataset_id FROM datasets WHERE name = :dataset_name",
        soci::into(dataset_id), soci::use(dataset_name);
    if (dataset_id == 0) {
      throw std::runtime_error("Dataset " + dataset_name + " not found");
    }
    std::string dataset_partition_table_name =
        "samples__did" + std::to_string(dataset_id);
    *session << "CREATE TABLE IF NOT EXISTS :dataset_partition_table_name "
                "PARTITION OF samples "
                "FOR VALUES IN (:dataset_id) "
                "PARTITION BY HASH (sample_id)",
        soci::use(dataset_partition_table_name), soci::use(dataset_id);

    for (int i = 0; i < this->hash_partition_modulus; i++) {
      std::string hash_partition_name =
          dataset_partition_table_name + "_part" + std::to_string(i);
      *session << "CREATE TABLE IF NOT EXISTS :hash_partition_name PARTITION "
                  "OF :dataset_partition_table_name "
                  "FOR VALUES WITH (modulus :hash_partition_modulus, "
                  "REMAINDER :i)",
          soci::use(hash_partition_name),
          soci::use(dataset_partition_table_name),
          soci::use(this->hash_partition_modulus), soci::use(i);
    }
  } else {
    SPDLOG_INFO("Skipping partition creation for dataset {}, not supported for "
                "driver {}",
                dataset_name, this->drivername);
  }
}
