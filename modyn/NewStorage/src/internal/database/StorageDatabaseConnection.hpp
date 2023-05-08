#ifndef STORAGE_DATABASE_CONNECTION_H
#define STORAGE_DATABASE_CONNECTION_H

#include <soci/postgresql/soci-postgresql.h>
#include <soci/soci.h>
#include <soci/sqlite3/soci-sqlite3.h>
#include <yaml-cpp/yaml.h>

namespace storage {
class StorageDatabaseConnection {
private:
  std::string username;
  std::string password;
  std::string host;
  std::string port;
  std::string database;
  int hash_partition_modulus = 8;
  bool sample_table_unlogged = true;

public:
  std::string drivername;
  StorageDatabaseConnection(YAML::Node config) {
    if (!config["storage"]["database"]) {
      throw std::runtime_error("No database configuration found");
    }
    this->drivername =
        config["storage"]["database"]["drivername"].as<std::string>();
    this->username =
        config["storage"]["database"]["username"].as<std::string>();
    this->password =
        config["storage"]["database"]["password"].as<std::string>();
    this->host = config["storage"]["database"]["host"].as<std::string>();
    this->port = config["storage"]["database"]["port"].as<std::string>();
    this->database =
        config["storage"]["database"]["database"].as<std::string>();
    if (config["storage"]["database"]["hash_partition_modulus"]) {
      this->hash_partition_modulus =
          config["storage"]["database"]["hash_partition_modulus"].as<int>();
    }
    if (config["storage"]["database"]["sample_table_unlogged"]) {
      this->sample_table_unlogged =
          config["storage"]["database"]["sample_table_unlogged"].as<bool>();
    }
  }
  void create_tables();
  bool add_dataset(std::string name, std::string base_path,
                   std::string filesystem_wrapper_type,
                   std::string file_wrapper_type, std::string description,
                   std::string version, std::string file_wrapper_config,
                   bool ignore_last_timestamp = false,
                   int file_watcher_interval = 5);
  bool delete_dataset(std::string name);
  void add_sample_dataset_partition(std::string dataset_name,
                                    soci::session *session);
  soci::session *get_session();
};

} // namespace storage

#endif