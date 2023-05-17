#pragma once

#include <spdlog/spdlog.h>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "soci/postgresql/soci-postgresql.h"
#include "soci/soci.h"
#include "soci/sqlite3/soci-sqlite3.h"
#include "yaml-cpp/yaml.h"
namespace storage {
class StorageDatabaseConnection {
 private:
  std::string username_;
  std::string password_;
  std::string host_;
  std::string port_;
  std::string database_;
  int16_t hash_partition_modulus_ = 8;

 public:
  std::string drivername;
  explicit StorageDatabaseConnection(const YAML::Node& config) {
    if (!config["storage"]["database"]) {
      throw std::runtime_error("No database configuration found");
    }
    drivername = config["storage"]["database"]["drivername"].as<std::string>();
    username_ = config["storage"]["database"]["username"].as<std::string>();
    password_ = config["storage"]["database"]["password"].as<std::string>();
    host_ = config["storage"]["database"]["host"].as<std::string>();
    port_ = config["storage"]["database"]["port"].as<std::string>();
    database_ = config["storage"]["database"]["database"].as<std::string>();
    if (config["storage"]["database"]["hash_partition_modulus"]) {
      hash_partition_modulus_ = config["storage"]["database"]["hash_partition_modulus"].as<int16_t>();
    }
  }
  void create_tables() const;
  bool add_dataset(const std::string& name, const std::string& base_path,
                   const FilesystemWrapperType& filesystem_wrapper_type, const FileWrapperType& file_wrapper_type,
                   const std::string& description, const std::string& version, const std::string& file_wrapper_config,
                   const bool& ignore_last_timestamp, const int& file_watcher_interval = 5) const;
  bool delete_dataset(const std::string& name) const;
  void add_sample_dataset_partition(const std::string& dataset_name) const;
  soci::session get_session() const;
};

}  // namespace storage
