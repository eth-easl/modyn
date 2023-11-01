#pragma once

#include <spdlog/spdlog.h>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "modyn/utils/utils.hpp"
#include "soci/postgresql/soci-postgresql.h"
#include "soci/soci.h"
#include "soci/sqlite3/soci-sqlite3.h"
#include "yaml-cpp/yaml.h"

namespace modyn::storage {

enum class DatabaseDriver { POSTGRESQL, SQLITE3 };

class StorageDatabaseConnection {
 public:
  explicit StorageDatabaseConnection(const YAML::Node& config) {
    if (!config["storage"]["database"]) {
      FAIL("No database configuration found");
    }
    drivername_ = get_drivername(config);
    username_ = config["storage"]["database"]["username"].as<std::string>();
    password_ = config["storage"]["database"]["password"].as<std::string>();
    host_ = config["storage"]["database"]["host"].as<std::string>();
    port_ = config["storage"]["database"]["port"].as<std::string>();
    database_ = config["storage"]["database"]["database"].as<std::string>();
    SPDLOG_INFO("Database configuration: username: {}, password: {}, host: {}, port: {}, database: {}", username_,
                password_, host_, port_, database_);
    if (config["storage"]["database"]["hash_partition_modulus"]) {
      hash_partition_modulus_ = config["storage"]["database"]["hash_partition_modulus"].as<int16_t>();
    }
    if (config["storage"]["sample_table_unlogged"]) {
      sample_table_unlogged_ = config["storage"]["sample_table_unlogged"].as<bool>();
    }
  }
  void create_tables() const;
  bool add_dataset(const std::string& name, const std::string& base_path,
                   const FilesystemWrapperType& filesystem_wrapper_type, const FileWrapperType& file_wrapper_type,
                   const std::string& description, const std::string& version, const std::string& file_wrapper_config,
                   bool ignore_last_timestamp, int64_t file_watcher_interval = 5) const;
  bool delete_dataset(const std::string& name, int64_t dataset_id) const;
  void add_sample_dataset_partition(const std::string& dataset_name) const;
  soci::session get_session() const;
  DatabaseDriver get_drivername() const { return drivername_; }

 private:
  static DatabaseDriver get_drivername(const YAML::Node& config);
  int64_t get_dataset_id(const std::string& name) const;
  std::string username_;
  std::string password_;
  std::string host_;
  std::string port_;
  std::string database_;
  bool sample_table_unlogged_ = false;
  int16_t hash_partition_modulus_ = 8;
  DatabaseDriver drivername_;
};

}  // namespace modyn::storage
