#pragma once

#include <spdlog/spdlog.h>

#include <typeinfo>

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
  template <typename T>
  static T get_from_row(soci::row& row, uint64_t pos) {
    // This function is needed to make dispatching soci's typing system easier...
    const soci::column_properties& props = row.get_properties(pos);
    if constexpr (std::is_same_v<T, int64_t>) {
      switch (props.get_data_type()) {
        case soci::dt_long_long:
          static_assert(sizeof(long long) <= sizeof(int64_t),  // NOLINT(google-runtime-int)
                        "We currently assume long long is equal to or less than 64 bit.");
          return static_cast<T>(row.get<long long>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_integer:
          // NOLINTNEXTLINE(google-runtime-int)
          static_assert(sizeof(int) <= sizeof(int64_t), "We currently assume int is equal to or less than 64 bit.");
          return static_cast<T>(row.get<int>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_unsigned_long_long:
          FAIL(fmt::format("Tried to extract integer from unsigned long long column {}", props.get_name()));
          break;
        case soci::dt_string:
          FAIL(fmt::format("Tried to extract integer from string column {}", props.get_name()));
          break;
        case soci::dt_double:
          FAIL(fmt::format("Tried to extract integer from double column {}", props.get_name()));
          break;
        case soci::dt_date:
          FAIL(fmt::format("Tried to extract integer from data column {}", props.get_name()));
          break;
        default:
          FAIL(fmt::format("Tried to extract integer from unknown data type ({}) column {}",
                           static_cast<int>(props.get_data_type()), props.get_name()));
      }
    }

    if constexpr (std::is_same_v<T, uint64_t>) {
      switch (props.get_data_type()) {
        case soci::dt_unsigned_long_long:
          static_assert(sizeof(unsigned long long) <= sizeof(uint64_t),  // NOLINT(google-runtime-int)
                        "We currently assume unsined long long is equal to or less than 64 bit.");
          return static_cast<T>(row.get<unsigned long long>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_long_long:
          FAIL(fmt::format("Tried to extract unsigned long long from signed long long column {}", props.get_name()));
        case soci::dt_integer:
          FAIL(fmt::format("Tried to extract unsigned long long from signed integer column {}", props.get_name()));
        case soci::dt_string:
          FAIL(fmt::format("Tried to extract integer from string column {}", props.get_name()));
          break;
        case soci::dt_double:
          FAIL(fmt::format("Tried to extract integer from double column {}", props.get_name()));
          break;
        case soci::dt_date:
          FAIL(fmt::format("Tried to extract integer from data column {}", props.get_name()));
          break;
        default:
          FAIL(fmt::format("Tried to extract integer from unknown data type ({}) column {}",
                           static_cast<int>(props.get_data_type()), props.get_name()));
      }
    }

    if constexpr (std::is_same_v<T, bool>) {
      switch (props.get_data_type()) {
        case soci::dt_unsigned_long_long:
          return static_cast<T>(row.get<unsigned long long>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_long_long:
          return static_cast<T>(row.get<long long>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_integer:
          return static_cast<T>(row.get<int>(pos));  // NOLINT(google-runtime-int)
        case soci::dt_string:
          FAIL(fmt::format("Tried to extract bool from string column {}", props.get_name()));
          break;
        case soci::dt_double:
          FAIL(fmt::format("Tried to extract bool from double column {}", props.get_name()));
          break;
        case soci::dt_date:
          FAIL(fmt::format("Tried to extract bool from data column {}", props.get_name()));
          break;
        default:
          FAIL(fmt::format("Tried to extract bool from unknown data type ({}) column {}",
                           static_cast<int>(props.get_data_type()), props.get_name()));
      }
    }

    const std::type_info& ti1 = typeid(T);
    const std::string type_id = ti1.name();
    FAIL(fmt::format("Unsupported type in get_from_row: {}", type_id));
  }

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
