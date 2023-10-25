#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <deque>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"
#include "internal/utils/utils.hpp"

namespace storage::file_watcher {

struct FileFrame {
  int64_t dataset_id;
  int64_t file_id;
  int64_t index;
  int64_t label;
};
class FileWatcher {
 public:
  std::atomic<bool>* stop_file_watcher;
  explicit FileWatcher(const YAML::Node& config, const int64_t& dataset_id, std::atomic<bool>* stop_file_watcher,
                       int16_t insertion_threads = 1)
      : stop_file_watcher{stop_file_watcher},
        config_{config},
        dataset_id_{dataset_id},
        insertion_threads_{insertion_threads},
        disable_multithreading_{insertion_threads <= 1},
        storage_database_connection_{storage::database::StorageDatabaseConnection(config)} {
    SPDLOG_INFO("Initializing file watcher for dataset {}.", dataset_id_);
    if (stop_file_watcher == nullptr) {
      FAIL("stop_file_watcher_ is nullptr.");
    }

    SPDLOG_INFO("Initializing file watcher for dataset {}.", dataset_id_);

    if (config_["storage"]["sample_dbinsertion_batchsize"]) {
      sample_dbinsertion_batchsize_ = config_["storage"]["sample_dbinsertion_batchsize"].as<int64_t>();
    }
    if (config_["storage"]["force_fallback"]) {
      force_fallback_ = config["storage"]["force_fallback"].as<bool>();
    }
    soci::session session = storage_database_connection_.get_session();

    std::string dataset_path;
    int64_t filesystem_wrapper_type_int;
    try {
      session << "SELECT base_path, filesystem_wrapper_type FROM datasets "
                 "WHERE dataset_id = :dataset_id",
          soci::into(dataset_path), soci::into(filesystem_wrapper_type_int), soci::use(dataset_id_);
    } catch (const soci::soci_error& e) {
      SPDLOG_ERROR("Error while reading dataset path and filesystem wrapper type from database: {}", e.what());
      *stop_file_watcher = true;
      return;
    }

    const auto filesystem_wrapper_type =
        static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type_int);

    if (dataset_path.empty()) {
      SPDLOG_ERROR("Dataset with id {} not found.", dataset_id_);
      *stop_file_watcher = true;
      return;
    }

    filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(dataset_path, filesystem_wrapper_type);

    dataset_path_ = dataset_path;
    filesystem_wrapper_type_ = filesystem_wrapper_type;

    if (!filesystem_wrapper->exists(dataset_path) || !filesystem_wrapper->is_directory(dataset_path)) {
      SPDLOG_ERROR("Dataset path {} does not exist or is not a directory.", dataset_path);
      *stop_file_watcher = true;
      return;
    }

    if (!disable_multithreading_) {
      insertion_thread_pool_ = std::vector<std::thread>(insertion_threads_);
    }
  }
  std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper> filesystem_wrapper;
  void run();
  static void handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                                const storage::file_wrapper::FileWrapperType& file_wrapper_type, int64_t timestamp,
                                const storage::filesystem_wrapper::FilesystemWrapperType& filesystem_wrapper_type,
                                int64_t dataset_id, const YAML::Node& file_wrapper_config, const YAML::Node& config,
                                int64_t sample_dbinsertion_batchsize, bool force_fallback);
  void update_files_in_directory(const std::string& directory_path, int64_t timestamp);
  static void insert_file_frame(const storage::database::StorageDatabaseConnection& storage_database_connection,
                                const std::vector<FileFrame>& file_frame, bool force_fallback);
  void seek_dataset();
  void seek();
  static bool check_valid_file(
      const std::string& file_path, const std::string& data_file_extension, bool ignore_last_timestamp,
      int64_t timestamp, storage::database::StorageDatabaseConnection& storage_database_connection,
      const std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper>& filesystem_wrapper);
  static void postgres_copy_insertion(const std::vector<FileFrame>& file_frame,
                                      const storage::database::StorageDatabaseConnection& storage_database_connection);
  static void fallback_insertion(const std::vector<FileFrame>& file_frame,
                                 const storage::database::StorageDatabaseConnection& storage_database_connection);

 private:
  YAML::Node config_;
  int64_t dataset_id_;
  int16_t insertion_threads_;
  bool disable_multithreading_;
  std::vector<std::thread> insertion_thread_pool_;
  int64_t sample_dbinsertion_batchsize_ = 1000000;
  bool force_fallback_ = false;
  storage::database::StorageDatabaseConnection storage_database_connection_;
  std::string dataset_path_;
  storage::filesystem_wrapper::FilesystemWrapperType filesystem_wrapper_type_;
};
}  // namespace storage::file_watcher
