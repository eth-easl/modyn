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
#include "internal/utils/utils.hpp"

namespace storage::file_watcher {
class FileWatcher {
 public:
  std::atomic<bool>* stop_file_watcher_;
  explicit FileWatcher(const YAML::Node& config, const int64_t& dataset_id, std::atomic<bool>* stop_file_watcher,
                       int16_t insertion_threads = 1)
      : config_{config},
        dataset_id_{dataset_id},
        insertion_threads_{insertion_threads},
        storage_database_connection_{StorageDatabaseConnection(config)},
        stop_file_watcher_{stop_file_watcher},
        disable_multithreading_{insertion_threads <= 1} {
    if (stop_file_watcher_ == nullptr) {
      FAIL("stop_file_watcher_ is nullptr.");
    }

    SPDLOG_INFO("Initializing file watcher for dataset {}.", dataset_id_);

    if (config_["storage"]["sample_dbinsertion_batchsize"]) {
      sample_dbinsertion_batchsize_ = config_["storage"]["sample_dbinsertion_batchsize"].as<int32_t>();
    }
    soci::session session = storage_database_connection_.get_session();

    std::string dataset_path;
    int64_t filesystem_wrapper_type_int;
    try {
      session << "SELECT base_path, filesystem_wrapper_type FROM datasets "
                 "WHERE dataset_id = :dataset_id",
          soci::into(dataset_path), soci::into(filesystem_wrapper_type_int), soci::use(dataset_id_);
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error while reading dataset path and filesystem wrapper type from database: {}", e.what());
      stop_file_watcher_->store(true);
      // This is for testing purposes
      filesystem_wrapper_type_int = 1;
    }
    const auto filesystem_wrapper_type = static_cast<FilesystemWrapperType>(filesystem_wrapper_type_int);

    if (dataset_path.empty()) {
      SPDLOG_ERROR("Dataset with id {} not found.", dataset_id_);
      stop_file_watcher_->store(true);
      return;
    }

    filesystem_wrapper = storage::utils::get_filesystem_wrapper(dataset_path, filesystem_wrapper_type);

    dataset_path_ = dataset_path;
    filesystem_wrapper_type_ = filesystem_wrapper_type;

    if (!filesystem_wrapper->exists(dataset_path) || !filesystem_wrapper->is_directory(dataset_path)) {
      SPDLOG_ERROR("Dataset path {} does not exist or is not a directory.", dataset_path);
      stop_file_watcher_->store(true);
      return;
    }
  }
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper;
  void run();
  static void handle_file_paths(
      const std::vector<std::string>& file_paths, const std::string& data_file_extension,
      const FileWrapperType& file_wrapper_type, int64_t timestamp, const YAML::Node& file_wrapper_config,
      const YAML::Node& config) void update_files_in_directory(const std::string& directory_path, int64_t timestamp);
  static void insert_file_frame(StorageDatabaseConnection storage_database_connection,
                                const std::vector<FileFrame>& file_frame);
  void seek_dataset();
  void seek();
  bool check_valid_file(const std::string& file_path, const std::string& data_file_extension,
                        bool ignore_last_timestamp, int64_t timestamp);
  static void postgres_copy_insertion(const std::vector<FileFrame>& file_frame) const;
  static void fallback_insertion(const std::vector<FileFrame>& file_frame) const;

 private:
  YAML::Node config_;
  int64_t dataset_id_;
  int16_t insertion_threads_;
  bool disable_multithreading_;
  int32_t sample_dbinsertion_batchsize_ = 1000000;
  StorageDatabaseConnection storage_database_connection_;
  std::string dataset_path_;
  FilesystemWrapperType filesystem_wrapper_type_;
  struct FileFrame {
    int64_t dataset_id;
    int64_t file_id;
    int32_t index;
    int32_t label;
  };
};
}  // namespace storage::file_watcher
