#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <string>
#include <vector>

#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/utils/utils.hpp"

namespace storage {
class FileWatcher {
 private:
  YAML::Node config_;
  int64_t dataset_id_;
  int16_t insertion_threads_;
  bool disable_multithreading_;
  int32_t sample_dbinsertion_batchsize_ = 1000000;
  StorageDatabaseConnection storage_database_connection_;
  std::atomic<bool>* stop_file_watcher_;
  std::string dataset_path_;
  FilesystemWrapperType filesystem_wrapper_type_;

 public:
  explicit FileWatcher(const YAML::Node& config, const int64_t& dataset_id,  // NOLINT
                       std::atomic<bool>* stop_file_watcher)
      : config_{config},
        dataset_id_{dataset_id},
        storage_database_connection_{StorageDatabaseConnection(config_)},
        stop_file_watcher_{stop_file_watcher} {
    insertion_threads_ = config_["storage"]["insertion_threads"].as<int16_t>();
    disable_multithreading_ = insertion_threads_ <= 1;  // NOLINT
    if (config_["storage"]["sample_dbinsertion_batchsize"]) {
      sample_dbinsertion_batchsize_ = config_["storage"]["sample_dbinsertion_batchsize"].as<int32_t>();
    }
    soci::session session = storage_database_connection_.get_session();

    std::string dataset_path;
    int64_t filesystem_wrapper_type_int;
    session << "SELECT base_path, filesystem_wrapper_type FROM datasets "
               "WHERE dataset_id = :dataset_id",
        soci::into(dataset_path), soci::into(filesystem_wrapper_type_int), soci::use(dataset_id_);
    const auto filesystem_wrapper_type = static_cast<FilesystemWrapperType>(filesystem_wrapper_type_int);

    if (dataset_path.empty()) {
      throw std::runtime_error("Loading dataset failed, is the dataset_id correct?");
    }

    filesystem_wrapper = Utils::get_filesystem_wrapper(dataset_path, filesystem_wrapper_type);

    dataset_path_ = dataset_path;
    filesystem_wrapper_type_ = filesystem_wrapper_type;

    if (filesystem_wrapper->exists(dataset_path) && filesystem_wrapper->is_directory(dataset_path)) {
      spdlog::info("Dataset path {} exists and is a directory.", dataset_path);
    } else {
      throw std::runtime_error("Dataset path " + dataset_path + " does not exist or is not a directory.");
    }
  }
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper;
  void run();
  void handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                         const FileWrapperType& file_wrapper_type, int64_t timestamp,
                         const YAML::Node& file_wrapper_config);
  void update_files_in_directory(const std::string& directory_path, int64_t timestamp);
  void seek_dataset();
  void seek();
  bool check_valid_file(const std::string& file_path, const std::string& data_file_extension,
                        bool ignore_last_timestamp, int64_t timestamp);
  void postgres_copy_insertion(const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame) const;
  void fallback_insertion(const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame) const;
};
}  // namespace storage
