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
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

struct FileFrame {
  // Struct to store file information for insertion into the database when watching a dataset.
  int64_t file_id;
  int64_t index;
  int64_t label;
};
class FileWatcher {
 public:
  explicit FileWatcher(const YAML::Node& config, int64_t dataset_id, std::atomic<bool>* stop_file_watcher,
                       int16_t insertion_threads = 1)
      : stop_file_watcher{stop_file_watcher},
        config_{config},
        dataset_id_{dataset_id},
        insertion_threads_{insertion_threads},
        disable_multithreading_{insertion_threads <= 1},
        storage_database_connection_{StorageDatabaseConnection(config)} {
    ASSERT(stop_file_watcher != nullptr, "stop_file_watcher_ is nullptr.");
    SPDLOG_INFO("Initializing file watcher for dataset {}.", dataset_id_);

    if (config_["storage"]["sample_dbinsertion_batchsize"]) {
      sample_dbinsertion_batchsize_ = config_["storage"]["sample_dbinsertion_batchsize"].as<int64_t>();
    }
    if (config_["storage"]["force_fallback"]) {
      force_fallback_ = config["storage"]["force_fallback"].as<bool>();
    }
    soci::session session = storage_database_connection_.get_session();

    std::string dataset_path;
    auto filesystem_wrapper_type_int = static_cast<int64_t>(FilesystemWrapperType::INVALID_FSW);
    std::string file_wrapper_config;
    auto file_wrapper_type_id = static_cast<int64_t>(FileWrapperType::INVALID_FW);
    try {
      session << "SELECT base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM datasets "
                 "WHERE dataset_id = :dataset_id",
          soci::into(dataset_path), soci::into(filesystem_wrapper_type_int), soci::into(file_wrapper_type_id),
          soci::into(file_wrapper_config), soci::use(dataset_id_);
    } catch (const soci::soci_error& e) {
      SPDLOG_ERROR("Error while reading dataset path and filesystem wrapper type from database: {}", e.what());
      *stop_file_watcher = true;
      return;
    }

    session.close();

    filesystem_wrapper_type_ = static_cast<FilesystemWrapperType>(filesystem_wrapper_type_int);

    SPDLOG_INFO("FileWatcher for dataset {} uses path {}, file_wrapper_id {} and file_system_id {}", dataset_id_,
                dataset_path, file_wrapper_type_id, filesystem_wrapper_type_int);

    if (dataset_path.empty()) {
      SPDLOG_ERROR("Dataset with id {} not found.", dataset_id_);
      *stop_file_watcher = true;
      return;
    }

    filesystem_wrapper = get_filesystem_wrapper(filesystem_wrapper_type_);

    dataset_path_ = dataset_path;

    if (!filesystem_wrapper->exists(dataset_path_) || !filesystem_wrapper->is_directory(dataset_path_)) {
      SPDLOG_ERROR("Dataset path {} does not exist or is not a directory.", dataset_path_);
      *stop_file_watcher = true;
      return;
    }

    if (file_wrapper_type_id == -1) {
      SPDLOG_ERROR("Failed to get file wrapper type");
      *stop_file_watcher = true;
      return;
    }

    file_wrapper_type_ = static_cast<FileWrapperType>(file_wrapper_type_id);

    if (file_wrapper_config.empty()) {
      SPDLOG_ERROR("Failed to get file wrapper config");
      *stop_file_watcher = true;
      return;
    }

    file_wrapper_config_node_ = YAML::Load(file_wrapper_config);

    if (!file_wrapper_config_node_["file_extension"]) {
      SPDLOG_ERROR("Config does not contain file_extension");
      *stop_file_watcher = true;
      return;
    }

    data_file_extension_ = file_wrapper_config_node_["file_extension"].as<std::string>();

    if (!disable_multithreading_) {
      insertion_thread_pool_.reserve(insertion_threads_);
      insertion_thread_exceptions_ = std::vector<std::atomic<bool>>(insertion_threads_);
    }
    SPDLOG_INFO("FileWatcher for dataset {} initialized", dataset_id_);
  }
  void run();
  void search_for_new_files_in_directory(const std::string& directory_path, int64_t timestamp);
  void seek_dataset(soci::session& session);
  void seek(soci::session& session);
  static void handle_file_paths(std::vector<std::string>::iterator file_paths_begin,
                                std::vector<std::string>::iterator file_paths_end, FileWrapperType file_wrapper_type,
                                int64_t timestamp, FilesystemWrapperType filesystem_wrapper_type, int64_t dataset_id,
                                const YAML::Node* file_wrapper_config, const YAML::Node* config,
                                int64_t sample_dbinsertion_batchsize, bool force_fallback,
                                std::atomic<bool>* exception_thrown);
  static void handle_files_for_insertion(std::vector<std::string>& files_for_insertion,
                                         const FileWrapperType& file_wrapper_type, int64_t dataset_id,
                                         const YAML::Node& file_wrapper_config, int64_t sample_dbinsertion_batchsize,
                                         bool force_fallback, soci::session& session, DatabaseDriver& database_driver,
                                         const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper);
  static void insert_file_samples(const std::vector<FileFrame>& file_samples, int64_t dataset_id, bool force_fallback,
                                  soci::session& session, DatabaseDriver& database_driver);
  static int64_t insert_file(const std::string& file_path, int64_t dataset_id,
                             const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper,
                             const std::unique_ptr<FileWrapper>& file_wrapper, soci::session& session,
                             DatabaseDriver& database_driver);
  static bool check_file_for_insertion(const std::string& file_path, bool ignore_last_timestamp, int64_t timestamp,
                                       int64_t dataset_id, const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper,
                                       soci::session& session);
  static void postgres_copy_insertion(const std::vector<FileFrame>& file_samples, int64_t dataset_id,
                                      soci::session& session);
  static void fallback_insertion(const std::vector<FileFrame>& file_samples, int64_t dataset_id,
                                 soci::session& session);
  static int64_t insert_file(const std::string& file_path, int64_t dataset_id, soci::session& session,
                             uint64_t number_of_samples, int64_t modified_time);
  static int64_t insert_file_using_returning_statement(const std::string& file_path, int64_t dataset_id,
                                                       soci::session& session, uint64_t number_of_samples,
                                                       int64_t modified_time);
  std::atomic<bool>* stop_file_watcher;
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper;

 private:
  YAML::Node config_;
  int64_t dataset_id_ = -1;
  int16_t insertion_threads_ = 1;
  bool disable_multithreading_ = false;
  std::vector<std::thread> insertion_thread_pool_;
  std::vector<std::atomic<bool>> insertion_thread_exceptions_;
  int64_t sample_dbinsertion_batchsize_ = 1000000;
  bool force_fallback_ = false;
  StorageDatabaseConnection storage_database_connection_;
  std::string dataset_path_;
  FilesystemWrapperType filesystem_wrapper_type_;
  FileWrapperType file_wrapper_type_;
  YAML::Node file_wrapper_config_node_;
  std::string data_file_extension_;
};
}  // namespace modyn::storage
