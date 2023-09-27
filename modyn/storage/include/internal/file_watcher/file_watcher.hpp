#pragma once

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <optional>

#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/utils/utils.hpp"

namespace storage {
class FileWatcher {
 public:
  std::atomic<bool>* stop_file_watcher_;
  explicit FileWatcher(const YAML::Node& config, const int64_t& dataset_id,  // NOLINT
                       std::atomic<bool>* stop_file_watcher, int16_t insertion_threads = 1)
      : config_{config},
        dataset_id_{dataset_id},
        insertion_threads_{insertion_threads},
        storage_database_connection_{StorageDatabaseConnection(config)},
        stop_file_watcher_{stop_file_watcher} {
    if (stop_file_watcher_ == nullptr) {
      FAIL("stop_file_watcher_ is nullptr.");
    }

    SPDLOG_INFO("Initializing file watcher for dataset {}.", dataset_id_);

    disable_multithreading_ = insertion_threads_ <= 1;  // NOLINT
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

    if (disable_multithreading_) {
      SPDLOG_INFO("Multithreading disabled.");
    } else {
      SPDLOG_INFO("Multithreading enabled.");

      thread_pool.resize(insertion_threads_);

      for (auto& thread : thread_pool) {
        thread = std::thread([&]() {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(mtx);
              cv.wait(lock, [&]() { return !tasks.empty(); });
              task = std::move(tasks.front());
              tasks.pop_front();
            }
            if (!task) break;  // If the task is empty, it's a signal to terminate the thread
            task();
          }
        });
      }
    }
  }
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper;
  void run();
  static void handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                         const FileWrapperType& file_wrapper_type, int64_t timestamp,
                         const YAML::Node& file_wrapper_config, const YAML::Node& config) {
    StorageDatabaseConnection storage_database_connection(config);
    soci::session session = storage_database_connection.get_session();

    std::vector<std::string> valid_files;
    for (const auto& file_path : file_paths) {
      if (check_valid_file(file_path, data_file_extension, /*ignore_last_timestamp=*/false, timestamp)) {
        valid_files.push_back(file_path);
      }
    }

    SPDLOG_INFO("Found {} valid files", valid_files.size());

    if (!valid_files.empty()) {
      std::string file_path = valid_files.front();
      int64_t number_of_samples;
      std::vector<FileFrame> file_frame;
      auto file_wrapper = storage::utils::get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config, filesystem_wrapper);
      for (const auto& file_path : valid_files) {
        file_wrapper->set_file_path(file_path);
        number_of_samples = file_wrapper->get_number_of_samples();
        int64_t modified_time = filesystem_wrapper->get_modified_time(file_path);
        session << "INSERT INTO files (dataset_id, path, number_of_samples, "
                    "updated_at) VALUES (:dataset_id, :path, "
                    ":number_of_samples, :updated_at)",
            soci::use(dataset_id_), soci::use(file_path), soci::use(number_of_samples), soci::use(modified_time);

        // Check if the insert was successful.
        std::optional<long long> file_id = session.get_last_insert_id<long long>("files");
        if (!file_id) {
          // The insert was not successful.
          SPDLOG_ERROR("Failed to insert file into database");
          continue;
        }

        const std::vector<int64_t> labels = file_wrapper->get_all_labels();

        int32_t index = 0;
        for (const auto& label : labels) {
          file_frame.emplace_back(dataset_id_, *file_id, index, label);
          index++;
        }
      }

      // Move the file_frame vector into the insertion function.
      switch (storage_database_connection_.get_driver()) {
        case DatabaseDriver::POSTGRESQL:
          postgres_copy_insertion(std::move(file_frame));
          break;
        case DatabaseDriver::SQLITE3:
          fallback_insertion(std::move(file_frame));
          break;
        default:
          FAIL("Unsupported database driver");
      }
    }
  }
  void update_files_in_directory(const std::string& directory_path, int64_t timestamp);
  void seek_dataset();
  void seek();
  bool check_valid_file(const std::string& file_path, const std::string& data_file_extension,
                        bool ignore_last_timestamp, int64_t timestamp);
  void postgres_copy_insertion(const std::vector<FileFrame>& file_frame) const;
  void fallback_insertion(const std::vector<FileFrame>& file_frame) const;

 private:
  YAML::Node config_;
  int64_t dataset_id_;
  int16_t insertion_threads_;
  bool disable_multithreading_;
  int32_t sample_dbinsertion_batchsize_ = 1000000;
  StorageDatabaseConnection storage_database_connection_;
  std::string dataset_path_;
  FilesystemWrapperType filesystem_wrapper_type_;
  std::vector<std::thread> thread_pool;
  std::deque<std::function<void()>> tasks;
  std::mutex mtx;
  std::condition_variable cv;
  struct FileFrame {
    int64_t dataset_id;
    int64_t file_id;
    int32_t index;
    int32_t label;
  };
};
}  // namespace storage
