#include "internal/file_watcher/file_watcher.hpp"

#include <spdlog/spdlog.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace storage;

void FileWatcher::handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                                    const std::string& file_wrapper_type, int64_t timestamp,
                                    const YAML::Node& file_wrapper_config) {
  soci::session* sql = storage_database_connection_->get_session();

  std::vector<std::string> valid_files;
  for (const auto& file_path : file_paths) {
    if (check_valid_file(file_path, data_file_extension, /*ignore_last_timestamp=*/false, timestamp)) {
      valid_files.push_back(file_path);
    }
  }

  if (!valid_files.empty()) {
    std::string file_path;  // NOLINT // soci::use() requires a non-const reference
    int64_t number_of_samples;
    std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>> file_frame =
        std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>();
    for (const auto& file_path : valid_files) {
      auto file_wrapper =
          Utils::get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config, filesystem_wrapper);
      number_of_samples = file_wrapper->get_number_of_samples();
      int64_t modified_time = filesystem_wrapper->get_modified_time(file_path);
      int64_t created_time = filesystem_wrapper->get_created_time(file_path);
      *sql << "INSERT INTO files (dataset_id, path, number_of_samples, "
              "created_at, updated_at) VALUES (:dataset_id, :path, "
              ":number_of_samples, :created_at, :updated_at)",
          soci::use(dataset_id_), soci::use(file_path), soci::use(number_of_samples), soci::use(created_time),
          soci::use(modified_time);

      long long file_id;  // NOLINT // soci get_last_insert_id requires a long long
      sql->get_last_insert_id("files", file_id);

      const std::vector<int64_t> labels = file_wrapper->get_all_labels();

      int32_t index = 0;
      for (const auto& label : labels) {
        file_frame.emplace_back(dataset_id_, file_id, index, label);
        index++;
      }
    }

    if (storage_database_connection_->drivername == "postgresql") {
      postgres_copy_insertion(file_frame, sql);
    } else {
      fallback_insertion(file_frame, sql);
    }
  }
}

void FileWatcher::postgres_copy_insertion(const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame,
                                          soci::session* sql) const {
  const std::string table_name = "samples__did" + std::to_string(dataset_id_);
  const std::string table_columns = "(dataset_id,file_id,sample_index,label)";
  const std::string cmd =
      "COPY " + table_name + table_columns + " FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')";

  // Create stringbuffer, dump data into file buffer csv and send to
  // postgresql
  std::stringstream ss;
  for (const auto& frame : file_frame) {
    ss << std::get<0>(frame) << "," << std::get<1>(frame) << "," << std::get<2>(frame) << "," << std::get<3>(frame)
       << "\n";
  }

  std::string tmp_file_name = "temp.csv";
  std::ofstream file(tmp_file_name);
  if (file.is_open()) {
    file << ss.str();
    file.close();
  } else {
    SPDLOG_ERROR("Unable to open file");
  }

  *sql << cmd, soci::use(tmp_file_name);

  // Remove temp file
  (void)remove("temp.csv");
}

bool FileWatcher::check_valid_file(const std::string& file_path, const std::string& data_file_extension,
                                   bool ignore_last_timestamp, int64_t timestamp) {
  const std::string file_extension = file_path.substr(file_path.find_last_of('.'));
  if (file_extension != data_file_extension) {
    return false;
  }
  soci::session* sql = storage_database_connection_->get_session();

  int64_t file_id = -1;

  *sql << "SELECT file_id FROM files WHERE path = :file_path", soci::into(file_id), soci::use(file_path);

  if (file_id == -1) {
    if (ignore_last_timestamp) {
      return true;
    }
    return filesystem_wrapper->get_modified_time(file_path) > timestamp;
  }
  return false;
}

void FileWatcher::update_files_in_directory(const std::string& directory_path, int64_t timestamp) {
  std::string file_wrapper_config;
  std::string file_wrapper_type;

  soci::session* sql = storage_database_connection_->get_session();

  *sql << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
          "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type), soci::into(file_wrapper_config), soci::use(dataset_id_);

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
  const auto data_file_extension = file_wrapper_config_node["file_extension"].as<std::string>();

  std::vector<std::string> file_paths = filesystem_wrapper->list(directory_path, /*recursive=*/true);

  if (disable_multithreading_) {
    handle_file_paths(file_paths, data_file_extension, file_wrapper_type, timestamp, file_wrapper_config_node);
  } else {
    const int64_t files_per_thread = static_cast<int64_t>(file_paths.size()) / insertion_threads_;
    std::vector<std::thread> children;
    for (int64_t i = 0; i < insertion_threads_; i++) {
      std::vector<std::string> file_paths_thread = std::vector<std::string>();
      if (i == insertion_threads_ - 1) {
        file_paths_thread.insert(file_paths_thread.end(), file_paths.begin() + i * files_per_thread, file_paths.end());
      } else {
        file_paths_thread.insert(file_paths_thread.end(), file_paths.begin() + i * files_per_thread,
                                 file_paths.begin() + (i + 1) * files_per_thread);
      }
      std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
      const FileWatcher watcher(config_, dataset_id_, stop_file_watcher);
      children.emplace_back(&FileWatcher::handle_file_paths, watcher, file_paths_thread, data_file_extension,
                            file_wrapper_type, timestamp, file_wrapper_config_node);
    }

    for (auto& child : children) {
      child.join();
    }
  }
}

void FileWatcher::seek_dataset() {
  soci::session* sql = storage_database_connection_->get_session();

  int64_t last_timestamp;

  *sql << "SELECT last_timestamp FROM datasets "
          "WHERE dataset_id = :dataset_id",
      soci::into(last_timestamp), soci::use(dataset_id_);

  update_files_in_directory(dataset_path_, last_timestamp);
}

void FileWatcher::seek() {
  soci::session* sql = storage_database_connection_->get_session();
  std::string dataset_name;

  *sql << "SELECT name FROM datasets WHERE dataset_id = :dataset_id", soci::into(dataset_name), soci::use(dataset_id_);

  try {
    seek_dataset();

    int64_t last_timestamp;
    *sql << "SELECT updated_at FROM files WHERE dataset_id = :dataset_id ORDER "
            "BY updated_at DESC LIMIT 1",
        soci::into(last_timestamp), soci::use(dataset_id_);

    if (last_timestamp > 0) {
      *sql << "UPDATE datasets SET last_timestamp = :last_timestamp WHERE dataset_id = "
              ":dataset_id",
          soci::use(last_timestamp), soci::use(dataset_id_);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("File watcher failed for dataset {} with error: {}", dataset_name, e.what());
    stop_file_watcher_->store(true);
  }
}

void FileWatcher::run() {
  soci::session* sql = storage_database_connection_->get_session();

  int64_t file_watcher_interval;
  *sql << "SELECT file_watcher_interval FROM datasets WHERE dataset_id = :dataset_id",
      soci::into(file_watcher_interval), soci::use(dataset_id_);

  if (file_watcher_interval == 0) {
    throw std::runtime_error("File watcher interval is invalid, does the dataset exist?");
  }

  while (true) {
    try {
      seek();
      if (stop_file_watcher_->load()) {
        break;
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("File watcher failed: {}", e.what());
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}
