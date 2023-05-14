#include "internal/file_watcher/file_watcher.hpp"

#include <spdlog/spdlog.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>

#include "internal/utils/utils.hpp"

using namespace storage;

void FileWatcher::handle_file_paths(std::vector<std::string>* file_paths, std::string data_file_extension,
                                    std::string file_wrapper_type, AbstractFilesystemWrapper* filesystem_wrapper,
                                    int timestamp, YAML::Node file_wrapper_config) {
  soci::session* sql = this->storage_database_connection->get_session();

  std::vector<std::string> valid_files;
  for (const auto& file_path : *file_paths) {
    if (this->check_valid_file(file_path, data_file_extension, false, timestamp, filesystem_wrapper)) {
      valid_files.push_back(file_path);
    }
  }

  if (valid_files.size() > 0) {
    std::string file_path;
    int number_of_samples;
    std::vector<std::tuple<long long, long long, int, int>> file_frame =
        std::vector<std::tuple<long long, long long, int, int>>();
    for (const auto& file_path : valid_files) {
      AbstractFileWrapper* file_wrapper =
          Utils::get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config, filesystem_wrapper);
      number_of_samples = file_wrapper->get_number_of_samples();

      *sql << "INSERT INTO files (dataset_id, path, number_of_samples, "
              "created_at, updated_at) VALUES (:dataset_id, :path, "
              ":number_of_samples, :created_at, :updated_at)",
          soci::use(this->dataset_id), soci::use(file_path), soci::use(number_of_samples),
          soci::use(filesystem_wrapper->get_created_time(file_path)),
          soci::use(filesystem_wrapper->get_modified_time(file_path));

      long long file_id;
      sql->get_last_insert_id("files", file_id);

      std::vector<int> labels = *file_wrapper->get_all_labels();

      std::tuple<long long, long long, int, int> frame;
      int index = 0;
      for (const auto& label : labels) {
        frame = std::make_tuple(this->dataset_id, file_id, index, label);
        file_frame.push_back(frame);
        index++;
      }
    }

    if (this->storage_database_connection->drivername == "postgresql") {
      this->postgres_copy_insertion(file_frame, sql);
    } else {
      this->fallback_insertion(file_frame, sql);
    }
  }
}

void FileWatcher::fallback_insertion(std::vector<std::tuple<long long, long long, int, int>> file_frame,
                                     soci::session* sql) {
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  for (const auto& frame : file_frame) {
    query += "(" + std::to_string(std::get<0>(frame)) + "," + std::to_string(std::get<1>(frame)) + "," +
             std::to_string(std::get<2>(frame)) + "," + std::to_string(std::get<3>(frame)) + "),";
  }

  // Remove last comma
  query.pop_back();
  *sql << query;
}

void FileWatcher::postgres_copy_insertion(std::vector<std::tuple<long long, long long, int, int>> file_frame,
                                          soci::session* sql) {
  std::string table_name = "samples__did" + std::to_string(this->dataset_id);
  std::string table_columns = "(dataset_id,file_id,sample_index,label)";
  std::string cmd = "COPY " + table_name + table_columns + " FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')";

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
  remove("temp.csv");
}

bool FileWatcher::check_valid_file(std::string file_path, std::string data_file_extension, bool ignore_last_timestamp,
                                   int timestamp, AbstractFilesystemWrapper* filesystem_wrapper) {
  std::string file_extension = file_path.substr(file_path.find_last_of("."));
  if (file_extension != data_file_extension) {
    return false;
  }
  soci::session* sql = this->storage_database_connection->get_session();

  long long file_id = -1;

  *sql << "SELECT file_id FROM files WHERE path = :file_path", soci::into(file_id), soci::use(file_path);

  if (file_id == -1) {
    if (ignore_last_timestamp) {
      return true;
    }
    return filesystem_wrapper->get_modified_time(file_path) > timestamp;
  }
  return false;
}

void FileWatcher::update_files_in_directory(AbstractFilesystemWrapper* filesystem_wrapper, std::string directory_path,
                                            int timestamp) {
  std::string file_wrapper_config;
  std::string file_wrapper_type;

  soci::session* sql = this->storage_database_connection->get_session();

  *sql << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
          "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type), soci::into(file_wrapper_config), soci::use(this->dataset_id);

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
  std::string data_file_extension = file_wrapper_config_node["file_extension"].as<std::string>();

  std::vector<std::string>* file_paths = filesystem_wrapper->list(directory_path, true);

  if (this->disable_multithreading) {
    this->handle_file_paths(file_paths, data_file_extension, file_wrapper_type, filesystem_wrapper, timestamp,
                            file_wrapper_config_node);
  } else {
    int files_per_thread = file_paths->size() / this->insertion_threads;
    std::vector<std::thread> children;
    for (int i = 0; i < this->insertion_threads; i++) {
      std::vector<std::string>* file_paths_thread = new std::vector<std::string>();
      if (i == this->insertion_threads - 1) {
        file_paths_thread->insert(file_paths_thread->end(), file_paths->begin() + i * files_per_thread,
                                  file_paths->end());
      } else {
        file_paths_thread->insert(file_paths_thread->end(), file_paths->begin() + i * files_per_thread,
                                  file_paths->begin() + (i + 1) * files_per_thread);
      }
      std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
      FileWatcher watcher(this->config_file, this->dataset_id, true, stop_file_watcher);
      children.push_back(std::thread(&FileWatcher::handle_file_paths, watcher, file_paths_thread, data_file_extension,
                                     file_wrapper_type, filesystem_wrapper, timestamp, file_wrapper_config_node));
    }

    for (unsigned long i = 0; i < children.size(); i++) {
      children[i].join();
    }
  }
}

void FileWatcher::seek_dataset() {
  soci::session* sql = this->storage_database_connection->get_session();

  std::string dataset_path;
  std::string dataset_filesystem_wrapper_type;
  int last_timestamp;

  *sql << "SELECT base_path, filesystem_wrapper_type, last_timestamp FROM datasets "
          "WHERE dataset_id = :dataset_id",
      soci::into(dataset_path), soci::into(dataset_filesystem_wrapper_type), soci::into(last_timestamp),
      soci::use(this->dataset_id);

  if (dataset_path.empty()) {
    throw std::runtime_error("Loading dataset failed, is the dataset_id correct?");
  }

  AbstractFilesystemWrapper* filesystem_wrapper =
      Utils::get_filesystem_wrapper(dataset_path, dataset_filesystem_wrapper_type);

  if (filesystem_wrapper->exists(dataset_path) && filesystem_wrapper->is_directory(dataset_path)) {
    this->update_files_in_directory(filesystem_wrapper, dataset_path, last_timestamp);
  } else {
    throw std::runtime_error("Dataset path does not exist or is not a directory.");
  }
}

void FileWatcher::seek() {
  soci::session* sql = this->storage_database_connection->get_session();
  std::string dataset_name;

  *sql << "SELECT name FROM datasets WHERE dataset_id = :dataset_id", soci::into(dataset_name),
      soci::use(this->dataset_id);

  try {
    this->seek_dataset();

    int last_timestamp;
    *sql << "SELECT updated_at FROM files WHERE dataset_id = :dataset_id ORDER "
            "BY updated_at DESC LIMIT 1",
        soci::into(last_timestamp), soci::use(this->dataset_id);

    if (last_timestamp > 0) {
      *sql << "UPDATE datasets SET last_timestamp = :last_timestamp WHERE dataset_id = "
              ":dataset_id",
          soci::use(last_timestamp), soci::use(this->dataset_id);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("File watcher failed for dataset {} with error: {}", dataset_name, e.what());
    this->stop_file_watcher.get()->store(true);
  }
}

void FileWatcher::run() {
  soci::session* sql = this->storage_database_connection->get_session();

  int file_watcher_interval;
  *sql << "SELECT file_watcher_interval FROM datasets WHERE dataset_id = :dataset_id", soci::into(file_watcher_interval),
      soci::use(this->dataset_id);

  if (file_watcher_interval == 0) {
    throw std::runtime_error("File watcher interval is invalid, does the dataset exist?");
  }

  while (true) {
    try {
      this->seek();
      if (this->stop_file_watcher.get()->load()) {
        break;
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("File watcher failed: {}", e.what());
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}
