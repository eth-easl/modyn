#include "internal/file_watcher/file_watcher.hpp"

#include <spdlog/spdlog.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace storage;

/*
 * Handles the file paths that are passsed.
 * 
 * Checks if the file is valid and if so, inserts the file into the database.
 * 
 * Valid files are files that pass the checks in check_valid_file().
 * 
 * @param file_paths The file paths to be handled.
 * @param data_file_extension The extension of the data files.
 * @param file_wrapper_type The type of the file wrapper.
 * @param timestamp The timestamp to be used for the file.
 */
void FileWatcher::handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                                    const FileWrapperType& file_wrapper_type, int64_t timestamp,
                                    const YAML::Node& file_wrapper_config) {
  soci::session session = storage_database_connection_.get_session();

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
      session << "INSERT INTO files (dataset_id, path, number_of_samples, "
                 "updated_at) VALUES (:dataset_id, :path, "
                 ":number_of_samples, :updated_at)",
          soci::use(dataset_id_), soci::use(file_path), soci::use(number_of_samples), soci::use(modified_time);

      long long file_id;  // NOLINT // soci get_last_insert_id requires a long long
      session.get_last_insert_id("files", file_id);

      const std::vector<int64_t> labels = file_wrapper->get_all_labels();

      int32_t index = 0;
      for (const auto& label : labels) {
        file_frame.emplace_back(dataset_id_, file_id, index, label);
        index++;
      }
    }

    if (storage_database_connection_.drivername == "postgresql") {
      postgres_copy_insertion(file_frame);
    } else {
      fallback_insertion(file_frame);
    }
  }
}

/*
 * Inserts the file frame into the database using the optimized postgresql copy command.
 * 
 * The data is expected in a vector of tuples frame which is defined as dataset_id, file_id, sample_index, label.
 * It is then dumped into a csv file buffer and sent to postgresql using the copy command.
 * 
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::postgres_copy_insertion(
    const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame) const {
  soci::session session = storage_database_connection_.get_session();
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

  session << cmd, soci::use(tmp_file_name);

  // Remove temp file
  (void)remove("temp.csv");
}

/*
 * Inserts the file frame into the database using the fallback method.
 * 
 * The data is expected in a vector of tuples frame which is defined as dataset_id, file_id, sample_index, label.
 * It is then inserted into the database using a prepared statement.
 * 
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::fallback_insertion(
    const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame) const {
  soci::session session = storage_database_connection_.get_session();
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";
  for (const auto& frame : file_frame) {
    query += "(" + std::to_string(std::get<0>(frame)) + "," + std::to_string(std::get<1>(frame)) + "," +
             std::to_string(std::get<2>(frame)) + "," + std::to_string(std::get<3>(frame)) + "),";
  }

  // Remove last comma
  query.pop_back();
  session << query;
}

/*
 * Checks if the file is valid for the dataset.
 * 
 * Valid files are defined as files that adhere to the following rules:
 * - The file extension is the same as the data file extension.
 * - The file is not already in the database.
 * - If we are not ignoring the last modified timestamp, the file has been modified since the last check.
 * 
 * @param file_path The path to the file.
 * @param data_file_extension The extension of the data files.
 * @param ignore_last_timestamp If true, the last modified timestamp of the file is ignored.
 * @param timestamp The last modified timestamp of the file.
 * @return True if the file is valid, false otherwise.
 */
bool FileWatcher::check_valid_file(const std::string& file_path, const std::string& data_file_extension,
                                   bool ignore_last_timestamp, int64_t timestamp) {
  const std::string file_extension = file_path.substr(file_path.find_last_of('.'));
  if (file_extension != data_file_extension) {
    return false;
  }
  soci::session session = storage_database_connection_.get_session();

  int64_t file_id = -1;

  session << "SELECT file_id FROM files WHERE path = :file_path", soci::into(file_id), soci::use(file_path);

  if (file_id == -1) {
    if (ignore_last_timestamp) {
      return true;
    }
    return filesystem_wrapper->get_modified_time(file_path) > timestamp;
  }
  return false;
}

/*
 * Updates the files in the database for the given directory.
 * 
 * Iterates over all files in the directory and depending on whether we are multi or single threaded, either handles the 
 * file paths directly or spawns new threads to handle the file paths.
 * 
 * Each thread spawned will handle an equal share of the files in the directory.
 * 
 * @param directory_path The path to the directory.
 * @param timestamp The last modified timestamp of the file.
 */
void FileWatcher::update_files_in_directory(const std::string& directory_path, int64_t timestamp) {
  std::string file_wrapper_config;
  int64_t file_wrapper_type_id;

  soci::session session = storage_database_connection_.get_session();

  session << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type_id), soci::into(file_wrapper_config), soci::use(dataset_id_);
  const auto file_wrapper_type = static_cast<FileWrapperType>(file_wrapper_type_id);

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
      std::atomic<bool> stop_file_watcher = false;
      const FileWatcher watcher(config_, dataset_id_, &stop_file_watcher);
      children.emplace_back(&FileWatcher::handle_file_paths, watcher, file_paths_thread, data_file_extension,
                            file_wrapper_type, timestamp, file_wrapper_config_node);
    }

    for (auto& child : children) {
      child.join();
    }
  }
}

/*
 * Updating the files in the database for the given directory with the last inserted timestamp.
 */
void FileWatcher::seek_dataset() {
  soci::session session = storage_database_connection_.get_session();

  int64_t last_timestamp;

  session << "SELECT last_timestamp FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(last_timestamp), soci::use(dataset_id_);

  update_files_in_directory(dataset_path_, last_timestamp);
}

/*
 * Seeking the dataset and updating the last inserted timestamp.
 */
void FileWatcher::seek() {
  soci::session session = storage_database_connection_.get_session();
  std::string dataset_name;

  session << "SELECT name FROM datasets WHERE dataset_id = :dataset_id", soci::into(dataset_name),
      soci::use(dataset_id_);

  try {
    seek_dataset();

    int64_t last_timestamp;
    session << "SELECT updated_at FROM files WHERE dataset_id = :dataset_id ORDER "
               "BY updated_at DESC LIMIT 1",
        soci::into(last_timestamp), soci::use(dataset_id_);

    if (last_timestamp > 0) {
      session << "UPDATE datasets SET last_timestamp = :last_timestamp WHERE dataset_id = "
                 ":dataset_id",
          soci::use(last_timestamp), soci::use(dataset_id_);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("File watcher failed for dataset {} with error: {}", dataset_name, e.what());
    stop_file_watcher_->store(true);
  }
}

void FileWatcher::run() {
  soci::session session = storage_database_connection_.get_session();

  int64_t file_watcher_interval;
  session << "SELECT file_watcher_interval FROM datasets WHERE dataset_id = :dataset_id",
      soci::into(file_watcher_interval), soci::use(dataset_id_);

  if (file_watcher_interval == 0) {
    throw std::runtime_error("File watcher interval is invalid, does the dataset exist?");
  }

  while (true) {
    try {
      seek();
      SPDLOG_INFO("File watcher for dataset {} is sleeping for {} seconds", dataset_id_, file_watcher_interval);
      SPDLOG_INFO("Current flag value: {}", stop_file_watcher_->load());
      if (stop_file_watcher_->load()) {
        break;
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("File watcher failed: {}", e.what());
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}
