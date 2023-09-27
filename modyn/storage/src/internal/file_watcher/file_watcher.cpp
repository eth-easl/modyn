#include "internal/file_watcher/file_watcher.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <functional>

using namespace storage;

/*
 * Inserts the file frame into the database using the optimized postgresql copy command.
 *
 * The data is expected in a vector of tuples frame which is defined as dataset_id, file_id, sample_index, label.
 * It is then dumped into a csv file buffer and sent to postgresql using the copy command.
 *
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::postgres_copy_insertion(const std::vector<FileFrame>& file_frame) const {
  soci::session session = storage_database_connection_.get_session();
  const std::string table_name = fmt::format("samples__did{}", dataset_id_);
  const std::string table_columns = "(dataset_id,file_id,sample_index,label)";
  const std::string cmd =
      fmt::format("COPY {}{} FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')", table_name, table_columns);

  // Create stringbuffer, dump data into file buffer csv and send to postgresql
  std::stringstream ss;
  for (const auto& frame : file_frame) {
    ss << fmt::format("{},{},{},{}\n", frame.dataset_id, frame.file_id, frame.index, frame.label);
  }

  // Create a temporary stream object and pipe the stringbuffer to it
  std::istringstream is(ss.str());

  // Execute the COPY command using the temporary stream object
  session << cmd, soci::use(is);
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
    const std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>>& file_frame)  // NOLINT (misc-unused-parameters)
    const {
  soci::session session = storage_database_connection_.get_session();
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  if (!file_frame.empty()) {
    for (auto frame = file_frame.cbegin(); frame != std::prev(file_frame.cend()); ++frame) {
      query += fmt::format("({},{},{},{}),", frame->dataset_id, frame->file_id, frame->index, frame->label);
    }

    // Add the last tuple without the trailing comma
    const auto& last_frame = file_frame.back();
    query += fmt::format("({},{},{},{})", last_frame.dataset_id, last_frame.file_id, last_frame.index, last_frame.label);

    session << query;
  }
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
  if (file_path.empty()) {
    return false;
  }
  const std::size_t last_occurence_dot = file_path.find_last_of('.');
  if (last_occurence_dot == std::string::npos) {
    return false;
  }
  const std::string file_extension = file_path.substr(last_occurence_dot);
  if (file_extension != data_file_extension) {
    return false;
  }
  soci::session session = storage_database_connection_.get_session();

  int64_t file_id = 0;
  session << "SELECT file_id FROM files WHERE path = :file_path", soci::into(file_id), soci::use(file_path);

  if (file_id == 0) {
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
  int64_t file_wrapper_type_id = 0;

  soci::session session = storage_database_connection_.get_session();

  session << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type_id), soci::into(file_wrapper_config), soci::use(dataset_id_);
  const auto file_wrapper_type = static_cast<FileWrapperType>(file_wrapper_type_id);

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
  const auto data_file_extension = file_wrapper_config_node["file_extension"].as<std::string>();

  std::vector<std::string> file_paths = filesystem_wrapper->list(directory_path, /*recursive=*/true);

  if (disable_multithreading_) {
    FileWatcher.handle_file_paths(file_paths, data_file_extension, file_wrapper_type, timestamp, file_wrapper_config_node);
  } else {
    const size_t chunk_size = file_paths.size() / thread_pool.size();

    for (size_t i = 0; i < thread_pool.size(); ++i) {
      auto begin = file_paths.begin() + i * chunk_size;
      auto end = (i < thread_pool.size() - 1) ? (begin + chunk_size) : file_paths.end();

      std::vector<std::string> file_paths_thread(begin, end);

      SPDLOG_INFO("File watcher thread {} will handle {} files", i, file_paths_thread.size());
      std::function<void()> task = std::move([this, file_paths_thread, &data_file_extension, &file_wrapper_type, &timestamp,
                                              &file_wrapper_config_node, &config_]() mutable {
        FileWatcher.handle_file_paths(file_paths_thread, data_file_extension, file_wrapper_type, timestamp,
                                        file_wrapper_config_node, config_);
      });

      tasks.push_back(task);
      SPDLOG_INFO("File watcher thread {} started", i);
    }

    // join all threads
    for (auto& thread : thread_pool) {
      thread.join();
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
}

void FileWatcher::run() {
  soci::session session = storage_database_connection_.get_session();

  int64_t file_watcher_interval;
  session << "SELECT file_watcher_interval FROM datasets WHERE dataset_id = :dataset_id",
      soci::into(file_watcher_interval), soci::use(dataset_id_);

  while (true) {
    seek();
    if (stop_file_watcher_->load()) {
      SPDLOG_INFO("File watcher for dataset {} is stopping", dataset_id_);
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}
