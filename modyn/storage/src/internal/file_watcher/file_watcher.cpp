#include "internal/file_watcher/file_watcher.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"

using namespace storage::file_watcher;


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
                                   bool ignore_last_timestamp, int64_t timestamp,
                                   storage::database::StorageDatabaseConnection& storage_database_connection,
                                   const std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper>& filesystem_wrapper) {
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

  soci::session session = storage_database_connection.get_session();

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
  int64_t file_wrapper_type_id = 0;

  soci::session session = storage_database_connection_.get_session();

  session << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type_id), soci::into(file_wrapper_config), soci::use(dataset_id_);
  const auto file_wrapper_type = static_cast<storage::file_wrapper::FileWrapperType>(file_wrapper_type_id);

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  if (!file_wrapper_config_node["file_extension"]) {
    // Check this regularly, as it is a required field and should always be present.
    SPDLOG_ERROR("Config does not contain file_extension");
    stop_file_watcher->store(true);
    return;
  }

  const auto data_file_extension = file_wrapper_config_node["file_extension"].as<std::string>();

  std::vector<std::string> file_paths = filesystem_wrapper->list(directory_path, /*recursive=*/true);

  if (disable_multithreading_) {
    FileWatcher::handle_file_paths(file_paths, data_file_extension, file_wrapper_type, timestamp,
                                   filesystem_wrapper_type_, dataset_id_, file_wrapper_config_node, config_,
                                   sample_dbinsertion_batchsize_, force_fallback_);
  } else {
    const int16_t chunk_size = static_cast<int16_t>(file_paths.size() / insertion_threads_);

    for (int16_t i = 0; i < insertion_threads_; ++i) {
      auto begin = file_paths.begin() + i * chunk_size;
      auto end = (i < insertion_threads_ - 1) ? (begin + chunk_size) : file_paths.end();

      std::vector<std::string> const file_paths_thread(begin, end);

      insertion_thread_pool_[i] = std::thread(
          [this, file_paths_thread, &data_file_extension, &file_wrapper_type, &timestamp, &file_wrapper_config_node]() {
            FileWatcher::handle_file_paths(file_paths_thread, data_file_extension, file_wrapper_type, timestamp,
                                           filesystem_wrapper_type_, dataset_id_, file_wrapper_config_node, config_,
                                           sample_dbinsertion_batchsize_, force_fallback_);
          });
    }

    for (auto& thread : insertion_thread_pool_) {
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
    if (stop_file_watcher->load()) {
      SPDLOG_INFO("File watcher for dataset {} is stopping", dataset_id_);
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}

void FileWatcher::handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                                    const storage::file_wrapper::FileWrapperType& file_wrapper_type, int64_t timestamp,
                                    const storage::filesystem_wrapper::FilesystemWrapperType& filesystem_wrapper_type,
                                    const int64_t dataset_id, const YAML::Node& file_wrapper_config,
                                    const YAML::Node& config, const int64_t sample_dbinsertion_batchsize,
                                    const bool force_fallback) {
  if (file_paths.empty()) {
    return;
  }

  storage::database::StorageDatabaseConnection storage_database_connection(config);
  soci::session session = storage_database_connection.get_session();

  std::vector<std::string> valid_files;
  const std::string& file_path = file_paths.front();
  auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(file_path, filesystem_wrapper_type);

  for (const auto& file_path : file_paths) {
    if (check_valid_file(file_path, data_file_extension, /*ignore_last_timestamp=*/false, timestamp,
                         storage_database_connection, filesystem_wrapper)) {
      valid_files.push_back(file_path);
    }
  }

  if (!valid_files.empty()) {
    std::string const file_path = valid_files.front();
    int64_t number_of_samples;
    std::vector<FileFrame> file_frame(sample_dbinsertion_batchsize);
    auto file_wrapper = storage::file_wrapper::get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config,
                                                                filesystem_wrapper);

    int64_t inserted_samples = 0;
    for (const auto& file_path : valid_files) {
      file_wrapper->set_file_path(file_path);
      number_of_samples = file_wrapper->get_number_of_samples();
      int64_t modified_time = filesystem_wrapper->get_modified_time(file_path);
      session << "INSERT INTO files (dataset_id, path, number_of_samples, "
                 "updated_at) VALUES (:dataset_id, :path, "
                 ":number_of_samples, :updated_at)",
          soci::use(dataset_id), soci::use(file_path), soci::use(number_of_samples), soci::use(modified_time);

      // Check if the insert was successful.
      int64_t file_id;
      if (!session.get_last_insert_id("files", file_id)) {
        // The insert was not successful.
        SPDLOG_ERROR("Failed to insert file into database");
        continue;
      }

      const std::vector<int64_t> labels = file_wrapper->get_all_labels();

      int32_t index = 0;
      for (const auto& label : labels) {
        if (inserted_samples == sample_dbinsertion_batchsize) {
          insert_file_frame(storage_database_connection, file_frame, force_fallback);
          file_frame.clear();
          inserted_samples = 0;
        }
        file_frame.push_back({dataset_id, file_id, index, label});
        index++;
        inserted_samples++;
      }
    }

    if (!file_frame.empty()) {
      // Move the file_frame vector into the insertion function.
      insert_file_frame(storage_database_connection, file_frame, force_fallback);
    }
  }
}

void FileWatcher::insert_file_frame(const storage::database::StorageDatabaseConnection& storage_database_connection,
                                    const std::vector<FileFrame>& file_frame, const bool  /*force_fallback*/) {
  switch (storage_database_connection.get_drivername()) {
    case storage::database::DatabaseDriver::POSTGRESQL:
      postgres_copy_insertion(file_frame, storage_database_connection);
      break;
    case storage::database::DatabaseDriver::SQLITE3:
      fallback_insertion(file_frame, storage_database_connection);
      break;
    default:
      FAIL("Unsupported database driver");
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
void FileWatcher::postgres_copy_insertion(const std::vector<FileFrame>& file_frame,
                                          const storage::database::StorageDatabaseConnection& storage_database_connection) {
  soci::session session = storage_database_connection.get_session();
  int64_t dataset_id = file_frame.front().dataset_id;
  const std::string table_name = fmt::format("samples__did{}", dataset_id);
  const std::string table_columns = "(dataset_id,file_id,sample_index,label)";
  const std::string cmd =
      fmt::format("COPY {}{} FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')", table_name, table_columns);

  // Create stringbuffer, dump data into file buffer csv and send to postgresql
  std::stringstream ss;
  for (const auto& frame : file_frame) {
    ss << fmt::format("{},{},{},{}\n", frame.dataset_id, frame.file_id, frame.index, frame.label);
  }

  // Execute the COPY command using the temporary stream object
  session << cmd;
  session << ss.str();
}

/*
 * Inserts the file frame into the database using the fallback method.
 *
 * The data is expected in a vector of tuples frame which is defined as dataset_id, file_id, sample_index, label.
 * It is then inserted into the database using a prepared statement.
 *
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::fallback_insertion(const std::vector<FileFrame>& file_frame,
                                     const storage::database::StorageDatabaseConnection& storage_database_connection) {
  soci::session session = storage_database_connection.get_session();
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  if (!file_frame.empty()) {
    for (auto frame = file_frame.cbegin(); frame != std::prev(file_frame.cend()); ++frame) {
      query += fmt::format("({},{},{},{}),", frame->dataset_id, frame->file_id, frame->index, frame->label);
    }

    // Add the last tuple without the trailing comma
    const auto& last_frame = file_frame.back();
    query +=
        fmt::format("({},{},{},{})", last_frame.dataset_id, last_frame.file_id, last_frame.index, last_frame.label);

    session << query;
  }
}
