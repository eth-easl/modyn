#include "internal/file_watcher/file_watcher.hpp"

#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <spdlog/spdlog.h>

#include <csignal>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"

using namespace modyn::storage;

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
bool FileWatcher::check_valid_file(
    const std::string& file_path, const std::string& data_file_extension, bool ignore_last_timestamp, int64_t timestamp,
    StorageDatabaseConnection& storage_database_connection,
    const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
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
  int64_t file_wrapper_type_id = -1;

  soci::session session = storage_database_connection_.get_session();

  session << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(file_wrapper_type_id), soci::into(file_wrapper_config), soci::use(dataset_id_);

  if (file_wrapper_type_id == -1) {
    SPDLOG_ERROR("Failed to get file wrapper type");
    *stop_file_watcher = true;
    return;
  }

  const auto file_wrapper_type = static_cast<FileWrapperType>(file_wrapper_type_id);

  if (file_wrapper_config.empty()) {
    SPDLOG_ERROR("Failed to get file wrapper config");
    *stop_file_watcher = true;
    return;
  }

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  if (!file_wrapper_config_node["file_extension"]) {
    // Check this regularly, as it is a required field and should always be present.
    SPDLOG_ERROR("Config does not contain file_extension");
    *stop_file_watcher = true;
    return;
  }

  const auto data_file_extension = file_wrapper_config_node["file_extension"].as<std::string>();

  std::vector<std::string> file_paths = filesystem_wrapper->list(directory_path, /*recursive=*/true);

  if (disable_multithreading_) {
    FileWatcher::handle_file_paths(file_paths, data_file_extension, file_wrapper_type, timestamp,
                                   filesystem_wrapper_type_, dataset_id_, file_wrapper_config_node, config_,
                                   sample_dbinsertion_batchsize_, force_fallback_);
  } else {
    const auto chunk_size = static_cast<int16_t>(file_paths.size() / insertion_threads_);

    for (int16_t i = 0; i < insertion_threads_; ++i) {
      auto begin = file_paths.begin() + static_cast<long long>(i * chunk_size);  // NOLINT google-runtime-int
      auto end = (i < insertion_threads_ - 1) ? (begin + chunk_size) : file_paths.end();

      const std::vector<std::string> file_paths_thread(begin, end);

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

  int64_t last_timestamp = -1;

  session << "SELECT last_timestamp FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(last_timestamp), soci::use(dataset_id_);

  try {
    update_files_in_directory(dataset_path_, last_timestamp);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error while updating files in directory: {}", e.what());
  }
}

/*
 * Seeking the dataset and updating the last inserted timestamp.
 */
void FileWatcher::seek() {
  soci::session session = storage_database_connection_.get_session();

  seek_dataset();

  int64_t last_timestamp = -1;
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

  int64_t file_watcher_interval = -1;
  session << "SELECT file_watcher_interval FROM datasets WHERE dataset_id = :dataset_id",
      soci::into(file_watcher_interval), soci::use(dataset_id_);

  if (file_watcher_interval == -1) {
    SPDLOG_ERROR("Failed to get file watcher interval");
    *stop_file_watcher = true;
    return;
  }

  while (true) {
    try {
      seek();
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error while seeking dataset: {}", e.what());
      stop_file_watcher->store(true);
    }
    if (stop_file_watcher->load()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(file_watcher_interval));
  }
}

void FileWatcher::handle_file_paths(const std::vector<std::string>& file_paths, const std::string& data_file_extension,
                                    const FileWrapperType& file_wrapper_type, int64_t timestamp,
                                    const FilesystemWrapperType& filesystem_wrapper_type,
                                    const int64_t dataset_id, const YAML::Node& file_wrapper_config,
                                    const YAML::Node& config, const int64_t sample_dbinsertion_batchsize,
                                    const bool force_fallback) {
  if (file_paths.empty()) {
    return;
  }

  StorageDatabaseConnection storage_database_connection(config);
  soci::session session = storage_database_connection.get_session();  // NOLINT misc-const-correctness

  std::vector<std::string> valid_files;
  const std::string& file_path = file_paths.front();
  auto filesystem_wrapper = get_filesystem_wrapper(file_path, filesystem_wrapper_type);

  for (const auto& file_path : file_paths) {
    if (check_valid_file(file_path, data_file_extension, /*ignore_last_timestamp=*/false, timestamp,
                         storage_database_connection, filesystem_wrapper)) {
      valid_files.push_back(file_path);
    }
  }

  if (!valid_files.empty()) {
    const std::string file_path = valid_files.front();
    std::vector<FileFrame> file_frame = {};
    auto file_wrapper =
        get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config, filesystem_wrapper);

    int64_t inserted_samples = 0;
    for (const auto& file_path : valid_files) {
      file_wrapper->set_file_path(file_path);
      int64_t file_id =  // NOLINT misc-const-correctness
          insert_file(file_path, dataset_id, storage_database_connection, filesystem_wrapper, file_wrapper);

      if (file_id == -1) {
        SPDLOG_ERROR("Failed to insert file into database");
        continue;
      }

      const std::vector<int64_t> labels = file_wrapper->get_all_labels();

      int32_t index = 0;
      for (const auto& label : labels) {
        if (inserted_samples == sample_dbinsertion_batchsize) {
          insert_file_frame(storage_database_connection, file_frame, dataset_id, force_fallback);
          file_frame.clear();
          inserted_samples = 0;
        }
        file_frame.push_back({file_id, index, label});
        index++;
        inserted_samples++;
      }
    }

    if (!file_frame.empty()) {
      // Move the file_frame vector into the insertion function.
      insert_file_frame(storage_database_connection, file_frame, dataset_id, force_fallback);
    }
  }
}

int64_t FileWatcher::insert_file(
    const std::string& file_path, const int64_t dataset_id,
    const StorageDatabaseConnection& storage_database_connection,
    const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper,
    const std::unique_ptr<FileWrapper>& file_wrapper) {
  int64_t number_of_samples = 0;
  number_of_samples = file_wrapper->get_number_of_samples();
  int64_t modified_time = filesystem_wrapper->get_modified_time(file_path);
  int64_t file_id = -1;

  // soci::session::get_last_insert_id() is not supported by postgresql, so we need to use a different query.
  if (storage_database_connection.get_drivername() == DatabaseDriver::SQLITE3) {
    soci::session session = storage_database_connection.get_session();
    session << "INSERT INTO files (dataset_id, path, number_of_samples, "
               "updated_at) VALUES (:dataset_id, :path, "
               ":updated_at, :number_of_samples)",
        soci::use(dataset_id), soci::use(file_path), soci::use(modified_time), soci::use(number_of_samples);

    // Check if the insert was successful.
    static_assert(sizeof(long long) == sizeof(int64_t));  // NOLINT google-runtime-int
    long long inner_file_id = -1;                         // NOLINT google-runtime-int
    if (!session.get_last_insert_id("files", inner_file_id)) {
      SPDLOG_ERROR("Failed to insert file into database");
      return -1;
    }
    file_id = static_cast<int64_t>(inner_file_id);
  } else if (storage_database_connection.get_drivername() == DatabaseDriver::POSTGRESQL) {
    soci::session session = storage_database_connection.get_session();
    session << "INSERT INTO files (dataset_id, path, number_of_samples, "
               "updated_at) VALUES (:dataset_id, :path, "
               ":updated_at, :number_of_samples) RETURNING file_id",
        soci::use(dataset_id), soci::use(file_path), soci::use(modified_time), soci::use(number_of_samples),
        soci::into(file_id);

    if (file_id == -1) {
      // The insert was not successful.
      SPDLOG_ERROR("Failed to insert file into database");
      return -1;
    }
  }
  return file_id;
}

void FileWatcher::insert_file_frame(const StorageDatabaseConnection& storage_database_connection,
                                    const std::vector<FileFrame>& file_frame, const int64_t dataset_id,
                                    const bool /*force_fallback*/) {
  switch (storage_database_connection.get_drivername()) {
    case DatabaseDriver::POSTGRESQL:
      postgres_copy_insertion(file_frame, storage_database_connection, dataset_id);
      break;
    case DatabaseDriver::SQLITE3:
      fallback_insertion(file_frame, storage_database_connection, dataset_id);
      break;
    default:
      FAIL("Unsupported database driver");
  }
}

/*
 * Inserts the file frame into the database using the optimized postgresql copy command.
 *
 * The data is expected in a vector of FileFrame which is defined as file_id, sample_index, label.
 *
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::postgres_copy_insertion(
    const std::vector<FileFrame>& file_frame,
    const StorageDatabaseConnection& storage_database_connection, const int64_t dataset_id) {
  soci::session session = storage_database_connection.get_session();
  auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session.get_backend());
  PGconn* conn = postgresql_session_backend->conn_;

  std::string copy_query =  // NOLINT misc-const-correctness
      fmt::format("COPY samples(dataset_id,file_id,sample_index,label) FROM STDIN WITH (DELIMITER ',', FORMAT CSV)");

  PQexec(conn, copy_query.c_str());
  // put the data into the buffer
  std::stringstream ss;
  for (const auto& frame : file_frame) {
    ss << fmt::format("{},{},{},{}\n", dataset_id, frame.file_id, frame.index, frame.label);
  }

  PQputline(conn, ss.str().c_str());
  PQputline(conn, "\\.\n");
  PQendcopy(conn);
}

/*
 * Inserts the file frame into the database using the fallback method.
 *
 * The data is expected in a vector of FileFrame structs which is defined as file_id, sample_index, label.
 * It is then inserted into the database using a prepared statement.
 *
 * @param file_frame The file frame to be inserted.
 */
void FileWatcher::fallback_insertion(const std::vector<FileFrame>& file_frame,
                                     const StorageDatabaseConnection& storage_database_connection,
                                     const int64_t dataset_id) {
  soci::session session = storage_database_connection.get_session();
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  if (!file_frame.empty()) {
    for (auto frame = file_frame.cbegin(); frame != std::prev(file_frame.cend()); ++frame) {
      query += fmt::format("({},{},{},{}),", dataset_id, frame->file_id, frame->index, frame->label);
    }

    // Add the last frame without a comma
    const auto& last_frame = file_frame.back();
    query += fmt::format("({},{},{},{})", dataset_id, last_frame.file_id, last_frame.index, last_frame.label);

    session << query;
  }
}
