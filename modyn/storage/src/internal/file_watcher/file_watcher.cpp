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
 * Checks if the file is to be inserted into the database.
 *
 * Files to be inserted into the database are defined as files that adhere to the following rules:
 * - The file extension is the same as the data file extension.
 * - The file is not already in the database.
 * - If we are not ignoring the last modified timestamp, the file has been modified since the last check.
 */
bool FileWatcher::check_file_for_insertion(const std::string& file_path, const std::string& data_file_extension,
                                           bool ignore_last_timestamp, int64_t timestamp,
                                           const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper,
                                           soci::session& session) {
  if (file_path.empty()) {
    return false;
  }
  const std::string file_extension = std::filesystem::path(file_path).extension().string();
  if (file_extension != data_file_extension) {
    return false;
  }

  int64_t file_id = -1;
  session << "SELECT file_id FROM files WHERE path = :file_path", soci::into(file_id), soci::use(file_path);

  if (file_id == -1) {
    if (ignore_last_timestamp) {
      return true;
    }
    try {
      SPDLOG_INFO(fmt::format("Modified time of {} is {}, timestamp is {}", file_path,
                              filesystem_wrapper->get_modified_time(file_path), timestamp));
      return filesystem_wrapper->get_modified_time(file_path) > timestamp;
    } catch (const std::exception& e) {
      SPDLOG_ERROR(fmt::format(
          "Error while checking modified time of file {}. It could be that a deletion request is currently running: {}",
          file_path, e.what()));
      return false;
    }
  }
  return false;
}

/*
 * Searches for new files in the given directory and updates the files in the database.
 *
 * Iterates over all files in the directory and depending on whether we are multi or single threaded, either handles the
 * file paths directly or spawns new threads to handle the file paths.
 *
 * Each thread spawned will handle an equal share of the files in the directory.
 */
void FileWatcher::search_for_new_files_in_directory(const std::string& directory_path, int64_t timestamp) {
  std::vector<std::string> file_paths = filesystem_wrapper->list(directory_path, /*recursive=*/true);

  if (disable_multithreading_) {
    std::atomic<bool> exception_thrown(false);
    FileWatcher::handle_file_paths(file_paths, data_file_extension_, file_wrapper_type_, timestamp,
                                   filesystem_wrapper_type_, dataset_id_, file_wrapper_config_node_, config_,
                                   sample_dbinsertion_batchsize_, force_fallback_, exception_thrown);
    if (exception_thrown.load()) {
      *stop_file_watcher = true;
    }
  } else {
    const auto chunk_size = static_cast<int16_t>(file_paths.size() / insertion_threads_);

    for (int16_t i = 0; i < insertion_threads_; ++i) {
      auto begin = file_paths.begin() + static_cast<int32_t>(i * chunk_size);
      auto end = (i < insertion_threads_ - 1) ? (begin + chunk_size) : file_paths.end();

      const std::vector<std::string> file_paths_thread(begin, end);

      insertion_thread_exceptions_[i].store(false);

      insertion_thread_pool_[i] = std::thread([this, file_paths_thread, &timestamp, &i]() {
        FileWatcher::handle_file_paths(file_paths_thread, data_file_extension_, file_wrapper_type_, timestamp,
                                       filesystem_wrapper_type_, dataset_id_, file_wrapper_config_node_, config_,
                                       sample_dbinsertion_batchsize_, force_fallback_, insertion_thread_exceptions_[i]);
      });
    }

    int index = 0;
    for (auto& thread : insertion_thread_pool_) {
      // handle if any thread throws an exception
      if (insertion_thread_exceptions_[index].load()) {
        *stop_file_watcher = true;
        break;
      }
      index++;
      thread.join();
    }
  }
}

/*
 * Updating the files in the database for the given directory with the last inserted timestamp.
 */
void FileWatcher::seek_dataset(soci::session& session) {
  int64_t last_timestamp = -1;

  session << "SELECT last_timestamp FROM datasets "
             "WHERE dataset_id = :dataset_id",
      soci::into(last_timestamp), soci::use(dataset_id_);

  search_for_new_files_in_directory(dataset_path_, last_timestamp);
}

/*
 * Seeking the dataset and updating the last inserted timestamp.
 */
void FileWatcher::seek(soci::session& session) {
  seek_dataset(session);

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
      seek(session);
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
                                    const FilesystemWrapperType& filesystem_wrapper_type, const int64_t dataset_id,
                                    const YAML::Node& file_wrapper_config, const YAML::Node& config,
                                    const int64_t sample_dbinsertion_batchsize, const bool force_fallback,
                                    std::atomic<bool>& exception_thrown) {
  if (file_paths.empty()) {
    return;
  }

  try {
    const StorageDatabaseConnection storage_database_connection(config);
    soci::session session = storage_database_connection.get_session();

    std::vector<std::string> files_for_insertion;
    auto filesystem_wrapper = get_filesystem_wrapper(filesystem_wrapper_type);

    int ignore_last_timestamp = 0;
    session << "SELECT ignore_last_timestamp FROM datasets WHERE dataset_id = :dataset_id",
        soci::into(ignore_last_timestamp), soci::use(dataset_id);

    std::copy_if(file_paths.begin(), file_paths.end(), std::back_inserter(files_for_insertion),
                 [&data_file_extension, &timestamp, &session, &filesystem_wrapper,
                  &ignore_last_timestamp](const std::string& file_path) {
                   return check_file_for_insertion(file_path, data_file_extension,
                                                   static_cast<bool>(ignore_last_timestamp), timestamp,
                                                   filesystem_wrapper, session);
                 });

    if (!files_for_insertion.empty()) {
      DatabaseDriver database_driver = storage_database_connection.get_drivername();
      handle_files_for_insertion(files_for_insertion, file_wrapper_type, dataset_id, file_wrapper_config,
                                 sample_dbinsertion_batchsize, force_fallback, session, database_driver,
                                 filesystem_wrapper);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error while handling file paths: {}", e.what());
    exception_thrown.store(true);
  }
}

void FileWatcher::handle_files_for_insertion(std::vector<std::string>& files_for_insertion,
                                             const FileWrapperType& file_wrapper_type, const int64_t dataset_id,
                                             const YAML::Node& file_wrapper_config,
                                             const int64_t sample_dbinsertion_batchsize, const bool force_fallback,
                                             soci::session& session, DatabaseDriver& database_driver,
                                             const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
  const std::string file_path = files_for_insertion.front();
  std::vector<FileFrame> file_samples = {};
  auto file_wrapper = get_file_wrapper(file_path, file_wrapper_type, file_wrapper_config, filesystem_wrapper);

  int64_t current_file_samples_to_be_inserted = 0;
  for (const auto& file_path : files_for_insertion) {
    file_wrapper->set_file_path(file_path);
    // TODO(MaxiBoether): isn't this batched in Python?
    const int64_t file_id =
        insert_file(file_path, dataset_id, filesystem_wrapper, file_wrapper, session, database_driver);

    if (file_id == -1) {
      SPDLOG_ERROR("Failed to insert file into database");
      continue;
    }

    const std::vector<int64_t> labels = file_wrapper->get_all_labels();

    int32_t index = 0;
    for (const auto& label : labels) {
      if (current_file_samples_to_be_inserted == sample_dbinsertion_batchsize) {
        insert_file_samples(file_samples, dataset_id, force_fallback, session, database_driver);
        file_samples.clear();
        current_file_samples_to_be_inserted = 0;
      }
      file_samples.push_back({file_id, index, label});
      index++;
      current_file_samples_to_be_inserted++;
    }
  }

  if (!file_samples.empty()) {
    insert_file_samples(file_samples, dataset_id, force_fallback, session, database_driver);
  }
}

int64_t FileWatcher::insert_file(const std::string& file_path, const int64_t dataset_id,
                                 const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper,
                                 const std::unique_ptr<FileWrapper>& file_wrapper, soci::session& session,
                                 DatabaseDriver& database_driver) {
  uint64_t number_of_samples = 0;
  number_of_samples = file_wrapper->get_number_of_samples();
  const int64_t modified_time = filesystem_wrapper->get_modified_time(file_path);
  int64_t file_id = -1;

  // soci::session::get_last_insert_id() is not supported by postgresql, so we need to use a different query.
  if (database_driver == DatabaseDriver::SQLITE3) {
    file_id = insert_file(file_path, dataset_id, session, number_of_samples, modified_time);
  } else if (database_driver == DatabaseDriver::POSTGRESQL) {
    file_id = insert_file_using_returning_statement(file_path, dataset_id, session, number_of_samples, modified_time);
  }
  return file_id;
}

int64_t FileWatcher::insert_file(const std::string& file_path, const int64_t dataset_id, soci::session& session,
                                 uint64_t number_of_samples, int64_t modified_time) {
  session << "INSERT INTO files (dataset_id, path, number_of_samples, "
             "updated_at) VALUES (:dataset_id, :path, "
             ":updated_at, :number_of_samples)",
      soci::use(dataset_id), soci::use(file_path), soci::use(modified_time), soci::use(number_of_samples);

  long long file_id = -1;  // NOLINT google-runtime-int (Linux otherwise complains about the following call)
  if (!session.get_last_insert_id("files", file_id)) {
    SPDLOG_ERROR("Failed to insert file into database");
    return -1;
  }
  return file_id;
}

int64_t FileWatcher::insert_file_using_returning_statement(const std::string& file_path, const int64_t dataset_id,
                                                           soci::session& session, uint64_t number_of_samples,
                                                           int64_t modified_time) {
  SPDLOG_INFO(
      fmt::format("Inserting file {} with {} samples for dataset {}", file_path, number_of_samples, dataset_id));
  int64_t file_id = -1;
  session << "INSERT INTO files (dataset_id, path, number_of_samples, "
             "updated_at) VALUES (:dataset_id, :path, "
             ":number_of_samples, :updated_at) RETURNING file_id",
      soci::use(dataset_id), soci::use(file_path), soci::use(number_of_samples), soci::use(modified_time),
      soci::into(file_id);
  SPDLOG_INFO(fmt::format("Inserted file {} into file ID {}", file_path, file_id));

  if (file_id == -1) {
    SPDLOG_ERROR("Failed to insert file into database");
    return -1;
  }
  return file_id;
}

void FileWatcher::insert_file_samples(const std::vector<FileFrame>& file_samples, const int64_t dataset_id,
                                      const bool force_fallback, soci::session& session,
                                      DatabaseDriver& database_driver) {
  if (force_fallback) {
    fallback_insertion(file_samples, dataset_id, session);
  } else {
    switch (database_driver) {
      case DatabaseDriver::POSTGRESQL:
        postgres_copy_insertion(file_samples, dataset_id, session);
        break;
      case DatabaseDriver::SQLITE3:
        fallback_insertion(file_samples, dataset_id, session);
        break;
      default:
        FAIL("Unsupported database driver");
    }
  }
}

/*
 * Inserts the file frame into the database using the optimized postgresql copy command.
 *
 * The data is expected in a vector of FileFrame which is defined as file_id, sample_index, label.
 */
void FileWatcher::postgres_copy_insertion(const std::vector<FileFrame>& file_samples, const int64_t dataset_id,
                                          soci::session& session) {
  SPDLOG_INFO(fmt::format("Doing copy insertion for {} samples", file_samples.size()));
  auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session.get_backend());
  PGconn* conn = postgresql_session_backend->conn_;

  const std::string copy_query =
      "COPY samples(dataset_id,file_id,sample_index,label) FROM STDIN WITH (DELIMITER ',', FORMAT CSV)";

  PQexec(conn, copy_query.c_str());
  // put the data into the buffer
  std::stringstream ss;
  for (const auto& frame : file_samples) {
    ss << fmt::format("{},{},{},{}\n", dataset_id, frame.file_id, frame.index, frame.label);
  }

  PQputline(conn, ss.str().c_str());
  PQputline(conn, "\\.\n");  // Note the application must explicitly send the two characters "\." on a final line to
                             // indicate to the backend that it has finished sending its data.
                             // https://web.mit.edu/cygwin/cygwin_v1.3.2/usr/doc/postgresql-7.1.2/html/libpq-copy.html
  PQendcopy(conn);
}

/*
 * Inserts the file frame into the database using the fallback method.
 *
 * The data is expected in a vector of FileFrame structs which is defined as file_id, sample_index, label.
 * It is then inserted into the database using a prepared statement.
 */
void FileWatcher::fallback_insertion(const std::vector<FileFrame>& file_samples, const int64_t dataset_id,
                                     soci::session& session) {
  // Prepare query
  std::string query = "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  if (!file_samples.empty()) {
    for (auto frame = file_samples.cbegin(); frame != std::prev(file_samples.cend()); ++frame) {
      query += fmt::format("({},{},{},{}),", dataset_id, frame->file_id, frame->index, frame->label);
    }

    // Add the last frame without a comma
    const auto& last_frame = file_samples.back();
    query += fmt::format("({},{},{},{})", dataset_id, last_frame.file_id, last_frame.index, last_frame.label);

    session << query;
  }
}
