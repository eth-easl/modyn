#include "FileWatcher.hpp"
#include "../utils/utils.hpp"
#include <boost/process.hpp>
#include <csignal>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <sstream>

using namespace storage;

volatile sig_atomic_t file_watcher_sigflag = 0;
void file_watcher_signal_handler(int signal) { file_watcher_sigflag = 1; }

void FileWatcher::handle_file_paths(
    std::vector<std::string> file_paths, std::string data_file_extension,
    std::string file_wrapper_type,
    AbstractFilesystemWrapper *filesystem_wrapper, int timestamp) {
  soci::session *sql = this->storage_database_connection->get_session();

  std::vector<std::string> valid_files;
  for (auto const &file_path : file_paths) {
    if (this->checkValidFile(file_path, data_file_extension, false, timestamp,
                             filesystem_wrapper)) {
      valid_files.push_back(file_path);
    }
  }

  if (valid_files.size() > 0) {
    std::string file_path;
    int number_of_samples;
    std::vector<std::tuple<long long, long long, int, int>> file_frame =
        std::vector<std::tuple<long long, long long, int, int>>();
    for (auto const &file_path : valid_files) {
      AbstractFileWrapper *file_wrapper = Utils::get_file_wrapper(
          file_path, file_wrapper_type, this->config, filesystem_wrapper);
      number_of_samples = file_wrapper->get_number_of_samples();

      *sql << "INSERT INTO files (dataset_id, path, number_of_samples, "
              "created_at, updated_at) VALUES (:dataset_id, :path, "
              ":number_of_samples, :created_at, :updated_at)",
          soci::use(this->dataset_id), soci::use(file_path),
          soci::use(number_of_samples),
          soci::use(filesystem_wrapper->get_created_time(file_path)),
          soci::use(filesystem_wrapper->get_modified_time(file_path));

      long long file_id;
      sql->get_last_insert_id("files", file_id);

      SPDLOG_DEBUG("[Process {}] Extracting samples from file {}",
                   boost::this_process::get_id(), file_path);

      std::vector<int> labels = *file_wrapper->get_all_labels();

      std::tuple<long long, long long, int, int> frame;
      int index = 0;
      for (auto const &label : labels) {
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

void FileWatcher::fallback_insertion(
    std::vector<std::tuple<long long, long long, int, int>> file_frame,
    soci::session *sql) {
  // Prepare query
  std::string query =
      "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES ";

  for (auto const &frame : file_frame) {
    query += "(" + std::to_string(std::get<0>(frame)) + "," +
             std::to_string(std::get<1>(frame)) + "," +
             std::to_string(std::get<2>(frame)) + "," +
             std::to_string(std::get<3>(frame)) + "),";
  }

  // Remove last comma
  query.pop_back();
  *sql << query;
}

void FileWatcher::postgres_copy_insertion(
    std::vector<std::tuple<long long, long long, int, int>> file_frame,
    soci::session *sql) {
  std::string table_name = "samples__did" + std::to_string(this->dataset_id);
  std::string table_columns = "(dataset_id,file_id,sample_index,label)";
  std::string cmd =
      "COPY " + table_name + table_columns +
      " FROM STDIN WITH (FORMAT CSV, HEADER FALSE, DELIMITER ',')";

  // Create stringbuffer, dump data into file buffer csv and send to
  // postgresql
  std::stringstream ss;
  for (auto const &frame : file_frame) {
    ss << std::get<0>(frame) << "," << std::get<1>(frame) << ","
       << std::get<2>(frame) << "," << std::get<3>(frame) << "\n";
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

bool FileWatcher::checkValidFile(
    std::string file_path, std::string data_file_extension,
    bool ignore_last_timestamp, int timestamp,
    AbstractFilesystemWrapper *filesystem_wrapper) {
  std::string file_extension =
      file_path.substr(file_path.find_last_of(".") + 1);
  if (file_extension != data_file_extension) {
    return false;
  }
  soci::session *sql = this->storage_database_connection->get_session();

  long long file_id;

  *sql << "SELECT id FROM files WHERE path = :file_path", soci::into(file_id),
      soci::use(file_path);

  if (file_id) {
    if (ignore_last_timestamp) {
      return true;
    }
    return filesystem_wrapper->get_modified_time(file_path) < timestamp;
  }
  return false;
}

void FileWatcher::update_files_in_directory(
    AbstractFilesystemWrapper *filesystem_wrapper, std::string directory_path,
    int timestamp) {
  std::string file_wrapper_config;
  std::string file_wrapper_type;

  soci::session *sql = this->storage_database_connection->get_session();

  *sql << "SELECT file_wrapper_type, file_wrapper_config FROM datasets "
          "WHERE id = :dataset_id",
      soci::into(file_wrapper_type), soci::into(file_wrapper_config),
      soci::use(this->dataset_id);

  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
  std::string data_file_extension =
      file_wrapper_config_node["extension"].as<std::string>();

  std::vector<std::string> file_paths =
      *filesystem_wrapper->list(directory_path, true);

  if (this->disable_multithreading) {
    this->handle_file_paths(file_paths, data_file_extension, file_wrapper_type,
                            filesystem_wrapper, timestamp);
  } else {
    int files_per_thread = file_paths.size() / this->insertion_threads;
    std::vector<boost::process::child> children;
    for (int i = 0; i < this->insertion_threads; i++) {
      int start_index = i * files_per_thread;
      int end_index = start_index + files_per_thread
                          ? i < this->insertion_threads - 1
                          : file_paths.size() - 1;
      std::vector<std::string> file_paths_thread(
          file_paths.begin() + start_index, file_paths.begin() + end_index);
      std::string file_paths_thread_string =
          Utils::joinStringList(file_paths_thread, ",");
      children.push_back(boost::process::child(
          boost::process::search_path("FileWatcher"),
          std::vector<std::string>{
              file_paths_thread_string, std::to_string(this->dataset_id),
              file_wrapper_type, file_wrapper_config, std::to_string(timestamp),
              this->config_path},
          boost::process::std_out > boost::process::null,
          boost::process::std_err > boost::process::null));
    }

    for (int i = 0; i < children.size(); i++) {
      children[i].wait();
    }
  }
}

void FileWatcher::seek_dataset() {
  soci::session *sql = this->storage_database_connection->get_session();

  std::string dataset_path;
  std::string dataset_filesystem_wrapper_type;
  int last_timestamp;

  *sql << "SELECT path, filesystem_wrapper_type, last_timestamp FROM datasets "
          "WHERE id = :dataset_id",
      soci::into(dataset_path), soci::into(dataset_filesystem_wrapper_type),
      soci::into(last_timestamp), soci::use(this->dataset_id);

  AbstractFilesystemWrapper *filesystem_wrapper = Utils::get_filesystem_wrapper(
      dataset_path, dataset_filesystem_wrapper_type);

  if (filesystem_wrapper->exists(dataset_path) &&
      filesystem_wrapper->is_directory(dataset_path)) {
    this->update_files_in_directory(filesystem_wrapper, dataset_path,
                                    last_timestamp);
  } else {
    throw std::runtime_error(
        "Dataset path does not exist or is not a directory.");
  }
}

void FileWatcher::seek() {
  soci::session *sql = this->storage_database_connection->get_session();
  std::string dataset_name;

  *sql << "SELECT name FROM datasets WHERE id = :dataset_id",
      soci::into(dataset_name), soci::use(this->dataset_id);

  try {
    this->seek_dataset();

    int last_timestamp;
    *sql << "SELECT updated_at FROM files WHERE dataset_id = :dataset_id ORDER "
            "BY updated_at DESC LIMIT 1",
        soci::into(last_timestamp), soci::use(this->dataset_id);

    if (last_timestamp > 0) {
      *sql << "UPDATE datasets SET last_timestamp = :last_timestamp WHERE id = "
              ":dataset_id",
          soci::use(last_timestamp), soci::use(this->dataset_id);
    }
  } catch (std::exception &e) {
    SPDLOG_ERROR("Dataset {} was deleted while the file watcher was running. "
                 "Stopping file watcher.",
                 this->dataset_id);
    sql->rollback();
    storage_database_connection->delete_dataset(dataset_name);
  }
}

void FileWatcher::run() {
  std::signal(SIGTERM, file_watcher_signal_handler);

  soci::session *sql = this->storage_database_connection->get_session();

  int file_watcher_interval;
  *sql << "SELECT file_watcher_interval FROM datasets WHERE id = :dataset_id",
      soci::into(file_watcher_interval), soci::use(this->dataset_id);

  if (file_watcher_interval == 0) {
    throw std::runtime_error(
        "File watcher interval is invalid, does the dataset exist?");
  }

  while (true) {
    this->seek();
    if (file_watcher_sigflag) {
      break;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(file_watcher_interval));
  }
}
