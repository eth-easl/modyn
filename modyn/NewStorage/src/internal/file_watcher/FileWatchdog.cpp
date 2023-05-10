#include "FileWatchdog.hpp"
#include "../database/StorageDatabaseConnection.hpp"
#include <soci/soci.h>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#include <filesystem>
#include <spdlog/spdlog.h>

using namespace storage;
namespace bp = boost::process;

volatile sig_atomic_t file_watchdog_sigflag = 0;
void file_watchdog_signal_handler(int signal) { file_watchdog_sigflag = 1; }

void FileWatchdog::start_file_watcher_process(long long dataset_id) {
  // Start a new child process of a FileWatcher
  std::string exec = std::filesystem::current_path() / "executables" /
                     "FileWatcher" / "FileWatcher";
  bp::child subprocess(
      exec, bp::args({this->config_file, std::to_string(dataset_id), "false"}));

  this->file_watcher_processes[dataset_id] =
      std::tuple(std::move(subprocess), 0);
}

void FileWatchdog::stop_file_watcher_process(long long dataset_id) {
  if (this->file_watcher_processes.count(dataset_id) == 1) {
    std::get<0>(this->file_watcher_processes[dataset_id]).terminate();
    SPDLOG_INFO("FileWatcher process for dataset {} stopped", dataset_id);
    std::unordered_map<long long,
                       std::tuple<boost::process::child, int>>::iterator it;
    it = this->file_watcher_processes.find(dataset_id);
    this->file_watcher_processes.erase(it);
  } else {
    throw std::runtime_error("FileWatcher process not found");
  }
}

void FileWatchdog::watch_file_watcher_processes(
    StorageDatabaseConnection *storage_database_connection) {
  soci::session *sql = storage_database_connection->get_session();
  int number_of_datasets = 0;
  *sql << "SELECT COUNT(dataset_id) FROM datasets",
      soci::into(number_of_datasets);
  if (number_of_datasets == 0) {
    // There are no datasets in the database. Stop all FileWatcher processes.
    for (auto const &pair : this->file_watcher_processes) {
      this->stop_file_watcher_process(pair.first);
    }
    return;
  }
  std::vector<long long> dataset_ids =
      std::vector<long long>(number_of_datasets);
  *sql << "SELECT dataset_id FROM datasets", soci::into(dataset_ids);

  long long dataset_id;
  for (auto const &pair : this->file_watcher_processes) {
    dataset_id = pair.first;
    if (std::find(dataset_ids.begin(), dataset_ids.end(), dataset_id) ==
        dataset_ids.end()) {
      // There is a FileWatcher process running for a dataset that was deleted
      // from the database. Stop the process.
      this->stop_file_watcher_process(dataset_id);
    }
  }

  for (auto const &dataset_id : dataset_ids) {
    if (this->file_watcher_processes.find(dataset_id) ==
        this->file_watcher_processes.end()) {
      // There is no FileWatcher process running for this dataset. Start one.
      this->start_file_watcher_process(dataset_id);
    }

    if (std::get<1>(this->file_watcher_processes[dataset_id]) > 3) {
      // There have been more than 3 restart attempts for this process. Stop it.
      this->stop_file_watcher_process(dataset_id);
    } else if (!std::get<0>(this->file_watcher_processes[dataset_id]).running()) {
      // The process is not running. Start it.
      this->start_file_watcher_process(dataset_id);
      std::get<1>(this->file_watcher_processes[dataset_id])++;
    } else {
      // The process is running. Reset the restart attempts counter.
      std::get<1>(this->file_watcher_processes[dataset_id]) = 0;
    }
  }
}

void FileWatchdog::run() {
  std::signal(SIGKILL, file_watchdog_signal_handler);

  StorageDatabaseConnection storage_database_connection =
      StorageDatabaseConnection(this->config);
  storage_database_connection.create_tables();

  SPDLOG_INFO("FileWatchdog running");

  while (true) {
    if (file_watchdog_sigflag) {
      break;
    }
    this->watch_file_watcher_processes(&storage_database_connection);
    // Wait for 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  for (auto &file_watcher_process : this->file_watcher_processes) {
    std::get<0>(file_watcher_process.second).terminate();
  }
}

std::vector<long long> FileWatchdog::get_running_file_watcher_processes() {
  std::vector<long long> running_file_watcher_processes;
  for (auto const &pair : this->file_watcher_processes) {
    if (std::get<0>(pair.second).exit_code() == 383) { // TODO: This is very Unix specific
                                          // (as is most of the c++ code...)
      running_file_watcher_processes.push_back(pair.first);
    }
  }
  return running_file_watcher_processes;
}