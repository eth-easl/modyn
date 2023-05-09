#include "FileWatchdog.hpp"
#include "../database/StorageDatabaseConnection.hpp"
#include <soci/soci.h>
#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#include <filesystem>

using namespace storage;
namespace bp = boost::process;

volatile sig_atomic_t file_watchdog_sigflag = 0;
void file_watchdog_signal_handler(int signal) { file_watchdog_sigflag = 1; }

void FileWatchdog::start_file_watcher_process(long long dataset_id) {
  // Start a new child process of a FileWatcher
  bp::child subprocess(
      bp::search_path("./executables/FileWatcher/FileWatcher"),
      bp::args({this->config_file, std::to_string(dataset_id), "false"}));

  this->file_watcher_processes[dataset_id] = std::move(subprocess);
  this->file_watcher_process_restart_attempts[dataset_id] = 0;
}

void FileWatchdog::stop_file_watcher_process(long long dataset_id) {
  if (this->file_watcher_processes[dataset_id]) {
    this->file_watcher_processes[dataset_id].terminate();
    this->file_watcher_processes.erase(dataset_id);
    this->file_watcher_process_restart_attempts.erase(dataset_id);
  } else {
    throw std::runtime_error("FileWatcher process not found");
  }
}

void FileWatchdog::watch_file_watcher_processes() {
  StorageDatabaseConnection storage_database_connection =
      StorageDatabaseConnection(this->config);
  soci::session *sql = storage_database_connection.get_session();
  std::vector<long long> dataset_ids;
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

    if (this->file_watcher_process_restart_attempts[dataset_id] > 3) {
      // There have been more than 3 restart attempts for this process. Stop it.
      this->stop_file_watcher_process(dataset_id);
    } else if (!this->file_watcher_processes[dataset_id].running()) {
      // The process is not running. Start it.
      this->start_file_watcher_process(dataset_id);
      this->file_watcher_process_restart_attempts[dataset_id]++;
    } else {
      // The process is running. Reset the restart attempts counter.
      this->file_watcher_process_restart_attempts[dataset_id] = 0;
    }
  }
}

void FileWatchdog::run() {
  std::signal(SIGKILL, file_watchdog_signal_handler);

  while (true) {
    if (file_watchdog_sigflag) {
      break;
    }
    this->watch_file_watcher_processes();
    // Wait for 3 seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }
  for (auto &file_watcher_process : this->file_watcher_processes) {
    file_watcher_process.second.terminate();
  }
}
