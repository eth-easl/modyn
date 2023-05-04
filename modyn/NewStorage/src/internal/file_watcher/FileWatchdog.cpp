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

void FileWatchdog::start_file_watcher_process(int dataset_id) {
  // Start a new child process of a FileWatcher
  bp::ipstream out;
  std::atomic<bool> is_running = true;

  // Path to FileWatcher executable
  std::filesystem::path file_watcher_path =
      std::filesystem::current_path() / "FileWatcher";

  bp::child subprocess(bp::search_path(file_watcher_path),
                       bp::args({std::to_string(dataset_id), "false"}),
                       bp::std_out > out);

  this->file_watcher_processes[dataset_id] = std::move(subprocess);
}

void FileWatchdog::stop_file_watcher_process(int dataset_id) {
  if (this->file_watcher_processes[dataset_id]) {
    this->file_watcher_processes[dataset_id].terminate();
  } else {
    throw std::runtime_error("FileWatcher process not found");
  }
}

void FileWatchdog::watch_file_watcher_processes() {
  StorageDatabaseConnection storage_database_connection =
      StorageDatabaseConnection(this->config);
  soci::session *sql = storage_database_connection.get_session();
  std::vector<int> dataset_ids;
  *sql << "SELECT id FROM datasets", soci::into(dataset_ids);
  // TODO: Check if dataset is already being watched or if it was deleted
}

void FileWatchdog::run() {
  std::signal(SIGTERM, file_watchdog_signal_handler);

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
