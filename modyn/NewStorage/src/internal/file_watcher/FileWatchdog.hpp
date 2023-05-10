#ifndef FILE_WATCHDOG_HPP
#define FILE_WATCHDOG_HPP

#include "FileWatcher.hpp"
#include <boost/process.hpp>
#include <map>
#include <tuple>
#include <yaml-cpp/yaml.h>
#include "../database/StorageDatabaseConnection.hpp"

namespace storage {
class FileWatchdog {
private:
  YAML::Node config;
  std::string config_file;
  std::unordered_map<long long, std::tuple<boost::process::child, int>> file_watcher_processes;
  void start_file_watcher_process(long long dataset_id);
  void stop_file_watcher_process(long long dataset_id);

public:
  FileWatchdog(std::string config_file) {
    this->config_file = config_file;
    this->config = YAML::LoadFile(config_file);
    this->file_watcher_processes = std::unordered_map<long long, std::tuple<boost::process::child, int>>();
  }
  void watch_file_watcher_processes(StorageDatabaseConnection *storage_database_connection);
  void run();
  std::vector<long long> get_running_file_watcher_processes();
};
} // namespace storage

#endif