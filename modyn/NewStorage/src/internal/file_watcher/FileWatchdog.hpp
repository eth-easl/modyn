#ifndef FILE_WATCHDOG_HPP
#define FILE_WATCHDOG_HPP

#include "FileWatcher.hpp"
#include <boost/process.hpp>
#include <map>
#include <tuple>
#include <yaml-cpp/yaml.h>

namespace storage {
class FileWatchdog {
private:
  YAML::Node config;
  std::string config_file;
  std::map<long long, boost::process::child> file_watcher_processes;
  std::map<long long, int> file_watcher_process_restart_attempts;
  void watch_file_watcher_processes();
  void start_file_watcher_process(long long dataset_id);
  void stop_file_watcher_process(long long dataset_id);

public:
  FileWatchdog(std::string config_file) {
    this->config_file = config_file;
    this->config = YAML::LoadFile(config_file);
    this->file_watcher_processes = std::map<long long, boost::process::child>();
    this->file_watcher_process_restart_attempts = std::map<long long, int>();
  }
  void run();
};
} // namespace storage

#endif