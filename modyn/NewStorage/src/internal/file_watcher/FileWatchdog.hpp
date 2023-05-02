#include "FileWatcher.hpp"
#include <boost/process.hpp>
#include <map>
#include <tuple>
#include <yaml-cpp/yaml.h>

namespace storage {
class FileWatchdog {
private:
  YAML::Node config;
  std::map<int, std::tuple<boost::process::child, int>> file_watcher_processes;
  void watch_file_watcher_processes();
  void start_file_watcher_process(int dataset_id);
  void stop_file_watcher_process(int dataset_id);
  std::atomic<bool> is_running;

public:
  FileWatchdog(YAML::Node config) {
    this->config = config;
    this->file_watcher_processes =
        std::map<int, std::tuple<boost::process::child, int>>();
    this->is_running = true;
  }
  void run();
};
} // namespace storage