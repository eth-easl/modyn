#pragma once

#include <string>

#include "internal/file_watcher/file_watchdog.hpp"
#include "yaml-cpp/yaml.h"

namespace storage {
class Storage {
 private:
  YAML::Node config_;

 public:
  explicit Storage(const std::string& config_file) { config_ = YAML::LoadFile(config_file); }
  void run();
};
}  // namespace storage
