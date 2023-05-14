#pragma once

#include <string>

#include "yaml-cpp/yaml.h"

namespace storage {
class Storage {
 private:
  YAML::Node config_;

 public:
  explicit Storage(const std::string& config_file);
  void run();
};
}  // namespace storage
