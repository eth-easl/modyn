#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <string>
#include <yaml-cpp/yaml.h>

namespace storage {
class Storage {
private:
  YAML::Node config;

public:
  Storage(std::string config_file);
  void run();
};
} // namespace storage

#endif