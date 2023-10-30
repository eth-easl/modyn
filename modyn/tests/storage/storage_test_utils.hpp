#ifndef UTILS_H
#define UTILS_H

#include <yaml-cpp/yaml.h>

#include <fstream>

namespace modyn::storage {
class StorageTestUtils {
 public:
  static YAML::Node get_dummy_file_wrapper_config();
  static std::string get_dummy_file_wrapper_config_inline();
};
}  // namespace modyn::storage

#endif