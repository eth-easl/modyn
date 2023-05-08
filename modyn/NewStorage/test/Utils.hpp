#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <yaml-cpp/yaml.h>

namespace storage {
class Utils {
public:
  static void create_dummy_yaml();
  static void delete_dummy_yaml();
  static YAML::Node get_dummy_config();
  static YAML::Node get_dummy_file_wrapper_config();
};
} // namespace storage

#endif