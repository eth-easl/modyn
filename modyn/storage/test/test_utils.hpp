#ifndef UTILS_H
#define UTILS_H

#include <yaml-cpp/yaml.h>

#include <fstream>

namespace storage::test {
class TestUtils {
 public:
  static void create_dummy_yaml();
  static void delete_dummy_yaml();
  static YAML::Node get_dummy_config();
  static YAML::Node get_dummy_file_wrapper_config();
  static std::string get_dummy_file_wrapper_config_inline();
  static std::string join(const std::vector<std::string>& strings, const std::string& delimiter = "");
};
}  // namespace storage::test

#endif