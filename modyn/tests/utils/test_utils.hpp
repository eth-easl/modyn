#ifndef UTILS_H
#define UTILS_H


#include <yaml-cpp/yaml.h>

#include <fstream>

namespace modyn::test {
class TestUtils {
 public:
  static void create_dummy_yaml();
  static void delete_dummy_yaml();
  static YAML::Node get_dummy_config();
};
} // namespace modyn

#endif