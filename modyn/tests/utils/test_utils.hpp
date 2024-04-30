#pragma once

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>

namespace modyn::test {
class TestUtils {
 public:
  static void create_dummy_yaml();
  static void delete_dummy_yaml();
  static YAML::Node get_dummy_config();
  static std::string get_tmp_testdir(const std::string& subsdir = "");
};
}  // namespace modyn::test
