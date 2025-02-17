#include "storage_test_utils.hpp"

using namespace modyn::storage;

YAML::Node StorageTestUtils::get_dummy_file_wrapper_config(const std::string& byteorder) {
  YAML::Node config;
  config["byteorder"] = byteorder;
  config["file_extension"] = ".txt";
  config["label_file_extension"] = ".json";
  config["label_size"] = 2;
  config["record_size"] = 4;
  config["label_index"] = 0;
  config["encoding"] = "utf-8";
  config["has_labels"] = true;
  config["validate_file_content"] = false;
  config["ignore_first_line"] = true;
  config["separator"] = ',';
  return config;
}

std::string StorageTestUtils::get_dummy_file_wrapper_config_inline(const std::string& byteorder) {
  std::string test_config = R"(
byteorder: ")" + byteorder + R"("
file_extension: ".txt"
label_file_extension: ".lbl"
label_size: 1
record_size: 2
label_index: 0
has_labels: true
encoding: "utf-8"
validate_file_content: false
ignore_first_line: false
separator: ','
)";
  return test_config;
}
