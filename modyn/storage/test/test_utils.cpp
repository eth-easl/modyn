#include "test_utils.hpp"

using namespace storage::test;

void TestUtils::create_dummy_yaml() {
  std::ofstream out("config.yaml");
  out << "storage:" << std::endl;
  out << "  port: 50042" << std::endl;
  out << "  sample_batch_size: 5" << std::endl;
  out << "  sample_dbinsertion_batchsize: 10" << std::endl;
  out << "  insertion_threads: 1" << std::endl;
  out << "  retrieval_threads: 1" << std::endl;
  out << "  database:" << std::endl;
  out << "    drivername: sqlite3" << std::endl;
  out << "    database: test.db" << std::endl;
  out << "    username: ''" << std::endl;
  out << "    password: ''" << std::endl;
  out << "    host: ''" << std::endl;
  out << "    port: ''" << std::endl;
  out.close();
}

void TestUtils::delete_dummy_yaml() { (void)std::remove("config.yaml"); }

YAML::Node TestUtils::get_dummy_config() {
  YAML::Node config;
  config["storage"]["database"]["drivername"] = "sqlite3";
  config["storage"]["database"]["database"] = "test.db";
  config["storage"]["database"]["username"] = "";
  config["storage"]["database"]["password"] = "";
  config["storage"]["database"]["host"] = "";
  config["storage"]["database"]["port"] = "";
  return config;
}

YAML::Node TestUtils::get_dummy_file_wrapper_config() {
  YAML::Node config;
  config["file_extension"] = ".txt";
  config["label_file_extension"] = ".json";
  config["label_size"] = 1;
  config["record_size"] = 2;
  config["label_index"] = 0;
  config["encoding"] = "utf-8";
  config["validate_file_content"] = false;
  config["ignore_first_line"] = false;
  config["separator"] = ',';
  return config;
}

std::string TestUtils::get_dummy_file_wrapper_config_inline() {
  std::string test_config = R"(
file_extension: ".txt"
label_file_extension: ".lbl"
)";
  return test_config;
}

std::string TestUtils::join(const std::vector<std::string>& strings, const std::string& delimiter) {
  std::string result;
  for (size_t i = 0; i < strings.size(); ++i) {
    result += strings[i];
    if (i != strings.size() - 1) {
      result += delimiter;
    }
  }
  return result;
}