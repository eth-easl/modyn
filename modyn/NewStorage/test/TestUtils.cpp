#include "TestUtils.hpp"

using namespace storage;

void TestUtils::create_dummy_yaml() {
  std::ofstream out("config.yaml");
  out << "test: 1" << std::endl;
  out.close();
}

void TestUtils::delete_dummy_yaml() { std::remove("config.yaml"); }

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
  return config;
}
