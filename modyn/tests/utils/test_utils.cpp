#include "test_utils.hpp"

using namespace modyn::test;

void TestUtils::create_dummy_yaml() {
  std::ofstream out("config.yaml");
  out << "storage:" << '\n';
  out << "  port: 50042" << '\n';
  out << "  sample_batch_size: 5" << '\n';
  out << "  sample_dbinsertion_batchsize: 10" << '\n';
  out << "  insertion_threads: 1" << '\n';
  out << "  retrieval_threads: 1" << '\n';
  out << "  database:" << '\n';
  out << "    drivername: sqlite3" << '\n';
  out << "    database: test.db" << '\n';
  out << "    username: ''" << '\n';
  out << "    password: ''" << '\n';
  out << "    host: ''" << '\n';
  out << "    port: ''" << '\n';
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