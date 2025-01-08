#include <spdlog/spdlog.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>

#include "modyn/utils/utils.hpp"
#include "storage_server.hpp"

using namespace modyn::storage;

void setup_logger() { spdlog::set_pattern("[%Y-%m-%d:%H:%M:%S] [%s:%#] [%l] [p%P:t%t] %v"); }

int main(int argc, char* argv[]) {
  /* Entrypoint for the storage service. */
  setup_logger();

  argparse::ArgumentParser parser("Modyn Storage");
  parser.add_argument("config").help("Modyn infrastructure configuration file");
  parser.parse_args(argc, argv);

  const auto config_file = parser.get<std::string>("config");

  ASSERT(std::filesystem::exists(config_file), "Config file does not exist.");

  // Verify that the config file exists and is readable.
  const YAML::Node config = YAML::LoadFile(config_file);

  SPDLOG_INFO("Initializing storage.");
  StorageServer storage(config_file);
  SPDLOG_INFO("Starting storage.");
  storage.run();

  SPDLOG_INFO("Storage returned, exiting.");

  return 0;
}
