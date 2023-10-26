#include <spdlog/spdlog.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>

#include "internal/utils/utils.hpp"
#include "storage.hpp"

using namespace storage;

void setup_logger() { spdlog::set_pattern("[%Y-%m-%d:%H:%M:%S] [%s:%#] [%l] %v"); }

argparse::ArgumentParser setup_argparser() {
  argparse::ArgumentParser parser("Modyn Storage");

  parser.add_argument("config").help("Modyn infrastructure configuration file");

  return parser;
}

int main(int argc, char* argv[]) {
  /* Entrypoint for the storage service. */
  setup_logger();

  auto parser = setup_argparser();

  parser.parse_args(argc, argv);

  std::string config_file = parser.get<std::string>("config");  // NOLINT misc-const-correctness

  ASSERT(std::filesystem::exists(config_file), "Config file does not exist.");
  if (!std::filesystem::exists(config_file)) {
    FAIL("Config file does not exist.");
  }

  // Verify that the config file exists and is readable.
  const YAML::Node config = YAML::LoadFile(config_file);

  SPDLOG_INFO("Initializing storage.");
  Storage storage(config_file);
  SPDLOG_INFO("Starting storage.");
  storage.run();

  SPDLOG_INFO("Storage returned, exiting.");

  return 0;
}