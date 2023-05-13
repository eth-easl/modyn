#include "storage.hpp"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

using namespace storage;

void setup_logger() {
  spdlog::set_pattern("[%Y-%m-%d:%H:%M:%S] [%s:%#] [%l] %v");
}

argparse::ArgumentParser setup_argparser() {
  argparse::ArgumentParser parser("Modyn Storage");

  parser.add_argument("config").help("Modyn infrastructure configuration file");

  return parser;
}

int main(int argc, char *argv[]) {
  /* Entrypoint for the storage service. */
  setup_logger();

  auto parser = setup_argparser();

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    SPDLOG_ERROR("{}", err.what());
    exit(0);
  }

  std::string config_file = parser.get<std::string>("config");

  if (std::filesystem::exists(config_file) == false) {
    SPDLOG_ERROR("Config file {} does not exist.", config_file);
    exit(1);
  }

  // Verify that the config file exists and is readable.
  YAML::Node config = YAML::LoadFile(config_file);

  SPDLOG_INFO("Initializing storage.");
  Storage storage(config_file);
  SPDLOG_INFO("Starting storage.");
  storage.run();

  SPDLOG_INFO("Storage returned, exiting.");

  return 0;
}