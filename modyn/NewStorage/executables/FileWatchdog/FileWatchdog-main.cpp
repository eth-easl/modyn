#include "../../src/internal/file_watcher/FileWatchdog.hpp"
#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>

void setup_logger() {
  spdlog::set_pattern("[%Y-%m-%d:%H:%M:%S] [%s:%#] [%l] %v");
}

argparse::ArgumentParser setup_argparser() {
  argparse::ArgumentParser parser("Modyn FileWatcher");

  parser.add_argument("config").help("Modyn infrastructure configuration file");

  return parser;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser parser = setup_argparser();

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

    storage::FileWatchdog file_watchdog(config_file);
    file_watchdog.run();
}