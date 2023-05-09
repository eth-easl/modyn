#include "../../src/internal/file_watcher/FileWatcher.hpp"
#include "../../src/internal/utils/Utils.hpp"
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

void setup_logger() {
  spdlog::set_pattern("[%Y-%m-%d:%H:%M:%S] [%s:%#] [%l] %v");
}

argparse::ArgumentParser setup_argparser() {
  argparse::ArgumentParser parser("Modyn FileWatcher");

  parser.add_argument("config").help("Modyn infrastructure configuration file");
  parser.add_argument("dataset_id").help("Dataset ID to watch");
  parser.add_argument("is_test").help("Whether this is a test run or not");
  parser.add_argument("--fptf").help("File containing the file paths to watch");
  parser.add_argument("--dfe").help("Data File Extension (DFE) to use");
  parser.add_argument("--fwt").help("File Wrapper Type (FWT) to use");
  parser.add_argument("--t").help("Timestamp to start watching from");
  parser.add_argument("--fsw").help("File System Wrapper (FSW) to use");
  parser.add_argument("--dp").help("Data Path (DP) to use");

  return parser;
}

int main(int argc, char *argv[]) {
  setup_logger();

  auto parser = setup_argparser();

  try {
    parser.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    SPDLOG_ERROR("{}", err.what());
    exit(0);
  }

  std::string config_file = parser.get<std::string>("config");
  long long dataset_id = parser.get<long long>("dataset_id");
  bool is_test = parser.get<bool>("is_test");

  if (std::filesystem::exists(config_file) == false) {
    SPDLOG_ERROR("Config file {} does not exist.", config_file);
    exit(1);
  }

  // Verify that the config file exists and is readable.
  YAML::Node config = YAML::LoadFile(config_file);

  if (auto fn = parser.present("--fptf")) {
    std::string file_paths_to_watch_file = parser.get<std::string>("--fptf");
    if (std::filesystem::exists(file_paths_to_watch_file) == false) {
      SPDLOG_ERROR("File paths to watch file {} does not exist.",
                   file_paths_to_watch_file);
      exit(1);
    }
    // if fptf is present, then fwt, dfe, fsw, dp, and t must also be present
    if (auto fn = parser.present("--fwt")) {
      SPDLOG_ERROR("File Wrapper Type (FWT) must be specified.");
      exit(1);
    }
    std::string file_wrapper_type = parser.get<std::string>("--fwt");
    if (auto fn = parser.present("--dfe")) {
      SPDLOG_ERROR("Data File Extension (DFE) must be specified.");
      exit(1);
    }
    std::string data_file_extension = parser.get<std::string>("--dfe");
    if (auto fn = parser.present("--t")) {
      SPDLOG_ERROR("Timestamp (t) must be specified.");
      exit(1);
    }
    long long timestamp = parser.get<long long>("--t");
    if (auto fn = parser.present("--fsw")) {
      SPDLOG_ERROR("File System Wrapper (FSW) must be specified.");
      exit(1);
    }
    std::string file_system_wrapper_type = parser.get<std::string>("--fsw");
    if (auto fn = parser.present("--dp")) {
      SPDLOG_ERROR("Data Path (DP) must be specified.");
      exit(1);
    }
    std::string data_path = parser.get<std::string>("--dp");

    // Extract the file paths which are written in the file comma separated
    std::ifstream file_paths_to_watch_file_stream(file_paths_to_watch_file);
    std::string file_paths_to_watch_file_line;
    std::vector<std::string> file_paths_to_watch;
    while (std::getline(file_paths_to_watch_file_stream,
                        file_paths_to_watch_file_line, ',')) {
      file_paths_to_watch.push_back(file_paths_to_watch_file_line);
    }

    // Run the file watcher to handle the file paths in the file
    storage::FileWatcher file_watcher(config_file, dataset_id, is_test);
    storage::AbstractFilesystemWrapper *file_system_wrapper =
        storage::Utils::get_filesystem_wrapper(file_system_wrapper_type,
                                               data_path);
    file_watcher.handle_file_paths(file_paths_to_watch, data_file_extension,
                                   file_wrapper_type, file_system_wrapper,
                                   timestamp);
  } else {
    // Run the file watche vanilla
    storage::FileWatcher file_watcher(config_file, dataset_id, is_test);
    file_watcher.run();
  }
}