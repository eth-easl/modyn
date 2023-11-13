#include "storage_server.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <thread>

#include "internal/file_watcher/file_watcher_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"

using namespace modyn::storage;

void StorageServer::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service. Creating tables.");

  connection_.create_tables();
  SPDLOG_INFO("Running storage service. Initializing datasets from config.");

  for (const YAML::Node& dataset_node : config_["storage"]["datasets"]) {
    const auto dataset_id = dataset_node["name"].as<std::string>();
    const auto base_path = dataset_node["base_path"].as<std::string>();
    const auto filesystem_wrapper_type = dataset_node["filesystem_wrapper_type"].as<std::string>();
    const auto file_wrapper_type = dataset_node["file_wrapper_type"].as<std::string>();
    const auto description = dataset_node["description"].as<std::string>();
    const auto version = dataset_node["version"].as<std::string>();
    const YAML::Node& file_wrapper_config_node = dataset_node["file_wrapper_config"];
    std::ostringstream fwc_stream;
    fwc_stream << file_wrapper_config_node;
    const std::string file_wrapper_config = fwc_stream.str();

    SPDLOG_INFO("Parsed filewrapper_config: {}", file_wrapper_config);

    bool ignore_last_timestamp = false;
    int file_watcher_interval = 5;

    if (dataset_node["ignore_last_timestamp"]) {
      ignore_last_timestamp = dataset_node["ignore_last_timestamp"].as<bool>();
    }

    if (dataset_node["file_watcher_interval"]) {
      file_watcher_interval = dataset_node["file_watcher_interval"].as<int>();
    }

    const bool success = connection_.add_dataset(
        dataset_id, base_path, FilesystemWrapper::get_filesystem_wrapper_type(filesystem_wrapper_type),
        FileWrapper::get_file_wrapper_type(file_wrapper_type), description, version, file_wrapper_config,
        ignore_last_timestamp, file_watcher_interval);
    if (!success) {
      SPDLOG_ERROR(fmt::format("Could not register dataset {} - potentially it already exists.", dataset_id));
    }
  }

  SPDLOG_INFO("Starting file watcher watchdog.");

  // Start the file watcher watchdog
  std::thread file_watcher_watchdog_thread(&FileWatcherWatchdog::run, &file_watcher_watchdog_);

  SPDLOG_INFO("Starting storage gRPC server.");

  // Start the storage grpc server
  std::thread grpc_server_thread(&StorageGrpcServer::run, &grpc_server_);

  // Wait for shutdown signal (storage_shutdown_requested_ true)
  storage_shutdown_requested_.wait(true);

  SPDLOG_INFO("Storage service shutting down.");

  stop_grpc_server_.store(true);
  grpc_server_thread.join();

  stop_file_watcher_watchdog_.store(true);
  file_watcher_watchdog_thread.join();
}
