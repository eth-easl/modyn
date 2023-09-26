#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>

using namespace storage;

void Storage::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  connection_.create_tables();

  // Start the file watcher watchdog
  std::thread file_watcher_watchdog_thread(&FileWatchdog::run, file_watcher_watchdog_);

  // Start the storage grpc server
  std::thread grpc_server_thread(&StorageGrpcServer::run, grpc_server_);

  // Wait for the file watcher watchdog or grpc server to exit
  SPDLOG_INFO("Storage service shutting down.");

  // Stop the grpc server
  stop_grpc_server_.store(true);
  grpc_server_thread.join();

  // Stop the file watcher
  stop_file_watcher_.store(true);
  file_watcher_watchdog_thread.join();
}
