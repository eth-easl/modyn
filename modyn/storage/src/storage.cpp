#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>

#include "internal/file_watcher/file_watcher_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"

using namespace modyn::storage;

void Storage::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  connection_.create_tables();

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
