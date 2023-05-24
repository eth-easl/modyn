#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>

#include "internal/file_watcher/file_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"

using namespace storage;

void Storage::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  // Create the database tables
  const StorageDatabaseConnection connection(config_);
  connection.create_tables();

  // Create the dataset watcher process in a new thread
  std::atomic<bool> stop_file_watcher = false;
  const FileWatchdog watchdog = FileWatchdog(config_, &stop_file_watcher);

  std::thread file_watchdog_thread(&FileWatchdog::run, watchdog);

  // Start the storage grpc server
  std::atomic<bool> stop_grpc_server = false;
  const StorageGrpcServer grpc_server = StorageGrpcServer(config_, &stop_grpc_server);

  std::thread grpc_server_thread(&StorageGrpcServer::run_server, grpc_server);

  SPDLOG_INFO("Storage service shutting down.");

  // Stop the grpc server
  stop_grpc_server.store(true);
  grpc_server_thread.join();

  // Stop the file watcher
  stop_file_watcher.store(true);
  file_watchdog_thread.join();
}