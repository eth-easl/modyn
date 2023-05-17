#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>

#include "internal/file_watcher/file_watcher.hpp"

using namespace storage;

void Storage::run() {  // NOLINT // TODO: Remove NOLINT after implementation
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  // Create the database tables
  const StorageDatabaseConnection connection(config_);
  connection.create_tables();

  // Create the dataset watcher process in a new thread
  std::atomic<bool> stop_file_watcher = false;
  const std::shared_ptr<FileWatchdog> watchdog = std::make_shared<FileWatchdog>(config_, &stop_file_watcher);

  std::thread file_watchdog_thread(&FileWatchdog::run, watchdog);

  // Start the storage grpc server

  SPDLOG_INFO("Storage service shutting down.");
  stop_file_watcher = true;
  file_watchdog_thread.join();
}