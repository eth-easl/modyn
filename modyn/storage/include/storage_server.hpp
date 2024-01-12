#pragma once

#include <string>

#include "internal/file_watcher/file_watcher_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"
#include "yaml-cpp/yaml.h"

namespace modyn::storage {
class StorageServer {
 public:
  explicit StorageServer(const std::string& config_file)
      : config_{YAML::LoadFile(config_file)},
        connection_{config_},
        file_watcher_watchdog_{config_, &stop_file_watcher_watchdog_, &storage_shutdown_requested_},
        grpc_server_{config_, &stop_grpc_server_, &storage_shutdown_requested_} {}
  void run();

 private:
  YAML::Node config_;
  StorageDatabaseConnection connection_;
  std::atomic<bool> storage_shutdown_requested_ = false;
  std::atomic<bool> stop_file_watcher_watchdog_ = false;
  std::atomic<bool> stop_grpc_server_ = false;
  FileWatcherWatchdog file_watcher_watchdog_;
  StorageGrpcServer grpc_server_;
};
}  // namespace modyn::storage
