#pragma once

#include <string>

#include "internal/file_watcher/file_watcher_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"
#include "yaml-cpp/yaml.h"

namespace storage {
class Storage {
 public:
  explicit Storage(const std::string& config_file) { 
    config_ = YAML::LoadFile(config_file);
    connection_ = StorageDatabaseConnection(config_);
    file_watcher_watchdog_ = FileWatchdog(config_, &stop_file_watcher_watchdog_);
    grpc_server_ = StorageGrpcServer(config_, &stop_grpc_server_);
  }
  void run();

 private:
  YAML::Node config_;
  StorageDatabaseConnection connection_;
  std::atomic<bool> stop_file_watcher_watchdog_ = false;
  std::atomic<bool> stop_grpc_server_ = false;
  FileWatchdog file_watcher_watchdog_;
  StorageGrpcServer grpc_server_;
};
}  // namespace storage
