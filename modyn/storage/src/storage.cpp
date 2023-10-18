#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <thread>

#include "internal/file_watcher/file_watcher_watchdog.hpp"
#include "internal/grpc/storage_grpc_server.hpp"

using namespace storage;

void Storage::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  connection_.create_tables();

  // Start the file watcher watchdog
  std::thread file_watcher_watchdog_thread(&file_watcher::FileWatcherWatchdog::run, &file_watcher_watchdog_);

  // Start the storage grpc server
  std::thread grpc_server_thread(&grpc::StorageGrpcServer::run, &grpc_server_);

  // Create a condition variable to wait for the file watcher watchdog or gRPC server to exit.
  std::condition_variable cv;

  // Create a mutex to protect the `stop_grpc_server_` and `stop_file_watcher_watchdog_` variables.
  std::mutex stop_mutex;

  {
    std::unique_lock<std::mutex> lk(stop_mutex);
    cv.wait(lk, [&] { return stop_grpc_server_.load() || stop_file_watcher_watchdog_.load(); });
  }

  SPDLOG_INFO("Storage service shutting down.");

  stop_grpc_server_.store(true);
  grpc_server_thread.join();

  stop_file_watcher_watchdog_.store(true);
  file_watcher_watchdog_thread.join();
}
