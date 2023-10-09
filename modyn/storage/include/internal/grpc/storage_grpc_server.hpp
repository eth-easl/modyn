#pragma once

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "internal/grpc/storage_service_impl.hpp"

namespace storage::grpc {

class StorageGrpcServer {
 public:
  StorageGrpcServer(const YAML::Node& config, std::atomic<bool>* stop_grpc_server)
      : config_{config}, stop_grpc_server_(stop_grpc_server) {}
  void run();

 private:
  YAML::Node config_;
  std::atomic<bool>* stop_grpc_server_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace storage::grpc