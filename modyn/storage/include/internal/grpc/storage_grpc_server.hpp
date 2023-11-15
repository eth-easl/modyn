#pragma once

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <yaml-cpp/yaml.h>

#include <condition_variable>

namespace modyn::storage {

class StorageGrpcServer {
 public:
  StorageGrpcServer(const YAML::Node& config, std::atomic<bool>* stop_grpc_server,
                    std::atomic<bool>* request_storage_shutdown)
      : config_{config}, stop_grpc_server_{stop_grpc_server}, request_storage_shutdown_{request_storage_shutdown} {}
  void run();
  void stop() {
    SPDLOG_INFO("gRPC Server requesting storage shutdown");
    stop_grpc_server_->store(true);
    request_storage_shutdown_->store(true);
  }

 private:
  YAML::Node config_;
  std::atomic<bool>* stop_grpc_server_;
  std::atomic<bool>* request_storage_shutdown_;
};

}  // namespace modyn::storage