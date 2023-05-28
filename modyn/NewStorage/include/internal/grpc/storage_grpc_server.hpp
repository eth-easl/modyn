#pragma once

#include <absl/strings/str_format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include "internal/grpc/storage_service_impl.hpp"

namespace storage {

class StorageGrpcServer {
 private:
  YAML::Node config_;
  std::atomic<bool>* stop_grpc_server_;

 public:
  StorageGrpcServer(const YAML::Node& config, std::atomic<bool>* stop_grpc_server)
      : config_{config}, stop_grpc_server_(stop_grpc_server) {}
  void run_server() {
    int16_t port = config_["storage"]["port"].as<int16_t>();
    std::string server_address = absl::StrFormat("0.0.0.0:%d", port);
    StorageServiceImpl service;

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    SPDLOG_INFO("Server listening on {}", server_address);

    while (!stop_grpc_server_->load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    server->Shutdown();
  }
};

}  // namespace storage