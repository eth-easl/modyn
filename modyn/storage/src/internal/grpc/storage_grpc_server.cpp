#include "internal/grpc/storage_grpc_server.hpp"

#include "internal/grpc/storage_service_impl.hpp"

using namespace storage::grpcs;

void StorageGrpcServer::run() {
  if (!config_["storage"]["port"]) {
    SPDLOG_ERROR("No port specified in config.yaml");
    return;
  }
  auto port = config_["storage"]["port"].as<int64_t>();
  std::string server_address = fmt::format("[::]:{}", port);
  if (!config_["storage"]["retrieval_threads"]) {
    SPDLOG_ERROR("No retrieval_threads specified in config.yaml");
    return;
  }
  auto retrieval_threads = config_["storage"]["retrieval_threads"].as<int64_t>();
  StorageServiceImpl service(config_, retrieval_threads);

  ::grpc::EnableDefaultHealthCheckService(true);
  ::grpc::reflection::InitProtoReflectionServerBuilderPlugin();
  ::grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, ::grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  SPDLOG_INFO("Server listening on {}", server_address);

  {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [&] { return stop_grpc_server_->load(); });
  }

  server->Shutdown();
  stop_grpc_server_->store(true);
}