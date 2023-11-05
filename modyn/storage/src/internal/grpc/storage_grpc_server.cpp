#include "internal/grpc/storage_grpc_server.hpp"

#include "internal/grpc/storage_service_impl.hpp"

using namespace modyn::storage;

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
  auto retrieval_threads = config_["storage"]["retrieval_threads"].as<uint64_t>();
  StorageServiceImpl service(config_, retrieval_threads);

  EnableDefaultHealthCheckService(true);
  reflection::InitProtoReflectionServerBuilderPlugin();
  ServerBuilder builder;
  builder.AddListeningPort(server_address, InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  SPDLOG_INFO("Server listening on {}", server_address);

  // Wait for the server to shutdown or signal to shutdown.
  stop_grpc_server_->wait(true);
  server->Shutdown();

  stop();
}