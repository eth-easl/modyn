#include "internal/grpc/storage_grpc_server.hpp"

#include <thread>

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
  grpc::ResourceQuota quota;
  std::uint64_t num_cores = std::thread::hardware_concurrency();
  if (num_cores == 0) {
    SPDLOG_WARN("Could not get number of cores, assuming 64.");
    num_cores = 64;
  }
  // Note that in C++, everything is a thread in gRPC, but we want to keep the same logic as in Python
  const std::uint64_t num_processes =
      std::max(static_cast<uint64_t>(2), std::min(static_cast<uint64_t>(64), num_cores));
  const std::uint64_t num_threads_per_process = std::max(static_cast<uint64_t>(4), num_processes / 4);
  const int max_threads = static_cast<int>(num_processes * num_threads_per_process);
  SPDLOG_INFO("Using {} gRPC threads.", max_threads);
  quota.SetMaxThreads(max_threads);
  builder.SetResourceQuota(quota);

  builder.AddListeningPort(server_address, InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  SPDLOG_INFO("Server listening on {}", server_address);

  // Wait for the server to shutdown or signal to shutdown.
  stop_grpc_server_->wait(false);
  server->Shutdown();

  stop();
}