#pragma once

#include <grpcpp/grpcpp.h>
#include <spdlog/spdlog.h>

#include "storage.grpc.pb.h"

namespace storage {

class StorageServiceImpl final : public modyn::storage::Service {
 private:
  YAML::Node config_;
  int16_t sample_batch_size_;

 public:
  explicit StorageServiceImpl(const YAML::Node& config) : config_{config} : Service() {
    sample_batch_size_ = config_["storage"]["sample_batch_size"].as<int16_t>();
  }
  grpc::Status Get(grpc::ServerContext* context, const modyn::storage::GetRequest* request,
                   grpc::ServerWriter<modyn::storage::GetResponse>* writer) override;
  grpc::Status GetNewDataSince(grpc::ServerContext* context, const modyn::storage::GetNewDataSinceRequest* request,
                               grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) override;
  grpc::Status GetDataInInterval(grpc::ServerContext* context, const modyn::storage::GetDataInIntervalRequest* request,
                                 grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) override;
  grpc::Status CheckAvailability(grpc::ServerContext* context, const modyn::storage::DatasetAvailableRequest* request,
                                 modyn::storage::DatasetAvailableResponse* response) override;
  grpc::Status RegisterNewDataset(grpc::ServerContext* context,
                                  const modyn::storage::RegisterNewDatasetRequest* request,
                                  modyn::storage::RegisterNewDatasetResponse* response) override;
  grpc::Status GetCurrentTimestamp(grpc::ServerContext* context,
                                   const modyn::storage::GetCurrentTimestampRequest* request,
                                   modyn::storage::GetCurrentTimestampResponse* response) override;
  grpc::Status DeleteDataset(grpc::ServerContext* context, const modyn::storage::DatasetAvailableRequest* request,
                             modyn::storage::DeleteDatasetResponse* response) override;
  grpc::Status DeleteData(grpc::ServerContext* context, const modyn::storage::DeleteDataRequest* request,
                          modyn::storage::DeleteDataResponse* response) override;
};
}  // namespace storage