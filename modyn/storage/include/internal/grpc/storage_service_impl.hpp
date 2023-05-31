#pragma once

#include <grpcpp/grpcpp.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "storage.grpc.pb.h"

namespace storage {

class StorageServiceImpl final : public modyn::storage::Storage::Service {
 private:
  YAML::Node config_;
  int16_t sample_batch_size_;

 public:
  explicit StorageServiceImpl(const YAML::Node& config)
      : Service(), config_{config} {  // NOLINT (cppcoreguidelines-pro-type-member-init)
    if (!config_["storage"]["sample_batch_size"]) {
      SPDLOG_ERROR("No sample_batch_size specified in config.yaml");
      return;
    }
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
  static int64_t get_dataset_id(const std::string& dataset_name, soci::session& session) {
    int64_t dataset_id = 0;
    session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

    return dataset_id;
  }
};
}  // namespace storage