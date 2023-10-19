#pragma once

#include <grpcpp/grpcpp.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <deque>

#include "internal/database/storage_database_connection.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "storage.grpc.pb.h"

namespace storage::grpcs {

struct SampleData {
  std::vector<int64_t> ids{};
  std::vector<int64_t> indices{};
  std::vector<int64_t> labels{};
};

class StorageServiceImpl final : public modyn::storage::Storage::Service {
 public:
  explicit StorageServiceImpl(const YAML::Node& config, int64_t retrieval_threads = 1)
      : Service(),  // NOLINT readability-redundant-member-init
        config_{config},
        retrieval_threads_{retrieval_threads},
        disable_multithreading_{retrieval_threads <= 1},
        storage_database_connection_{config} {
    if (!config_["storage"]["sample_batch_size"]) {
      SPDLOG_ERROR("No sample_batch_size specified in config.yaml");
      return;
    }
    sample_batch_size_ = config_["storage"]["sample_batch_size"].as<int64_t>();

    if (disable_multithreading_) {
      SPDLOG_INFO("Multithreading disabled.");
    } else {
      SPDLOG_INFO("Multithreading enabled.");
      retrieval_threads_vector_ = std::vector<std::thread>(retrieval_threads_);
    }
  }
  ::grpc::Status Get(::grpc::ServerContext* context, const modyn::storage::GetRequest* request,
                     ::grpc::ServerWriter<modyn::storage::GetResponse>* writer) override;
  ::grpc::Status GetNewDataSince(::grpc::ServerContext* context, const modyn::storage::GetNewDataSinceRequest* request,
                                 ::grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) override;
  ::grpc::Status GetDataInInterval(::grpc::ServerContext* context,
                                   const modyn::storage::GetDataInIntervalRequest* request,
                                   ::grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) override;
  ::grpc::Status CheckAvailability(::grpc::ServerContext* context,
                                   const modyn::storage::DatasetAvailableRequest* request,
                                   modyn::storage::DatasetAvailableResponse* response) override;
  ::grpc::Status RegisterNewDataset(::grpc::ServerContext* context,
                                    const modyn::storage::RegisterNewDatasetRequest* request,
                                    modyn::storage::RegisterNewDatasetResponse* response) override;
  ::grpc::Status GetCurrentTimestamp(::grpc::ServerContext* context,
                                     const modyn::storage::GetCurrentTimestampRequest* request,
                                     modyn::storage::GetCurrentTimestampResponse* response) override;
  ::grpc::Status DeleteDataset(::grpc::ServerContext* context, const modyn::storage::DatasetAvailableRequest* request,
                               modyn::storage::DeleteDatasetResponse* response) override;
  ::grpc::Status DeleteData(::grpc::ServerContext* context, const modyn::storage::DeleteDataRequest* request,
                            modyn::storage::DeleteDataResponse* response) override;
  ::grpc::Status GetDataPerWorker(::grpc::ServerContext* context,
                                  const modyn::storage::GetDataPerWorkerRequest* request,
                                  ::grpc::ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) override;
  ::grpc::Status GetDatasetSize(::grpc::ServerContext* context, const modyn::storage::GetDatasetSizeRequest* request,
                                modyn::storage::GetDatasetSizeResponse* response) override;
  static std::tuple<int64_t, int64_t> get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                               int64_t total_num_elements);

 private:
  YAML::Node config_;
  int64_t sample_batch_size_{};
  int64_t retrieval_threads_;
  bool disable_multithreading_;
  std::vector<std::thread> retrieval_threads_vector_{};
  storage::database::StorageDatabaseConnection storage_database_connection_;
  static void get_sample_data(soci::session& session, int64_t dataset_id, const std::vector<int64_t>& sample_ids,
                              std::map<int64_t, SampleData>& file_id_to_sample_data);
  void send_get_response(::grpc::ServerWriter<modyn::storage::GetResponse>* writer, int64_t file_id,
                         const SampleData& sample_data, const YAML::Node& file_wrapper_config,
                         const std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper>& filesystem_wrapper,
                         int64_t file_wrapper_type);
  void send_get_new_data_since_response(::grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer,
                                        int64_t file_id);
  void send_get_new_data_in_interval_response(::grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer,
                                              int64_t file_id);
  static int64_t get_number_of_files(int64_t dataset_id, soci::session& session);
  static int64_t get_dataset_id(const std::string& dataset_name, soci::session& session);
};
}  // namespace storage::grpcs