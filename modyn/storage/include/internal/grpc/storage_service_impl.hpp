#pragma once

#include <grpcpp/grpcpp.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <future>
#include <queue>
#include <thread>
#include <variant>

#include "internal/database/storage_database_connection.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "storage.grpc.pb.h"

namespace modyn::storage {

using namespace grpc;

template <typename T>
using T_ptr = std::variant<
    std::enable_if_t<std::is_same<T, modyn::storage::GetDataInIntervalResponse>::value, T*>,  // NOLINT
                                                                                              // modernize-type-traits
    std::enable_if_t<std::is_same<T, modyn::storage::GetNewDataSinceResponse>::value, T*>>;   // NOLINT
                                                                                              // modernize-type-traits

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
  Status Get(ServerContext* context, const modyn::storage::GetRequest* request,
             ServerWriter<modyn::storage::GetResponse>* writer) override;
  Status GetNewDataSince(ServerContext* context, const modyn::storage::GetNewDataSinceRequest* request,
                         ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) override;
  Status GetDataInInterval(ServerContext* context, const modyn::storage::GetDataInIntervalRequest* request,
                           ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) override;
  Status CheckAvailability(ServerContext* context, const modyn::storage::DatasetAvailableRequest* request,
                           modyn::storage::DatasetAvailableResponse* response) override;
  Status RegisterNewDataset(ServerContext* context, const modyn::storage::RegisterNewDatasetRequest* request,
                            modyn::storage::RegisterNewDatasetResponse* response) override;
  Status GetCurrentTimestamp(ServerContext* context, const modyn::storage::GetCurrentTimestampRequest* request,
                             modyn::storage::GetCurrentTimestampResponse* response) override;
  Status DeleteDataset(ServerContext* context, const modyn::storage::DatasetAvailableRequest* request,
                       modyn::storage::DeleteDatasetResponse* response) override;
  Status DeleteData(ServerContext* context, const modyn::storage::DeleteDataRequest* request,
                    modyn::storage::DeleteDataResponse* response) override;
  Status GetDataPerWorker(ServerContext* context, const modyn::storage::GetDataPerWorkerRequest* request,
                          ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) override;
  Status GetDatasetSize(ServerContext* context, const modyn::storage::GetDatasetSizeRequest* request,
                        modyn::storage::GetDatasetSizeResponse* response) override;
  static std::tuple<int64_t, int64_t> get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                               int64_t total_num_elements);

 private:
  static void get_sample_data(soci::session& session, int64_t dataset_id, const std::vector<int64_t>& sample_ids,
                              std::map<int64_t, SampleData>& file_id_to_sample_data);
  void send_get_response(ServerWriter<modyn::storage::GetResponse>* writer, int64_t file_id,
                         const SampleData& sample_data, const YAML::Node& file_wrapper_config,
                         const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper, int64_t file_wrapper_type);
  template <typename T>
  void send_file_ids_and_labels(ServerWriter<T>* writer, int64_t dataset_id, int64_t start_timestamp = -1,
                                int64_t end_timestamp = -1);
  template <typename T>
  void send_samples_synchronous_retrieval(ServerWriter<T>* writer, int64_t file_id, soci::session& session,
                                          int64_t dataset_id);
  template <typename T>
  void send_samples_asynchronous_retrieval(ServerWriter<T>* writer, int64_t file_id, soci::session& session,
                                           int64_t dataset_id);
  static SampleData get_sample_subset(int64_t file_id, int64_t start_index, int64_t end_index, int64_t dataset_id,
                                      const StorageDatabaseConnection& storage_database_connection);
  static int64_t get_number_of_samples_in_file(int64_t file_id, soci::session& session);

  static std::vector<int64_t> get_file_ids(int64_t dataset_id, soci::session& session, int64_t start_timestamp = -1,
                                           int64_t end_timestamp = -1);
  static int64_t get_dataset_id(const std::string& dataset_name, soci::session& session);
  YAML::Node config_;
  int64_t sample_batch_size_{};
  int64_t retrieval_threads_;
  bool disable_multithreading_;
  std::vector<std::thread> retrieval_threads_vector_{};
  StorageDatabaseConnection storage_database_connection_;
};
}  // namespace modyn::storage