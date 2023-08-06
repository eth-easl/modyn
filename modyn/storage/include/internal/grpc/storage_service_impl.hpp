#pragma once

#include <grpcpp/grpcpp.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <deque>

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "storage.grpc.pb.h"

namespace storage {

struct SampleData {
  std::vector<int64_t> ids;
  std::vector<int64_t> indices;
  std::vector<int64_t> labels;
};

class StorageServiceImpl final : public modyn::storage::Storage::Service {
 private:
  YAML::Node config_;
  int16_t sample_batch_size_;
  std::vector<std::thread> thread_pool;
  std::deque<std::function<void()>> tasks;
  std::mutex mtx;
  std::condition_variable cv;
  int16_t retrieval_threads_;
  bool disable_multithreading_;
  void send_get_response(grpc::ServerWriter<modyn::storage::GetResponse>* writer, int64_t file_id,
                         SampleData sample_data, const YAML::Node& file_wrapper_config,
                         const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper, int64_t file_wrapper_type);
  void send_get_new_data_since_response(grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer,
                                        int64_t file_id);
  void send_get_new_data_in_interval_response(grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer,
                                              int64_t file_id);

 public:
  explicit StorageServiceImpl(const YAML::Node& config, int16_t retrieval_threads = 1)
      : Service(), config_{config}, retrieval_threads_{retrieval_threads} {  // NOLINT
                                                                             // (cppcoreguidelines-pro-type-member-init)
    if (!config_["storage"]["sample_batch_size"]) {
      SPDLOG_ERROR("No sample_batch_size specified in config.yaml");
      return;
    }
    sample_batch_size_ = config_["storage"]["sample_batch_size"].as<int16_t>();

    disable_multithreading_ = retrieval_threads_ <= 1;  // NOLINT

    if (disable_multithreading_) {
      SPDLOG_INFO("Multithreading disabled.");
    } else {
      SPDLOG_INFO("Multithreading enabled.");

      thread_pool.resize(retrieval_threads_);

      for (auto& thread : thread_pool) {
        thread = std::thread([&]() {
          while (true) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock(mtx);
              cv.wait(lock, [&]() { return !tasks.empty(); });
              task = std::move(tasks.front());
              tasks.pop_front();
            }
            if (!task) break;  // If the task is empty, it's a signal to terminate the thread
            task();
          }
        });
      }
    }
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
  grpc::Status GetDataPerWorker(grpc::ServerContext* context, const modyn::storage::GetDataPerWorkerRequest* request,
                                grpc::ServerWriter< ::modyn::storage::GetDataPerWorkerResponse>* writer) override;
  grpc::Status GetDatasetSize(grpc::ServerContext* context, const modyn::storage::GetDatasetSizeRequest* request,
                              modyn::storage::GetDatasetSizeResponse* response) override;
  virtual std::tuple<int64_t, int64_t> get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                      int64_t total_num_elements);
  static int64_t get_dataset_id(const std::string& dataset_name, soci::session& session) {
    int64_t dataset_id = 0;
    session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

    return dataset_id;
  }
};
}  // namespace storage