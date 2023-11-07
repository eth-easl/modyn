#pragma once

#include <grpcpp/grpcpp.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <future>
#include <queue>
#include <thread>
#include <variant>

#include "internal/database/cursor_handler.hpp"
#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"
#include "storage.grpc.pb.h"

namespace modyn::storage {

using namespace grpc;

template <typename T>
concept IsResponse = std::is_same_v<T, modyn::storage::GetDataInIntervalResponse> ||
                     std::is_same_v<T, modyn::storage::GetNewDataSinceResponse>;

struct SampleData {
  std::vector<int64_t> ids{};
  std::vector<int64_t> indices{};
  std::vector<int64_t> labels{};
};

struct DatasetData {
  int64_t dataset_id = -1;
  std::string base_path;
  FilesystemWrapperType filesystem_wrapper_type = FilesystemWrapperType::INVALID_FSW;
  FileWrapperType file_wrapper_type = FileWrapperType::INVALID_FW;
  std::string file_wrapper_config;
};

class StorageServiceImpl final : public modyn::storage::Storage::Service {
 public:
  explicit StorageServiceImpl(const YAML::Node& config, uint64_t retrieval_threads = 1)
      : Service(),  // NOLINT readability-redundant-member-init  (we need to call the base constructor)
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

  template <typename WriterT>
  Status Get_Impl(  // NOLINT readability-identifier-naming
      ServerContext* /*context*/, const modyn::storage::GetRequest* request, WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();

      // Check if the dataset exists
      std::string dataset_name = request->dataset_id();
      const DatasetData dataset_data = get_dataset_data(session, dataset_name);

      if (dataset_data.dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
        return {StatusCode::OK, "Dataset does not exist."};
      }

      const int keys_size = request->keys_size();
      std::vector<int64_t> request_keys(keys_size);
      for (int i = 0; i < keys_size; i++) {
        request_keys[i] = request->keys(i);
      }

      if (request_keys.empty()) {
        SPDLOG_ERROR("No keys provided.");
        return {StatusCode::OK, "No keys provided."};
      }

      send_sample_data_from_keys<WriterT>(writer, request_keys, dataset_data, session,
                                          storage_database_connection_.get_drivername());

      return {StatusCode::OK, "Data retrieved."};
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in Get: {}", e.what());
      return {StatusCode::OK, fmt::format("Error in Get: {}", e.what())};
    }
  }

  template <typename WriterT>
  Status GetNewDataSince_Impl(ServerContext* context, const modyn::storage::GetNewDataSinceRequest* request,
                              WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();
      const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
      if (dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
        return {StatusCode::OK, "Dataset does not exist."};
      }
      const int64_t request_timestamp = request->timestamp();

      SPDLOG_INFO(fmt::format("Received GetNewDataSince Request for dataset {} (id = {}) with timestamp {}.",
                              request->dataset_id(), dataset_id, request_timestamp));

      send_file_ids_and_labels<modyn::storage::GetNewDataSinceResponse, WriterT>(writer, dataset_id, request_timestamp);
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in GetNewDataSince: {}", e.what());
      return {StatusCode::OK, fmt::format("Error in GetNewDataSince: {}", e.what())};
    }
    return {StatusCode::OK, "Data retrieved."};
  }

  template <typename WriterT>
  Status GetDataInInterval_Impl(ServerContext* context, const modyn::storage::GetDataInIntervalRequest* request,
                                WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();
      const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
      if (dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
        return {StatusCode::OK, "Dataset does not exist."};
      }
      const int64_t start_timestamp = request->start_timestamp();
      const int64_t end_timestamp = request->end_timestamp();

      SPDLOG_INFO(
          fmt::format("Received GetDataInInterval Request for dataset {} (id = {}) with start = {} and end = {}.",
                      request->dataset_id(), dataset_id, start_timestamp, end_timestamp));

      send_file_ids_and_labels<modyn::storage::GetDataInIntervalResponse, WriterT>(writer, dataset_id, start_timestamp,
                                                                                   end_timestamp);
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in GetDataInInterval: {}", e.what());
      return {StatusCode::OK, fmt::format("Error in GetDataInInterval: {}", e.what())};
    }
    return {StatusCode::OK, "Data retrieved."};
  }

  template <typename WriterT = ServerWriter<modyn::storage::GetResponse>>
  void send_sample_data_from_keys(WriterT* writer, const std::vector<int64_t>& request_keys,
                                  const DatasetData& dataset_data, soci::session& session,
                                  const DatabaseDriver& driver) {
    // TODO(maxiBoether): we need to benchmark this. In Python, we just get all samples from the DB and then fetch then
    // from disk. Here, we first have to get all files with a big subq, then all samples for each file again. Not sure
    // if this is faster instead of one big query and then parallelizing over that result.
    const std::vector<int64_t> file_ids = get_file_ids_for_samples(request_keys, dataset_data.dataset_id, session);

    if (file_ids.empty()) {
      SPDLOG_ERROR("No files corresponding to the keys found in dataset {}.", dataset_data.dataset_id);
      return;
    }

    // create mutex to protect the writer from concurrent writes as this is not supported by gRPC
    std::mutex writer_mutex;

    if (disable_multithreading_) {
      for (auto file_id : file_ids) {
        const std::vector<int64_t> samples_corresponding_to_file =
            get_samples_corresponding_to_file(file_id, dataset_data.dataset_id, request_keys, session);
        send_sample_data_for_keys_and_file<WriterT>(writer, writer_mutex, file_id, samples_corresponding_to_file,
                                                    dataset_data, session, driver, sample_batch_size_);
      }
    } else {
      std::vector<std::vector<int64_t>> file_ids_per_thread = get_file_ids_per_thread(file_ids, retrieval_threads_);

      auto thread_function = [this, writer, &writer_mutex, &file_ids_per_thread, &request_keys, &dataset_data, &session,
                              &driver](int64_t thread_id) {
        for (const int64_t file_id : file_ids_per_thread[thread_id]) {
          const std::vector<int64_t>& samples_corresponding_to_file =
              get_samples_corresponding_to_file(file_id, dataset_data.dataset_id, request_keys, session);
          send_sample_data_for_keys_and_file<WriterT>(writer, writer_mutex, file_id, samples_corresponding_to_file,
                                                      dataset_data, session, driver, sample_batch_size_);
        }
      };

      std::vector<std::thread> threads;
      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        threads.emplace_back(thread_function, thread_id);
      }

      for (auto& thread : threads) {
        thread.join();
      }
    }
  }

  template <typename ResponseT, typename WriterT = ServerWriter<ResponseT>>
  void send_file_ids_and_labels(WriterT* writer, const int64_t dataset_id, const int64_t start_timestamp = -1,
                                int64_t end_timestamp = -1) {
    soci::session session = storage_database_connection_.get_session();

    const std::vector<int64_t> file_ids = get_file_ids(session, dataset_id, start_timestamp, end_timestamp);
    SPDLOG_INFO(fmt::format("send_file_ids_and_labels got {} file ids.", file_ids.size()));

    std::mutex writer_mutex;  // We need to protect the writer from concurrent writes as this is not supported by gRPC

    if (disable_multithreading_) {
      send_sample_id_and_label<ResponseT, WriterT>(writer, writer_mutex, file_ids, storage_database_connection_,
                                                   dataset_id, sample_batch_size_);
    } else {
      // Split the number of files over retrieval_threads_
      auto file_ids_per_thread = get_file_ids_per_thread(file_ids, retrieval_threads_);

      std::vector<std::thread> retrieval_threads_vector(retrieval_threads_);
      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        retrieval_threads_vector[thread_id] =
            std::thread([this, writer, &file_ids_per_thread, thread_id, dataset_id, &writer_mutex]() {
              send_sample_id_and_label<ResponseT, WriterT>(writer, writer_mutex, file_ids_per_thread[thread_id],
                                                           std::ref(storage_database_connection_), dataset_id,
                                                           sample_batch_size_);
            });
      }

      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        retrieval_threads_vector[thread_id].join();
      }
    }
  }

  template <typename ResponseT, typename WriterT = ServerWriter<ResponseT>>
  static void send_sample_id_and_label(WriterT* writer, std::mutex& writer_mutex, const std::vector<int64_t>& file_ids,
                                       StorageDatabaseConnection& storage_database_connection, int64_t dataset_id,
                                       int64_t sample_batch_size) {
    soci::session session = storage_database_connection.get_session();

    std::vector<SampleRecord> record_buf;
    record_buf.reserve(sample_batch_size);

    for (const int64_t file_id : file_ids) {
      const int64_t number_of_samples = get_number_of_samples_in_file(file_id, session, dataset_id);
      SPDLOG_INFO(fmt::format("file {} has {} samples", file_id, number_of_samples));
      if (number_of_samples > 0) {
        const std::string query = fmt::format(
            "SELECT sample_id, label FROM samples WHERE file_id = {} AND dataset_id = {}", file_id, dataset_id);
        const std::string cursor_name = fmt::format("cursor_{}_{}", dataset_id, file_id);
        CursorHandler cursor_handler(session, storage_database_connection.get_drivername(), query, cursor_name, 2);

        std::vector<SampleRecord> records;

        while (true) {
          records = cursor_handler.yield_per(sample_batch_size);

          SPDLOG_INFO(fmt::format("got {} records (batch size = {})", records.size(), sample_batch_size));
          if (records.empty()) {
            break;
          }
          const uint64_t obtained_records = records.size();
          ASSERT(static_cast<int64_t>(obtained_records) <= sample_batch_size, "Received too many samples");

          if (static_cast<int64_t>(records.size()) == sample_batch_size) {
            // If we obtained a full buffer, we can emit a response directly
            ResponseT response;
            for (const auto& record : records) {
              response.add_keys(record.id);
              response.add_labels(record.column_1);
            }

            {
              const std::lock_guard<std::mutex> lock(writer_mutex);
              writer->Write(response);
            }
          } else {
            // If not, we append to our record buf
            record_buf.insert(record_buf.end(), records.begin(), records.end());
            // If our record buf is big enough, emit a message
            if (static_cast<int64_t>(records.size()) >= sample_batch_size) {
              ResponseT response;

              // sample_batch_size is signed int...
              for (int64_t record_idx = 0; record_idx < sample_batch_size; ++record_idx) {
                const SampleRecord& record = record_buf[record_idx];
                response.add_keys(record.id);
                response.add_labels(record.column_1);
              }

              // Now, delete first sample_batch_size elements from vector as we are sending them
              record_buf.erase(record_buf.begin(), record_buf.begin() + sample_batch_size);

              ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
                     "The record buffer should never have more than 2*sample_batch_size elements!");

              {
                const std::lock_guard<std::mutex> lock(writer_mutex);
                writer->Write(response);
              }
            }
          }
        }
      }
    }

    // Iterated over all files, we now need to emit all data from buffer
    if (!record_buf.empty()) {
      ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size, "We should have written this buffer before!");

      ResponseT response;
      for (const auto& record : record_buf) {
        response.add_keys(record.id);
        response.add_labels(record.column_1);
      }

      {
        const std::lock_guard<std::mutex> lock(writer_mutex);
        writer->Write(response);
      }
    }
  }

  template <typename WriterT = ServerWriter<modyn::storage::GetResponse>>
  static void send_sample_data_for_keys_and_file(WriterT* writer, std::mutex& writer_mutex, int64_t file_id,
                                                 const std::vector<int64_t>& request_keys_per_file,
                                                 const DatasetData& dataset_data, soci::session& session,
                                                 const DatabaseDriver& driver, int64_t sample_batch_size) {
    try {
      std::string file_path;
      session << "SELECT path FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id", soci::into(file_path),
          soci::use(file_id), soci::use(dataset_data.dataset_id);

      if (file_path.empty()) {
        SPDLOG_ERROR(
            fmt::format("Could not obtain full path of file id {} in dataset {}", file_id, dataset_data.dataset_id));
      }

      std::vector<SampleRecord> record_buf;
      record_buf.reserve(sample_batch_size);

      std::vector<std::vector<unsigned char>> sample_buf;
      sample_buf.reserve(sample_batch_size);

      const YAML::Node file_wrapper_config_node = YAML::Load(dataset_data.file_wrapper_config);
      auto filesystem_wrapper =
          get_filesystem_wrapper(static_cast<FilesystemWrapperType>(dataset_data.filesystem_wrapper_type));
      auto file_wrapper = get_file_wrapper(file_path, static_cast<FileWrapperType>(dataset_data.file_wrapper_type),
                                           file_wrapper_config_node, filesystem_wrapper);

      CursorHandler cursor_handler(session, driver,
                                   fmt::format("SELECT sample_id, sample_index, label FROM samples WHERE file_id = "
                                               "{} AND dataset_id = {} AND sample_id IN ({})",
                                               file_id, dataset_data.dataset_id, fmt::join(request_keys_per_file, ",")),
                                   fmt::format("file_{}", file_id), 3);

      std::vector<SampleRecord> records;

      while (true) {
        records = cursor_handler.yield_per(sample_batch_size);
        if (records.empty()) {
          break;
        }
        const uint64_t obtained_records = records.size();
        ASSERT(static_cast<int64_t>(obtained_records) <= sample_batch_size, "Received too many samples");

        std::vector<int64_t> sample_indexes(obtained_records);
        for (size_t i = 0; i < obtained_records; ++i) {
          sample_indexes[i] = records[i].column_1;
        }
        const auto samples = file_wrapper->get_samples_from_indices(sample_indexes);

        if (static_cast<int64_t>(records.size()) == sample_batch_size) {
          // If we obtained a full buffer, we can emit a response directly

          modyn::storage::GetResponse response;
          for (int64_t i = 0; i < sample_batch_size; ++i) {
            response.add_keys(records[i].id);
            response.add_labels(records[i].column_2);
            response.add_samples(samples[i].data(), samples[i].size());
          }
          {
            const std::lock_guard<std::mutex> lock(writer_mutex);
            writer->Write(response);
          }
        } else {
          // If not, we append to our buffers
          record_buf.insert(record_buf.end(), records.begin(), records.end());
          sample_buf.insert(sample_buf.end(), samples.begin(), samples.end());

          // If our record buf is big enough, emit a message
          if (static_cast<int64_t>(records.size()) >= sample_batch_size) {
            modyn::storage::GetResponse response;
            for (int64_t i = 0; i < sample_batch_size; ++i) {
              response.add_keys(record_buf[i].id);
              response.add_labels(record_buf[i].column_2);
              response.add_samples(sample_buf[i].data(), sample_buf[i].size());
            }
            // Now, delete first sample_batch_size elements from vector as we are sending them
            record_buf.erase(record_buf.begin(), record_buf.begin() + sample_batch_size);
            sample_buf.erase(sample_buf.begin(), sample_buf.begin() + sample_batch_size);

            ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
                   "The record buffer should never have more than 2*sample_batch_size elements!");

            {
              const std::lock_guard<std::mutex> lock(writer_mutex);
              writer->Write(response);
            }
          }
        }
      }

      if (!record_buf.empty()) {
        ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
               "We should have written this buffer before!");
        const uint64_t buffer_size = record_buf.size();
        modyn::storage::GetResponse response;
        for (uint64_t i = 0; i < buffer_size; ++i) {
          response.add_keys(record_buf[i].id);
          response.add_labels(record_buf[i].column_2);
          response.add_samples(sample_buf[i].data(), sample_buf[i].size());
        }
        {
          const std::lock_guard<std::mutex> lock(writer_mutex);
          writer->Write(response);
        }
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in send_sample_data_for_keys_and_file with file_id = {}, sample_batch_size = {}: {}", file_id,
                   sample_batch_size, e.what());
      throw;
    }
  }

  static std::tuple<int64_t, int64_t> get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                               int64_t total_num_elements);
  static int64_t get_number_of_samples_in_file(int64_t file_id, soci::session& session, int64_t dataset_id);

  static std::vector<int64_t> get_file_ids(soci::session& session, int64_t dataset_id, int64_t start_timestamp = -1,
                                           int64_t end_timestamp = -1);
  static int64_t get_file_count(soci::session& session, int64_t dataset_id, int64_t start_timestamp,
                                int64_t end_timestamp);
  static std::vector<int64_t> get_file_ids_given_number_of_files(soci::session& session, int64_t dataset_id,
                                                                 int64_t start_timestamp, int64_t end_timestamp,
                                                                 int64_t number_of_files);
  static int64_t get_dataset_id(soci::session& session, const std::string& dataset_name);
  static std::vector<int64_t> get_file_ids_for_samples(const std::vector<int64_t>& request_keys, int64_t dataset_id,
                                                       soci::session& session);
  static std::vector<std::vector<int64_t>> get_file_ids_per_thread(const std::vector<int64_t>& file_ids,
                                                                   uint64_t retrieval_threads);
  static std::vector<int64_t> get_samples_corresponding_to_file(int64_t file_id, int64_t dataset_id,
                                                                const std::vector<int64_t>& request_keys,
                                                                soci::session& session);
  static DatasetData get_dataset_data(soci::session& session, std::string& dataset_name);

 private:
  YAML::Node config_;
  int64_t sample_batch_size_ = 10000;
  uint64_t retrieval_threads_;
  bool disable_multithreading_;
  StorageDatabaseConnection storage_database_connection_;
};
}  // namespace modyn::storage
