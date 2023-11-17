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
#include "modyn/utils/utils.hpp"

// Since grpc > 1.54.2, there are extra semicola and a missing override in
// the external generated header. Since we want to have -Werror and diagnostics
// on our code, we temporarily disable the warnings when importing this generated header.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra-semi"
#if defined(__clang__)
// This is only a clang error...
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"
#endif
#include "storage.grpc.pb.h"
#pragma GCC diagnostic pop

namespace modyn::storage {

using namespace grpc;

struct SampleData {
  std::vector<int64_t> ids;
  std::vector<int64_t> indices;
  std::vector<int64_t> labels;
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
  Status Get_Impl(  // NOLINT (readability-identifier-naming)
      ServerContext* /*context*/, const modyn::storage::GetRequest* request, WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();

      // Check if the dataset exists
      std::string dataset_name = request->dataset_id();
      const DatasetData dataset_data = get_dataset_data(session, dataset_name);

      SPDLOG_INFO(fmt::format("Received GetRequest for dataset {} (id = {}) with {} keys.", dataset_name,
                              dataset_data.dataset_id, request->keys_size()));

      if (dataset_data.dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
        session.close();
        return {StatusCode::OK, "Dataset does not exist."};
      }

      const auto keys_size = static_cast<int64_t>(request->keys_size());
      if (keys_size == 0) {
        return {StatusCode::OK, "No keys provided."};
      }

      std::vector<int64_t> request_keys;
      request_keys.reserve(keys_size);
      std::copy(request->keys().begin(), request->keys().end(), std::back_inserter(request_keys));

      send_sample_data_from_keys<WriterT>(writer, request_keys, dataset_data);

      // sqlite causes memory leaks otherwise
      if (session.get_backend_name() != "sqlite3" && session.is_connected()) {
        session.close();
      }

      return {StatusCode::OK, "Data retrieved."};
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in Get: {}", e.what());
      return {StatusCode::OK, fmt::format("Error in Get: {}", e.what())};
    }
  }

  template <typename WriterT>
  Status GetNewDataSince_Impl(  // NOLINT (readability-identifier-naming)
      ServerContext* /*context*/, const modyn::storage::GetNewDataSinceRequest* request, WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();
      const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
      if (dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
        session.close();
        return {StatusCode::OK, "Dataset does not exist."};
      }
      session.close();

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
  Status GetDataInInterval_Impl(  // NOLINT (readability-identifier-naming)
      ServerContext* /*context*/, const modyn::storage::GetDataInIntervalRequest* request, WriterT* writer) {
    try {
      soci::session session = storage_database_connection_.get_session();
      const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
      if (dataset_id == -1) {
        SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
        session.close();
        return {StatusCode::OK, "Dataset does not exist."};
      }
      session.close();

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
                                  const DatasetData& dataset_data) {
    // Create mutex to protect the writer from concurrent writes as this is not supported by gRPC
    std::mutex writer_mutex;

    if (disable_multithreading_) {
      const std::vector<int64_t>::const_iterator begin = request_keys.begin();  // NOLINT (modernize-use-auto)
      const std::vector<int64_t>::const_iterator end = request_keys.end();      // NOLINT (modernize-use-auto)

      get_samples_and_send<WriterT>(begin, end, writer, &writer_mutex, &dataset_data, &config_, sample_batch_size_);

    } else {
      std::vector<std::pair<std::vector<int64_t>::const_iterator, std::vector<int64_t>::const_iterator>>
          its_per_thread = get_keys_per_thread(request_keys, retrieval_threads_);
      std::vector<std::thread> retrieval_threads_vector(retrieval_threads_);
      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        const std::vector<int64_t>::const_iterator begin = its_per_thread[thread_id].first;
        const std::vector<int64_t>::const_iterator end = its_per_thread[thread_id].second;

        retrieval_threads_vector[thread_id] =
            std::thread(StorageServiceImpl::get_samples_and_send<WriterT>, begin, end, writer, &writer_mutex,
                        &dataset_data, &config_, sample_batch_size_);
      }

      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        if (retrieval_threads_vector[thread_id].joinable()) {
          retrieval_threads_vector[thread_id].join();
        }
      }
      retrieval_threads_vector.clear();
    }
  }

  template <typename ResponseT, typename WriterT = ServerWriter<ResponseT>>
  void send_file_ids_and_labels(WriterT* writer, const int64_t dataset_id, const int64_t start_timestamp = -1,
                                int64_t end_timestamp = -1) {
    soci::session session = storage_database_connection_.get_session();
    // TODO(create issue): We might want to have a cursor for this as well and iterate over it, since that can also
    // return millions of files
    const std::vector<int64_t> file_ids = get_file_ids(session, dataset_id, start_timestamp, end_timestamp);
    session.close();

    if (file_ids.empty()) {
      return;
    }
    std::mutex writer_mutex;  // We need to protect the writer from concurrent writes as this is not supported by gRPC
    bool force_no_mt = true;
    SPDLOG_ERROR("Multithreaded retrieval of new samples is currently broken, disabling...");

    if (force_no_mt || disable_multithreading_) {
      send_sample_id_and_label<ResponseT, WriterT>(writer, &writer_mutex, file_ids.begin(), file_ids.end(), &config_,
                                                   dataset_id, sample_batch_size_);
    } else {
      // Split the number of files over retrieval_threads_
      std::vector<std::pair<std::vector<int64_t>::const_iterator, std::vector<int64_t>::const_iterator>>
          file_ids_per_thread = get_keys_per_thread(file_ids, retrieval_threads_);

      std::vector<std::thread> retrieval_threads_vector(retrieval_threads_);
      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        retrieval_threads_vector[thread_id] =
            std::thread(StorageServiceImpl::send_sample_id_and_label<ResponseT, WriterT>, writer, &writer_mutex,
                        file_ids_per_thread[thread_id].first, file_ids_per_thread[thread_id].second, &config_,
                        dataset_id, sample_batch_size_);
      }

      for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
        if (retrieval_threads_vector[thread_id].joinable()) {
          retrieval_threads_vector[thread_id].join();
        }
      }
    }
  }

  template <typename ResponseT, typename WriterT = ServerWriter<ResponseT>>
  static void send_sample_id_and_label(WriterT* writer,  // NOLINT (readability-function-cognitive-complexity)
                                       std::mutex* writer_mutex, const std::vector<int64_t>::const_iterator begin,
                                       const std::vector<int64_t>::const_iterator end, const YAML::Node* config,
                                       int64_t dataset_id, int64_t sample_batch_size) {
    if (begin >= end) {
      return;
    }

    const StorageDatabaseConnection storage_database_connection(*config);
    soci::session session = storage_database_connection.get_session();

    uint64_t num_paths = end - begin;
    // TODO(MaxiBoether): use sample_dbinsertion_batchsize or sth instead of 1 mio
    uint64_t chunk_size = static_cast<uint64_t>(1000000);
    uint64_t num_chunks = num_paths / chunk_size;
    if (num_paths % chunk_size != 0) {
      ++num_chunks;
    }

    for (uint64_t i = 0; i < num_chunks; ++i) {
      auto start_it = begin + i * chunk_size;
      auto end_it = i < num_chunks - 1 ? start_it + chunk_size : end;

      std::vector<int64_t> file_ids(start_it, end_it);
      std::string file_placeholders = fmt::format("({})", fmt::join(file_ids, ","));

      std::vector<SampleRecord> record_buf;
      record_buf.reserve(sample_batch_size);

      const std::string query = fmt::format(
          "SELECT samples.sample_id, samples.label, files.updated_at "
          "FROM samples INNER JOIN files "
          "ON samples.file_id = files.file_id AND samples.dataset_id = files.dataset_id "
          "WHERE samples.file_id IN {} AND samples.dataset_id = {} "
          "ORDER BY asc(files.updated_at)",  // TODO(MaxiBoether): This breaks with > 1 thread!
          file_placeholders, dataset_id);
      const std::string cursor_name = fmt::format("cursor_{}_{}", dataset_id, file_ids.at(0));
      CursorHandler cursor_handler(session, storage_database_connection.get_drivername(), query, cursor_name, 3);

      std::vector<SampleRecord> records;

      while (true) {
        ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
               fmt::format("Should have written records buffer, size = {}", record_buf.size()));
        records = cursor_handler.yield_per(sample_batch_size);

        if (records.empty()) {
          break;
        }

        const uint64_t obtained_records = records.size();
        ASSERT(static_cast<int64_t>(obtained_records) <= sample_batch_size, "Received too many samples");

        if (static_cast<int64_t>(obtained_records) == sample_batch_size) {
          // If we obtained a full buffer, we can emit a response directly
          ResponseT response;
          for (const auto& record : records) {
            response.add_keys(record.id);
            response.add_labels(record.column_1);
            response.add_timestamps(record.column_2);
          }

          SPDLOG_INFO("Sending with response_keys = {}, response_labels = {}, records.size = {}", response.keys_size(),
                      response.labels_size(), records.size());

          records.clear();

          {
            const std::lock_guard<std::mutex> lock(*writer_mutex);
            writer->Write(response);
          }
        } else {
          // If not, we append to our record buf
          record_buf.insert(record_buf.end(), records.begin(), records.end());
          records.clear();
          // If our record buf is big enough, emit a message
          if (static_cast<int64_t>(record_buf.size()) >= sample_batch_size) {
            ResponseT response;

            // sample_batch_size is signed int...
            for (int64_t record_idx = 0; record_idx < sample_batch_size; ++record_idx) {
              const SampleRecord& record = record_buf[record_idx];
              response.add_keys(record.id);
              response.add_labels(record.column_1);
              response.add_timestamps(record.column_2);
            }
            SPDLOG_INFO(
                "Sending with response_keys = {}, response_labels = {}, record_buf.size = {} (minus sample_batch_size "
                "= "
                "{})",
                response.keys_size(), response.labels_size(), record_buf.size(), sample_batch_size);

            // Now, delete first sample_batch_size elements from vector as we are sending them
            record_buf.erase(record_buf.begin(), record_buf.begin() + sample_batch_size);

            SPDLOG_INFO("New record_buf size = {}", record_buf.size());

            ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
                   "The record buffer should never have more than 2*sample_batch_size elements!");

            {
              const std::lock_guard<std::mutex> lock(*writer_mutex);
              writer->Write(response);
            }
          }
        }
      }

      cursor_handler.close_cursor();

      // Iterated over all files, we now need to emit all data from buffer
      if (!record_buf.empty()) {
        ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size,
               fmt::format("We should have written this buffer before! Buffer has {} items.", record_buf.size()));

        ResponseT response;
        for (const auto& record : record_buf) {
          response.add_keys(record.id);
          response.add_labels(record.column_1);
          response.add_timestamps(record.column_2);
        }
        SPDLOG_INFO("Sending with response_keys = {}, response_labels = {}, record_buf.size = {}", response.keys_size(),
                    response.labels_size(), record_buf.size());
        record_buf.clear();
        {
          const std::lock_guard<std::mutex> lock(*writer_mutex);
          writer->Write(response);
        }
      }
    }

    // sqlite causes memory leaks otherwise
    if (session.get_backend_name() != "sqlite3" && session.is_connected()) {
      session.close();
    }
  }

  template <typename WriterT = ServerWriter<modyn::storage::GetResponse>>
  static void send_sample_data_for_keys_and_file(  // NOLINT(readability-function-cognitive-complexity)
      WriterT* writer, std::mutex& writer_mutex, const std::vector<int64_t>& sample_keys,
      const DatasetData& dataset_data, soci::session& session, int64_t /*sample_batch_size*/) {
    // Note that we currently ignore the sample batch size here, under the assumption that users do not request more
    // keys than this
    try {
      const uint64_t num_keys = sample_keys.size();
      std::vector<int64_t> sample_labels(num_keys);
      std::vector<int64_t> sample_indices(num_keys);
      std::vector<int64_t> sample_fileids(num_keys);
      SPDLOG_INFO("Querying labels and files for {} samples.", num_keys);
      const std::string sample_query = fmt::format(
          "SELECT label, sample_index, file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN ({}) ORDER "
          "BY file_id",
          fmt::join(sample_keys, ","));
      session << sample_query, soci::into(sample_labels), soci::into(sample_indices), soci::into(sample_fileids),
          soci::use(dataset_data.dataset_id);
      SPDLOG_INFO("Results for {} samples obtained.", num_keys);

      int64_t current_file_id = sample_fileids[0];
      int64_t current_file_start_idx = 0;
      SPDLOG_INFO("Obtaining path for file_id {}.", current_file_id);
      std::string current_file_path;
      session << "SELECT path FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
          soci::into(current_file_path), soci::use(current_file_id), soci::use(dataset_data.dataset_id);
      SPDLOG_INFO("Path for file_id {} obtained", current_file_id);

      if (current_file_path.empty()) {
        SPDLOG_ERROR(fmt::format("Could not obtain full path of file id {} in dataset {}", current_file_id,
                                 dataset_data.dataset_id));
      }
      const YAML::Node file_wrapper_config_node = YAML::Load(dataset_data.file_wrapper_config);
      auto filesystem_wrapper =
          get_filesystem_wrapper(static_cast<FilesystemWrapperType>(dataset_data.filesystem_wrapper_type));

      auto file_wrapper =
          get_file_wrapper(current_file_path, static_cast<FileWrapperType>(dataset_data.file_wrapper_type),
                           file_wrapper_config_node, filesystem_wrapper);

      for (uint64_t sample_idx = 0; sample_idx < num_keys; ++sample_idx) {
        const int64_t& sample_fileid = sample_fileids[sample_idx];

        if (sample_fileid != current_file_id) {
          // 1. Prepare response
          SPDLOG_INFO("Encountered new file, getting data from disk");
          const std::vector<int64_t> file_indexes(sample_indices.begin() + static_cast<int64_t>(current_file_start_idx),
                                                  sample_indices.begin() + static_cast<int64_t>(sample_idx));
          const std::vector<std::vector<unsigned char>> data = file_wrapper->get_samples_from_indices(file_indexes);
          SPDLOG_INFO("Got data from disk, preparing response.");

          // Protobuf expects the data as std::string...
          std::vector<std::string> stringified_data;
          stringified_data.reserve(data.size());
          for (const std::vector<unsigned char>& char_vec : data) {
            stringified_data.emplace_back(char_vec.begin(), char_vec.end());
          }

          modyn::storage::GetResponse response;
          response.mutable_samples()->Assign(stringified_data.begin(), stringified_data.end());
          response.mutable_keys()->Assign(sample_keys.begin() + static_cast<int64_t>(current_file_start_idx),
                                          sample_keys.begin() + static_cast<int64_t>(sample_idx));
          response.mutable_labels()->Assign(sample_labels.begin() + static_cast<int64_t>(current_file_start_idx),
                                            sample_labels.begin() + static_cast<int64_t>(sample_idx));
          SPDLOG_INFO("Response prepared.");

          // 2. Send response
          {
            const std::lock_guard<std::mutex> lock(writer_mutex);
            writer->Write(response);
          }
          SPDLOG_INFO("Response sent, updating local state.");

          // 3. Update state
          current_file_id = sample_fileid;
          current_file_path = "",
          session << "SELECT path FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
          soci::into(current_file_path), soci::use(current_file_id), soci::use(dataset_data.dataset_id);
          file_wrapper->set_file_path(current_file_path);
          current_file_start_idx = static_cast<int64_t>(sample_idx);
          SPDLOG_INFO("Local state updated.");
        }
      }

      // Send leftovers
      const std::vector<int64_t> file_indexes(sample_indices.begin() + current_file_start_idx, sample_indices.end());
      const std::vector<std::vector<unsigned char>> data = file_wrapper->get_samples_from_indices(file_indexes);
      // Protobuf expects the data as std::string...
      std::vector<std::string> stringified_data;
      stringified_data.reserve(data.size());
      for (const std::vector<unsigned char>& char_vec : data) {
        stringified_data.emplace_back(char_vec.begin(), char_vec.end());
      }

      modyn::storage::GetResponse response;
      response.mutable_samples()->Assign(stringified_data.begin(), stringified_data.end());
      response.mutable_keys()->Assign(sample_keys.begin() + current_file_start_idx, sample_keys.end());
      response.mutable_labels()->Assign(sample_labels.begin() + current_file_start_idx, sample_labels.end());

      {
        const std::lock_guard<std::mutex> lock(writer_mutex);
        writer->Write(response);
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error in send_sample_data_for_keys_and_file: {}", e.what());
      throw;
    }
  }

  template <typename WriterT>
  static void get_samples_and_send(const std::vector<int64_t>::const_iterator begin,
                                   const std::vector<int64_t>::const_iterator end, WriterT* writer,
                                   std::mutex* writer_mutex, const DatasetData* dataset_data, const YAML::Node* config,
                                   int64_t sample_batch_size) {
    if (begin >= end) {
      return;
    }
    const StorageDatabaseConnection storage_database_connection(*config);
    soci::session session = storage_database_connection.get_session();
    const std::vector<int64_t> sample_keys(begin, end);
    send_sample_data_for_keys_and_file<WriterT>(writer, *writer_mutex, sample_keys, *dataset_data, session,
                                                sample_batch_size);
    session.close();
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
  static std::vector<std::pair<std::vector<int64_t>::const_iterator, std::vector<int64_t>::const_iterator>>
  get_keys_per_thread(const std::vector<int64_t>& keys, uint64_t threads);
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
