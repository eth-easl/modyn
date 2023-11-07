#include "internal/grpc/storage_service_impl.hpp"

#include <mutex>

#include "internal/database/cursor_handler.hpp"
#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"
#include "modyn/utils/utils.hpp"

using namespace modyn::storage;

// ------- StorageServiceImpl -------

Status StorageServiceImpl::Get(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetRequest* request,
    ServerWriter<modyn::storage::GetResponse>* writer) {
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

    send_sample_data_from_keys(writer, request_keys, dataset_data, session,
                               storage_database_connection_.get_drivername());

    return {StatusCode::OK, "Data retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in Get: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in Get: {}", e.what())};
  }
}

Status StorageServiceImpl::GetNewDataSince(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetNewDataSinceRequest* request,
    ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {
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

    send_file_ids_and_labels<modyn::storage::GetNewDataSinceResponse>(writer, dataset_id, request_timestamp);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetNewDataSince: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetNewDataSince: {}", e.what())};
  }
  return {StatusCode::OK, "Data retrieved."};
}

Status StorageServiceImpl::GetDataInInterval(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetDataInIntervalRequest* request,
    ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) {
  try {
    soci::session session = storage_database_connection_.get_session();
    const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
      return {StatusCode::OK, "Dataset does not exist."};
    }
    const int64_t start_timestamp = request->start_timestamp();
    const int64_t end_timestamp = request->end_timestamp();

    SPDLOG_INFO(fmt::format("Received GetDataInInterval Request for dataset {} (id = {}) with start = {} and end = {}.",
                            request->dataset_id(), dataset_id, start_timestamp, end_timestamp));

    send_file_ids_and_labels<modyn::storage::GetDataInIntervalResponse>(writer, dataset_id, start_timestamp,
                                                                        end_timestamp);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDataInInterval: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetDataInInterval: {}", e.what())};
  }
  return {StatusCode::OK, "Data retrieved."};
}

Status StorageServiceImpl::CheckAvailability(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DatasetAvailableResponse* response) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
    SPDLOG_INFO(fmt::format("Received availability request for dataset {}", dataset_id));

    if (dataset_id == -1) {
      response->set_available(false);
      return {StatusCode::OK, "Dataset does not exist."};
    }
    response->set_available(true);
    return {StatusCode::OK, "Dataset exists."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in CheckAvailability: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in CheckAvailability: {}", e.what())};
  }
}

Status StorageServiceImpl::RegisterNewDataset(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::RegisterNewDatasetRequest* request,
    modyn::storage::RegisterNewDatasetResponse* response) {
  try {
    SPDLOG_INFO(fmt::format("Received register new dataset request for {} at {}.", request->dataset_id(),
                            request->base_path()));
    const bool success = storage_database_connection_.add_dataset(
        request->dataset_id(), request->base_path(),
        FilesystemWrapper::get_filesystem_wrapper_type(request->filesystem_wrapper_type()),
        FileWrapper::get_file_wrapper_type(request->file_wrapper_type()), request->description(), request->version(),
        request->file_wrapper_config(), request->ignore_last_timestamp(),
        static_cast<int>(request->file_watcher_interval()));
    response->set_success(success);
    return Status::OK;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in RegisterNewDataset: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in RegisterNewDataset: {}", e.what())};
  }
}

Status StorageServiceImpl::GetCurrentTimestamp(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetCurrentTimestampRequest* /*request*/,
    modyn::storage::GetCurrentTimestampResponse* response) {
  try {
    SPDLOG_INFO("ReceivedGetCurrentTimestamp request.");
    response->set_timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
    return {StatusCode::OK, "Timestamp retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetCurrentTimestamp: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetCurrentTimestamp: {}", e.what())};
  }
}

Status StorageServiceImpl::DeleteDataset(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DeleteDatasetResponse* response) {
  try {
    response->set_success(false);
    int64_t filesystem_wrapper_type;

    soci::session session = storage_database_connection_.get_session();
    int64_t dataset_id = get_dataset_id(session, request->dataset_id());
    SPDLOG_INFO(fmt::format("Received DeleteDataset Request for dataset {}", dataset_id));
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }
    session << "SELECT filesystem_wrapper_type FROM datasets WHERE name = :name", soci::into(filesystem_wrapper_type),
        soci::use(request->dataset_id());

    auto filesystem_wrapper = get_filesystem_wrapper(static_cast<FilesystemWrapperType>(filesystem_wrapper_type));

    int64_t number_of_files = 0;
    session << "SELECT COUNT(file_id) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
        soci::use(dataset_id);

    if (number_of_files > 0) {
      std::vector<std::string> file_paths(number_of_files + 1);
      session << "SELECT path FROM files WHERE dataset_id = :dataset_id", soci::into(file_paths), soci::use(dataset_id);

      try {
        for (const auto& file_path : file_paths) {
          filesystem_wrapper->remove(file_path);
        }
      } catch (const modyn::utils::ModynException& e) {
        SPDLOG_ERROR("Error deleting dataset: {}", e.what());
        return {StatusCode::OK, "Error deleting dataset."};
      }
    }

    const bool success = storage_database_connection_.delete_dataset(request->dataset_id(), dataset_id);

    response->set_success(success);
    return Status::OK;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in DeleteDataset: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in DeleteDataset: {}", e.what())};
  }
}

Status StorageServiceImpl::DeleteData(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::DeleteDataRequest* request,
    modyn::storage::DeleteDataResponse* response) {
  try {
    response->set_success(false);
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = -1;
    std::string base_path;
    int64_t filesystem_wrapper_type = -1;
    int64_t file_wrapper_type = -1;
    std::string file_wrapper_config;
    session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
               "datasets WHERE name = :name",
        soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type),
        soci::into(file_wrapper_type), soci::into(file_wrapper_config), soci::use(request->dataset_id());

    SPDLOG_INFO(fmt::format("Received DeleteData Request for dataset {}", dataset_id));

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }

    if (request->keys_size() == 0) {
      SPDLOG_ERROR("No keys provided.");
      return {StatusCode::OK, "No keys provided."};
    }

    std::vector<int64_t> sample_ids(request->keys_size());
    for (int64_t index = 0; index < request->keys_size(); ++index) {
      sample_ids[index] = request->keys(index);
    }

    int64_t number_of_files = 0;
    std::string sample_placeholders = fmt::format("({})", fmt::join(sample_ids, ","));

    std::string sql = fmt::format(
        "SELECT COUNT(DISTINCT file_id) FROM samples WHERE dataset_id = :dataset_id AND "
        "sample_id IN {}",
        sample_placeholders);
    session << sql, soci::into(number_of_files), soci::use(dataset_id);
    SPDLOG_INFO(fmt::format("DeleteData Request for dataset {} found {} relevant files", dataset_id, number_of_files));

    if (number_of_files == 0) {
      SPDLOG_ERROR("No samples found in dataset {}.", dataset_id);
      return {StatusCode::OK, "No samples found."};
    }

    // Get the file ids
    std::vector<int64_t> file_ids(number_of_files + 1);
    sql = fmt::format("SELECT DISTINCT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN {}",
                      sample_placeholders);
    session << sql, soci::into(file_ids), soci::use(dataset_id);

    if (file_ids.empty()) {
      SPDLOG_ERROR("No files found in dataset {}.", dataset_id);
      return {StatusCode::OK, "No files found."};
    }

    auto filesystem_wrapper = get_filesystem_wrapper(static_cast<FilesystemWrapperType>(filesystem_wrapper_type));
    const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
    std::string file_placeholders = fmt::format("({})", fmt::join(file_ids, ","));
    std::string index_placeholders;

    try {
      std::vector<std::string> file_paths(number_of_files + 1);
      sql = fmt::format("SELECT path FROM files WHERE file_id IN {}", file_placeholders);
      session << sql, soci::into(file_paths);
      if (file_paths.size() != file_ids.size()) {
        SPDLOG_ERROR("Error deleting data: Could not find all files.");
        return {StatusCode::OK, "Error deleting data."};
      }

      auto file_wrapper = get_file_wrapper(file_paths.front(), static_cast<FileWrapperType>(file_wrapper_type),
                                           file_wrapper_config_node, filesystem_wrapper);
      for (uint64_t i = 0; i < file_paths.size(); ++i) {
        const auto& file_id = file_ids[i];
        const auto& path = file_paths[i];
        SPDLOG_INFO(
            fmt::format("DeleteData Request for dataset {} handling path {} (file id {})", dataset_id, path, file_id));

        file_wrapper->set_file_path(path);

        int64_t samples_to_delete = 0;
        sql = fmt::format("SELECT COUNT(sample_id) FROM samples WHERE file_id = :file_id AND sample_id IN {}",
                          sample_placeholders);
        session << sql, soci::into(samples_to_delete), soci::use(file_id);

        std::vector<int64_t> sample_ids_to_delete_ids(samples_to_delete + 1);
        sql = fmt::format("SELECT sample_id FROM samples WHERE file_id = :file_id AND sample_id IN {}",
                          sample_placeholders);
        session << sql, soci::into(sample_ids_to_delete_ids), soci::use(file_id);

        file_wrapper->delete_samples(sample_ids_to_delete_ids);

        index_placeholders = fmt::format("({})", fmt::join(sample_ids_to_delete_ids, ","));
        sql = fmt::format("DELETE FROM samples WHERE file_id = :file_id AND sample_id IN {}", index_placeholders);
        session << sql, soci::use(file_id);

        int64_t number_of_samples_in_file = 0;
        session << "SELECT number_of_samples FROM files WHERE file_id = :file_id",
            soci::into(number_of_samples_in_file), soci::use(file_id);

        if (number_of_samples_in_file - samples_to_delete == 0) {
          session << "DELETE FROM files WHERE file_id = :file_id", soci::use(file_id);
          filesystem_wrapper->remove(path);
        } else {
          session << "UPDATE files SET number_of_samples = :number_of_samples WHERE file_id = :file_id",
              soci::use(number_of_samples_in_file - samples_to_delete), soci::use(file_id);
        }
      }
    } catch (const std::exception& e) {
      SPDLOG_ERROR("Error deleting data: {}", e.what());
      return {StatusCode::OK, "Error deleting data."};
    }
    response->set_success(true);
    return {StatusCode::OK, "Data deleted."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in DeleteData: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in DeleteData: {}", e.what())};
  }
}

Status StorageServiceImpl::GetDataPerWorker(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetDataPerWorkerRequest* request,
    ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = get_dataset_id(session, request->dataset_id());

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }
    SPDLOG_INFO(
        fmt::format("Received GetDataPerWorker Request for dataset {} (id = {}) and worker {} out of {} workers",
                    request->dataset_id(), dataset_id, request->worker_id(), request->total_workers()));

    int64_t total_keys = 0;
    session << "SELECT COALESCE(SUM(number_of_samples), 0) FROM files WHERE dataset_id = :dataset_id",
        soci::into(total_keys), soci::use(dataset_id);

    if (total_keys > 0) {
      int64_t start_index = 0;
      int64_t limit = 0;
      std::tie(start_index, limit) =
          get_partition_for_worker(request->worker_id(), request->total_workers(), total_keys);

      const std::string query =
          fmt::format("SELECT sample_id FROM samples WHERE dataset_id = {} ORDER BY sample_id OFFSET {} LIMIT {}",
                      dataset_id, start_index, limit);
      const std::string cursor_name = fmt::format("pw_cursor_{}_{}", dataset_id, request->worker_id());
      CursorHandler cursor_handler(session, storage_database_connection_.get_drivername(), query, cursor_name, 1);

      std::vector<SampleRecord> records;
      std::vector<SampleRecord> record_buf;
      record_buf.reserve(sample_batch_size_);

      while (true) {
        records = cursor_handler.yield_per(sample_batch_size_);

        SPDLOG_INFO(fmt::format("got {} records (batch size = {})", records.size(), sample_batch_size_));
        if (records.empty()) {
          break;
        }

        const uint64_t obtained_records = records.size();
        ASSERT(static_cast<int64_t>(obtained_records) <= sample_batch_size_, "Received too many samples");

        if (static_cast<int64_t>(records.size()) == sample_batch_size_) {
          // If we obtained a full buffer, we can emit a response directly
          modyn::storage::GetDataPerWorkerResponse response;
          for (const auto& record : records) {
            response.add_keys(record.id);
          }

          writer->Write(response);
        } else {
          // If not, we append to our record buf
          record_buf.insert(record_buf.end(), records.begin(), records.end());
          // If our record buf is big enough, emit a message
          if (static_cast<int64_t>(records.size()) >= sample_batch_size_) {
            modyn::storage::GetDataPerWorkerResponse response;

            // sample_batch_size is signed int...
            for (int64_t record_idx = 0; record_idx < sample_batch_size_; ++record_idx) {
              const SampleRecord& record = record_buf[record_idx];
              response.add_keys(record.id);
            }

            // Now, delete first sample_batch_size elements from vector as we are sending them
            record_buf.erase(record_buf.begin(), record_buf.begin() + sample_batch_size_);

            ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size_,
                   "The record buffer should never have more than 2*sample_batch_size elements!");

            writer->Write(response);
          }
        }
      }

      if (!record_buf.empty()) {
        ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size_,
               "We should have written this buffer before!");

        modyn::storage::GetDataPerWorkerResponse response;
        for (const auto& record : record_buf) {
          response.add_keys(record.id);
        }

        writer->Write(response);
      }
    }

    return {StatusCode::OK, "Data retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDataPerWorker: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetDataPerWorker: {}", e.what())};
  }
}

Status StorageServiceImpl::GetDatasetSize(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetDatasetSizeRequest* request,
    modyn::storage::GetDatasetSizeResponse* response) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = get_dataset_id(session, request->dataset_id());

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }

    int64_t total_keys = 0;
    session << "SELECT COALESCE(SUM(number_of_samples), 0) FROM files WHERE dataset_id = :dataset_id",
        soci::into(total_keys), soci::use(dataset_id);

    response->set_num_keys(total_keys);
    response->set_success(true);
    return {StatusCode::OK, "Dataset size retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDatasetSize: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetDatasetSize: {}", e.what())};
  }
}

// ------- Helper functions -------

template <typename T>
void StorageServiceImpl::send_file_ids_and_labels(ServerWriter<T>* writer, const int64_t dataset_id,
                                                  const int64_t start_timestamp, int64_t end_timestamp) {
  soci::session session = storage_database_connection_.get_session();

  const std::vector<int64_t> file_ids = get_file_ids(session, dataset_id, start_timestamp, end_timestamp);
  SPDLOG_INFO(fmt::format("send_file_ids_and_labels got {} file ids.", file_ids.size()));

  std::mutex writer_mutex;  // We need to protect the writer from concurrent writes as this is not supported by gRPC

  if (disable_multithreading_) {
    send_sample_id_and_label<T>(writer, writer_mutex, file_ids, storage_database_connection_, dataset_id,
                                sample_batch_size_);
  } else {
    // Split the number of files over retrieval_threads_
    auto file_ids_per_thread = get_file_ids_per_thread(file_ids, retrieval_threads_);

    std::vector<std::thread> retrieval_threads_vector(retrieval_threads_);
    for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
      retrieval_threads_vector[thread_id] =
          std::thread([this, writer, &file_ids_per_thread, thread_id, dataset_id, &writer_mutex]() {
            send_sample_id_and_label<T>(writer, writer_mutex, file_ids_per_thread[thread_id],
                                        std::ref(storage_database_connection_), dataset_id, sample_batch_size_);
          });
    }

    for (uint64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
      retrieval_threads_vector[thread_id].join();
    }
  }
}

template <typename T>
void StorageServiceImpl::send_sample_id_and_label(ServerWriter<T>* writer, std::mutex& writer_mutex,
                                                  const std::vector<int64_t>& file_ids,
                                                  StorageDatabaseConnection& storage_database_connection,
                                                  const int64_t dataset_id, const int64_t sample_batch_size) {
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
          T response;
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
            T response;

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

    T response;
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

void StorageServiceImpl::send_sample_data_from_keys(ServerWriter<modyn::storage::GetResponse>* writer,
                                                    const std::vector<int64_t>& request_keys,
                                                    const DatasetData& dataset_data, soci::session& session,
                                                    const DatabaseDriver& driver) {
  // TODO(maxiBoether): we need to benchmark this. In Python, we just get all samples from the DB and then fetch then
  // from disk. Here, we first have to get all files with a big subq, then all samples for each file again. Not sure if
  // this is faster instead of one big query and then parallelizing over that result.
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
      send_sample_data_for_keys_and_file(writer, writer_mutex, file_id, samples_corresponding_to_file, dataset_data,
                                         session, driver, sample_batch_size_);
    }
  } else {
    std::vector<std::vector<int64_t>> file_ids_per_thread = get_file_ids_per_thread(file_ids, retrieval_threads_);

    auto thread_function = [this, writer, &writer_mutex, &file_ids_per_thread, &request_keys, &dataset_data, &session,
                            &driver](int64_t thread_id) {
      for (const int64_t file_id : file_ids_per_thread[thread_id]) {
        const std::vector<int64_t>& samples_corresponding_to_file =
            get_samples_corresponding_to_file(file_id, dataset_data.dataset_id, request_keys, session);
        send_sample_data_for_keys_and_file(writer, writer_mutex, file_id, samples_corresponding_to_file, dataset_data,
                                           session, driver, sample_batch_size_);
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

std::vector<std::vector<int64_t>> StorageServiceImpl::get_file_ids_per_thread(const std::vector<int64_t>& file_ids,
                                                                              const uint64_t retrieval_threads) {
  ASSERT(retrieval_threads > 0, "This function is only intended for multi-threaded retrieval.");
  std::vector<std::vector<int64_t>> file_ids_per_thread(retrieval_threads);
  try {
    auto number_of_files = static_cast<uint64_t>(file_ids.size());
    const uint64_t subset_size = (number_of_files + retrieval_threads - 1) / retrieval_threads;
    for (uint64_t thread_id = 0; thread_id < retrieval_threads; ++thread_id) {
      const uint64_t start_index = thread_id * subset_size;
      const uint64_t end_index = (thread_id + 1) * subset_size;
      if (thread_id == retrieval_threads - 1) {
        file_ids_per_thread[thread_id] = std::vector<int64_t>(file_ids.begin() + start_index, file_ids.end());
      } else {
        file_ids_per_thread[thread_id] =
            std::vector<int64_t>(file_ids.begin() + start_index, file_ids.begin() + end_index);
      }
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in get_file_ids_per_thread with file_ids.size() = {}, retrieval_theads = {}: {}",
                 file_ids.size(), retrieval_threads, e.what());
    throw;
  }
  return file_ids_per_thread;
}

void StorageServiceImpl::send_sample_data_for_keys_and_file(ServerWriter<modyn::storage::GetResponse>* writer,
                                                            std::mutex& writer_mutex, const int64_t file_id,
                                                            const std::vector<int64_t>& request_keys_per_file,
                                                            const DatasetData& dataset_data, soci::session& session,
                                                            const DatabaseDriver& driver,
                                                            const int64_t sample_batch_size) {
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
      ASSERT(static_cast<int64_t>(record_buf.size()) < sample_batch_size, "We should have written this buffer before!");
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

std::vector<int64_t> StorageServiceImpl::get_samples_corresponding_to_file(const int64_t file_id,
                                                                           const int64_t dataset_id,
                                                                           const std::vector<int64_t>& request_keys,
                                                                           soci::session& session) {
  const auto number_of_samples = static_cast<uint64_t>(request_keys.size());
  std::vector<int64_t> sample_ids(number_of_samples + 1);

  try {
    const std::string sample_placeholders = fmt::format("({})", fmt::join(request_keys, ","));

    const std::string sql = fmt::format(
        "SELECT sample_id FROM samples WHERE file_id = :file_id AND dataset_id = "
        ":dataset_id AND sample_id IN {}",
        sample_placeholders);
    session << sql, soci::into(sample_ids), soci::use(file_id), soci::use(dataset_id);
  } catch (const std::exception& e) {
    SPDLOG_ERROR(
        "Error in get_samples_corresponding_to_file with file_id = {}, dataset_id = {}, number_of_samples = {}: {}",
        file_id, dataset_id, number_of_samples, e.what());
    throw;
  }
  return sample_ids;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids_for_samples(const std::vector<int64_t>& request_keys,
                                                                  const int64_t dataset_id, soci::session& session) {
  const auto number_of_samples = static_cast<int64_t>(request_keys.size());
  const std::string sample_placeholders = fmt::format("({})", fmt::join(request_keys, ","));

  const std::string sql = fmt::format(
      "SELECT DISTINCT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN {}", sample_placeholders);
  std::vector<int64_t> file_ids(number_of_samples + 1);
  session << sql, soci::into(file_ids), soci::use(dataset_id);

  return file_ids;
}

int64_t StorageServiceImpl::get_number_of_samples_in_file(int64_t file_id, soci::session& session,
                                                          const int64_t dataset_id) {
  int64_t number_of_samples = 0;
  int64_t number_of_rows = 0;
  // TODO remove this debug code
  session << "SELECT COUNT(*) FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
      soci::into(number_of_rows), soci::use(file_id), soci::use(dataset_id);

  if (number_of_rows != 1) {
    SPDLOG_ERROR(fmt::format("Warning! Number of rows for file id {}, dataset id {} == {}", file_id, dataset_id,
                             number_of_rows));
    return number_of_samples;
  }

  session << "SELECT number_of_samples FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
      soci::into(number_of_samples), soci::use(file_id), soci::use(dataset_id);
  return number_of_samples;
}

std::tuple<int64_t, int64_t> StorageServiceImpl::get_partition_for_worker(const int64_t worker_id,
                                                                          const int64_t total_workers,
                                                                          const int64_t total_num_elements) {
  if (worker_id < 0 || worker_id >= total_workers) {
    FAIL("Worker id must be between 0 and total_workers - 1.");
  }

  const int64_t subset_size = total_num_elements / total_workers;
  int64_t worker_subset_size = subset_size;

  const int64_t threshold = total_num_elements % total_workers;
  if (threshold > 0) {
    if (worker_id < threshold) {
      worker_subset_size += 1;
      const int64_t start_index = worker_id * (subset_size + 1);
      return {start_index, worker_subset_size};
    }
    const int64_t start_index = threshold * (subset_size + 1) + (worker_id - threshold) * subset_size;
    return {start_index, worker_subset_size};
  }
  const int64_t start_index = worker_id * subset_size;
  return {start_index, worker_subset_size};
}

int64_t StorageServiceImpl::get_dataset_id(soci::session& session, const std::string& dataset_name) {
  int64_t dataset_id = -1;
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

  return dataset_id;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids(soci::session& session, const int64_t dataset_id,
                                                      const int64_t start_timestamp, const int64_t end_timestamp) {
  const int64_t number_of_files = get_file_count(session, dataset_id, start_timestamp, end_timestamp);
  if (number_of_files == 0) {
    return {};
  }

  if (number_of_files < 0) {
    SPDLOG_ERROR(fmt::format("Number of files for dataset {} is below zero: {}", dataset_id, number_of_files));
    return {};
  }

  return get_file_ids_given_number_of_files(session, dataset_id, start_timestamp, end_timestamp, number_of_files);
}

int64_t StorageServiceImpl::get_file_count(soci::session& session, const int64_t dataset_id,
                                           const int64_t start_timestamp, const int64_t end_timestamp) {
  // TODO(MaxiBoether): DOesn'T this slow down because we are almost excecuting the same query twice? Can we get all
  // files into a vector without knowing how many?
  int64_t number_of_files = -1;
  try {
    if (start_timestamp >= 0 && end_timestamp == -1) {
      session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp",
          soci::into(number_of_files), soci::use(dataset_id), soci::use(start_timestamp);
    } else if (start_timestamp == -1 && end_timestamp >= 0) {
      session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at <= :end_timestamp",
          soci::into(number_of_files), soci::use(dataset_id), soci::use(end_timestamp);
    } else if (start_timestamp >= 0 && end_timestamp >= 0) {
      session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp AND "
                 "updated_at <= :end_timestamp",
          soci::into(number_of_files), soci::use(dataset_id), soci::use(start_timestamp), soci::use(end_timestamp);
    } else {
      session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
          soci::use(dataset_id);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in get_file_count with dataset_id = {}, start_timestamp = {}, end_timestamp = {}: {}",
                 dataset_id, start_timestamp, end_timestamp, e.what());
    throw;
  }
  return number_of_files;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids_given_number_of_files(soci::session& session,
                                                                            const int64_t dataset_id,
                                                                            const int64_t start_timestamp,
                                                                            const int64_t end_timestamp,
                                                                            const int64_t number_of_files) {
  ASSERT(number_of_files >= 0, "This function should only be called for a non-negative number of files");
  std::vector<int64_t> file_ids(number_of_files + 1);

  try {
    if (start_timestamp >= 0 && end_timestamp == -1) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp",
          soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp);
    } else if (start_timestamp == -1 && end_timestamp >= 0) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at <= :end_timestamp",
          soci::into(file_ids), soci::use(dataset_id), soci::use(end_timestamp);
    } else if (start_timestamp >= 0 && end_timestamp >= 0) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp AND "
                 "updated_at <= :end_timestamp",
          soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp), soci::use(end_timestamp);
    } else {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id", soci::into(file_ids),
          soci::use(dataset_id);
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR(
        "Error in get_file_ids_given_number_of_files with dataset_id = {}, start_timestamp = {}, end_timestamp = {}, "
        "number_of_files = {}: {}",
        dataset_id, start_timestamp, end_timestamp, number_of_files, e.what());
    throw;
  }
  return file_ids;
}

DatasetData StorageServiceImpl::get_dataset_data(soci::session& session, std::string& dataset_name) {
  int64_t dataset_id = -1;
  std::string base_path;
  auto filesystem_wrapper_type = static_cast<int64_t>(FilesystemWrapperType::INVALID_FSW);
  auto file_wrapper_type = static_cast<int64_t>(FileWrapperType::INVALID_FW);
  std::string file_wrapper_config;

  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE "
             "name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(dataset_name);

  return {dataset_id, base_path, static_cast<FilesystemWrapperType>(filesystem_wrapper_type),
          static_cast<FileWrapperType>(file_wrapper_type), file_wrapper_config};
}