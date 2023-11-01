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
    ServerWriter<modyn::storage::GetResponse>* /*writer*/) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = -1;
    std::string base_path;
    int64_t filesystem_wrapper_type = -1;
    int64_t file_wrapper_type = -1;
    std::string file_wrapper_config;

    session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
               "datasets WHERE "
               "name = :name",
        soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type),
        soci::into(file_wrapper_type), soci::into(file_wrapper_config), soci::use(request->dataset_id());

    if (dataset_id == -1) {
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

    if (disable_multithreading_) {
    }

    // TODO(vGsteiger): Implement with cursor and lock guard on the writer

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
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
      return {StatusCode::OK, "Dataset does not exist."};
    }
    const int64_t request_timestamp = request->timestamp();
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
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
      return {StatusCode::OK, "Dataset does not exist."};
    }
    const int64_t start_timestamp = request->start_timestamp();
    const int64_t end_timestamp = request->end_timestamp();
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
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

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
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
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

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }

    if (request->keys_size() == 0) {
      SPDLOG_ERROR("No keys provided.");
      return {StatusCode::OK, "No keys provided."};
    }

    std::vector<int64_t> sample_ids(request->keys_size());
    for (int index = 0; index < request->keys_size(); index++) {
      sample_ids[index] = request->keys(index);
    }

    int64_t number_of_files = 0;

    std::string sample_placeholders = fmt::format("({})", fmt::join(sample_ids, ","));

    std::string sql = fmt::format(
        "SELECT COUNT(DISTINCT file_id) FROM (SELECT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id "
        "IN {})",
        sample_placeholders);
    session << sql, soci::into(number_of_files), soci::use(dataset_id);

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
      for (size_t i = 0; i < file_paths.size(); ++i) {
        const auto& file_id = file_ids[i];
        const auto& path = file_paths[i];
        file_wrapper->set_file_path(path);

        int64_t samples_to_delete;
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

        int64_t number_of_samples_in_file;
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
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {StatusCode::OK, "Dataset does not exist."};
    }

    int64_t total_keys = 0;
    session << "SELECT COALESCE(SUM(number_of_samples), 0) FROM files WHERE dataset_id = :dataset_id",
        soci::into(total_keys), soci::use(dataset_id);

    int64_t start_index = 0;
    int64_t limit = 0;
    std::tie(start_index, limit) = get_partition_for_worker(request->worker_id(), request->total_workers(), total_keys);

    std::vector<int64_t> keys;
    soci::statement stmt = (session.prepare << "SELECT sample_id FROM Sample WHERE dataset_id = :dataset_id ORDER BY "
                                               "sample_id OFFSET :start_index LIMIT :limit",
                            soci::use(dataset_id), soci::use(start_index), soci::use(limit));
    stmt.execute();

    int64_t key_value = 0;
    stmt.exchange(soci::into(key_value));
    while (stmt.fetch()) {
      keys.push_back(key_value);
      if (keys.size() % sample_batch_size_ == 0) {
        modyn::storage::GetDataPerWorkerResponse response;
        for (auto key : keys) {
          response.add_keys(key);
        }
        writer->Write(response);
        keys.clear();
      }
    }

    modyn::storage::GetDataPerWorkerResponse response;
    for (auto key : keys) {
      response.add_keys(key);
    }

    if (response.keys_size() > 0) {
      writer->Write(response, WriteOptions().set_last_message());
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
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

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

  const std::vector<int64_t> file_ids = get_file_ids(dataset_id, session, start_timestamp, end_timestamp);

  std::mutex writer_mutex;  // We need to protect the writer from concurrent writes as this is not supported by gRPC

  if (disable_multithreading_) {
    send_sample_id_and_label<T>(writer, writer_mutex, file_ids, storage_database_connection_, dataset_id,
                                sample_batch_size_);
  } else {
    // Split the number of files over retrieval_threads_
    auto number_of_files = static_cast<int64_t>(file_ids.size());
    const int64_t subset_size = number_of_files / retrieval_threads_;
    std::vector<std::vector<int64_t>> file_ids_per_thread(retrieval_threads_);
    for (int64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
      const int64_t start_index = thread_id * subset_size;
      const int64_t end_index = (thread_id + 1) * subset_size;
      if (thread_id == retrieval_threads_ - 1) {
        file_ids_per_thread[thread_id] = std::vector<int64_t>(file_ids.begin() + start_index, file_ids.end());
      } else {
        file_ids_per_thread[thread_id] =
            std::vector<int64_t>(file_ids.begin() + start_index, file_ids.begin() + end_index);
      }
    }

    for (int64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
      retrieval_threads_vector_[thread_id] =
          std::thread([this, writer, &file_ids_per_thread, thread_id, dataset_id, &writer_mutex]() {
            send_sample_id_and_label<T>(writer, writer_mutex, file_ids_per_thread[thread_id],
                                        std::ref(storage_database_connection_), dataset_id, sample_batch_size_);
          });
    }

    for (int64_t thread_id = 0; thread_id < retrieval_threads_; ++thread_id) {
      retrieval_threads_vector_[thread_id].join();
    }
  }
}

template <typename T>
void StorageServiceImpl::send_sample_id_and_label(ServerWriter<T>* writer, std::mutex& writer_mutex,
                                                  const std::vector<int64_t>& file_ids,
                                                  StorageDatabaseConnection& storage_database_connection,
                                                  const int64_t dataset_id, const int64_t sample_batch_size) {
  soci::session session = storage_database_connection.get_session();
  for (const int64_t file_id : file_ids) {
    const int64_t number_of_samples = get_number_of_samples_in_file(file_id, session);
    if (number_of_samples > 0) {
      const std::string query =
          fmt::format("SELECT sample_id, label FROM samples WHERE file_id = {} AND dataset_id = ", file_id, dataset_id);
      const std::string cursor_name = fmt::format("cursor_{}_{}", dataset_id, file_id);
      CursorHandler cursor_handler(session, storage_database_connection.get_drivername(), query, cursor_name, 2);

      std::vector<SampleRecord> records(sample_batch_size);

      while (true) {
        records = cursor_handler.yield_per(sample_batch_size);
        if (records.empty()) {
          break;
        }
        T response;
        for (const auto& record : records) {
          response.add_keys(record.id);
          response.add_labels(record.label);
        }

        const std::lock_guard<std::mutex> lock(writer_mutex);
        writer->Write(response);
      }
    }
  }
}

int64_t StorageServiceImpl::get_number_of_samples_in_file(int64_t file_id, soci::session& session) {
  int64_t number_of_samples = 0;
  session << "SELECT number_of_samples FROM files WHERE file_id = :file_id", soci::into(number_of_samples),
      soci::use(file_id);
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

int64_t StorageServiceImpl::get_dataset_id(const std::string& dataset_name, soci::session& session) {
  int64_t dataset_id =
      -1;  // NOLINT misc-const-correctness  (the variable cannot be const to be usable as filling variable by soci)
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

  return dataset_id;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids(const int64_t dataset_id, soci::session& session,
                                                      const int64_t start_timestamp, const int64_t end_timestamp) {
  const int64_t number_of_files = get_file_count(session, dataset_id, start_timestamp, end_timestamp);
  if (number_of_files == 0) {
    return {};
  }

  return get_file_ids(session, dataset_id, start_timestamp, end_timestamp, number_of_files);
}

int64_t StorageServiceImpl::get_file_count(soci::session& session, const int64_t dataset_id,
                                           const int64_t start_timestamp, const int64_t end_timestamp) {
  int64_t number_of_files = -1;
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
  return number_of_files;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids(soci::session& session, const int64_t dataset_id,
                                                      const int64_t start_timestamp, const int64_t end_timestamp,
                                                      const int64_t number_of_files) {
  std::vector<int64_t> file_ids(number_of_files + 1);

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
    session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id", soci::into(file_ids), soci::use(dataset_id);
  }

  return file_ids;
}
