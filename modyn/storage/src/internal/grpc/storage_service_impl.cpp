#include "internal/grpc/storage_service_impl.hpp"

#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"
#include "internal/utils/utils.hpp"

using namespace storage::grpcs;

// ------- StorageServiceImpl -------

::grpc::Status StorageServiceImpl::Get(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetRequest* request,
    ::grpc::ServerWriter<modyn::storage::GetResponse>* writer) {
  try {
    SPDLOG_INFO("Get request received.");
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id;
    std::string base_path;
    int64_t filesystem_wrapper_type;
    int64_t file_wrapper_type;
    std::string file_wrapper_config;

    session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM datasets WHERE "
               "name = :name",
        soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
        soci::into(file_wrapper_config), soci::use(request->dataset_id());

    const int keys_size = request->keys_size();
    std::vector<int64_t> request_keys(keys_size + 1);
    for (int i = 0; i < keys_size; i++) {
      request_keys[i] = request->keys(i);
    }

    // TODO(vGsteiger): Implement with new parallelization scheme used in GetNewDataSince and GetDataInInterval

    return {::grpc::StatusCode::OK, "Data retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in Get: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in Get: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::GetNewDataSince(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetNewDataSinceRequest* request,
    ::grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {
  try {
    soci::session session = storage_database_connection_.get_session();
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }
    int64_t request_timestamp = request->timestamp();  // NOLINT misc-const-correctness
    send_file_ids_and_labels<modyn::storage::GetNewDataSinceResponse>(writer, dataset_id, request_timestamp);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetNewDataSince: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in GetNewDataSince: {}", e.what())};
  }
  return {::grpc::StatusCode::OK, "Data retrieved."};
}

::grpc::Status StorageServiceImpl::GetDataInInterval(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetDataInIntervalRequest* request,
    ::grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) {
  SPDLOG_INFO("GetDataInInterval request received.");
  try {
    soci::session session = storage_database_connection_.get_session();
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", dataset_id);
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }
    int64_t start_timestamp = request->start_timestamp();  // NOLINT misc-const-correctness
    int64_t end_timestamp = request->end_timestamp();      // NOLINT misc-const-correctness
    send_file_ids_and_labels<modyn::storage::GetDataInIntervalResponse>(writer, dataset_id, start_timestamp,
                                                                        end_timestamp);
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDataInInterval: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in GetDataInInterval: {}", e.what())};
  }
  SPDLOG_INFO("GetDataInInterval request finished.");
  return {::grpc::StatusCode::OK, "Data retrieved."};
}

::grpc::Status StorageServiceImpl::CheckAvailability(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DatasetAvailableResponse* response) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    const int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

    if (dataset_id == -1) {
      response->set_available(false);
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }
    response->set_available(true);
    return {::grpc::StatusCode::OK, "Dataset exists."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in CheckAvailability: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in CheckAvailability: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::RegisterNewDataset(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::RegisterNewDatasetRequest* request,
    modyn::storage::RegisterNewDatasetResponse* response) {
  try {
    bool success = storage_database_connection_.add_dataset(  // NOLINT misc-const-correctness
        request->dataset_id(), request->base_path(),
        storage::filesystem_wrapper::FilesystemWrapper::get_filesystem_wrapper_type(request->filesystem_wrapper_type()),
        storage::file_wrapper::FileWrapper::get_file_wrapper_type(request->file_wrapper_type()), request->description(),
        request->version(), request->file_wrapper_config(), request->ignore_last_timestamp(),
        static_cast<int>(request->file_watcher_interval()));
    response->set_success(success);
    return ::grpc::Status::OK;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in RegisterNewDataset: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in RegisterNewDataset: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::GetCurrentTimestamp(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetCurrentTimestampRequest* /*request*/,
    modyn::storage::GetCurrentTimestampResponse* response) {
  try {
    response->set_timestamp(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
    return {::grpc::StatusCode::OK, "Timestamp retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetCurrentTimestamp: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in GetCurrentTimestamp: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::DeleteDataset(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DeleteDatasetResponse* response) {
  try {
    response->set_success(false);
    std::string base_path;
    int64_t filesystem_wrapper_type;

    soci::session session = storage_database_connection_.get_session();
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
    session << "SELECT base_path, filesystem_wrapper_type FROM datasets WHERE name = :name", soci::into(base_path),
        soci::into(filesystem_wrapper_type), soci::use(request->dataset_id());

    auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
        base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));

    int64_t number_of_files;
    session << "SELECT COUNT(file_id) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
        soci::use(dataset_id);

    if (number_of_files > 0) {
      std::vector<std::string> file_paths(number_of_files + 1);
      session << "SELECT path FROM files WHERE dataset_id = :dataset_id", soci::into(file_paths), soci::use(dataset_id);

      try {
        for (const auto& file_path : file_paths) {
          filesystem_wrapper->remove(file_path);
        }
      } catch (const storage::utils::ModynException& e) {
        SPDLOG_ERROR("Error deleting dataset: {}", e.what());
        return {::grpc::StatusCode::OK, "Error deleting dataset."};
      }
    }

    const bool success = storage_database_connection_.delete_dataset(request->dataset_id(),
                                                                     dataset_id);  // NOLINT misc-const-correctness

    response->set_success(success);
    return ::grpc::Status::OK;
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in DeleteDataset: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in DeleteDataset: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::DeleteData(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::DeleteDataRequest* request,
    modyn::storage::DeleteDataResponse* response) {
  try {
    response->set_success(false);
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = -1;
    std::string base_path;
    int64_t filesystem_wrapper_type;
    int64_t file_wrapper_type;
    std::string file_wrapper_config;
    session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
               "datasets WHERE name = :name",
        soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type),
        soci::into(file_wrapper_type), soci::into(file_wrapper_config), soci::use(request->dataset_id());

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }

    if (request->keys_size() == 0) {
      SPDLOG_ERROR("No keys provided.");
      return {::grpc::StatusCode::OK, "No keys provided."};
    }

    std::vector<int64_t> sample_ids(request->keys_size() + 1);
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
      return {::grpc::StatusCode::OK, "No samples found."};
    }

    // Get the file ids
    std::vector<int64_t> file_ids(number_of_files + 1);
    sql = fmt::format("SELECT DISTINCT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN {}",
                      sample_placeholders);
    session << sql, soci::into(file_ids), soci::use(dataset_id);

    if (file_ids.empty()) {
      SPDLOG_ERROR("No files found in dataset {}.", dataset_id);
      return {::grpc::StatusCode::OK, "No files found."};
    }

    auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
        base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));
    const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
    std::string file_placeholders = fmt::format("({})", fmt::join(file_ids, ","));
    std::string index_placeholders;

    try {
      std::vector<std::string> file_paths(number_of_files + 1);
      sql = fmt::format("SELECT path FROM files WHERE file_id IN {}", file_placeholders);
      session << sql, soci::into(file_paths);
      if (file_paths.size() != file_ids.size()) {
        SPDLOG_ERROR("Error deleting data: Could not find all files.");
        return {::grpc::StatusCode::OK, "Error deleting data."};
      }

      auto file_wrapper = storage::file_wrapper::get_file_wrapper(
          file_paths.front(), static_cast<storage::file_wrapper::FileWrapperType>(file_wrapper_type),
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
      return {::grpc::StatusCode::OK, "Error deleting data."};
    }
    response->set_success(true);
    return {::grpc::StatusCode::OK, "Data deleted."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in DeleteData: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in DeleteData: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::GetDataPerWorker(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetDataPerWorkerRequest* request,
    ::grpc::ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) {  // NOLINT misc-const-correctness
  try {
    SPDLOG_INFO("GetDataPerWorker request received.");
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }

    int64_t total_keys = 0;
    session << "SELECT COALESCE(SUM(number_of_samples), 0) FROM files WHERE dataset_id = :dataset_id", soci::into(total_keys),
        soci::use(dataset_id);

    int64_t start_index;
    int64_t limit;
    std::tie(start_index, limit) = get_partition_for_worker(request->worker_id(), request->total_workers(), total_keys);

    std::vector<int64_t> keys;
    soci::statement stmt = (session.prepare << "SELECT sample_id FROM Sample WHERE dataset_id = :dataset_id ORDER BY "
                                               "sample_id OFFSET :start_index LIMIT :limit",
                            soci::use(dataset_id), soci::use(start_index), soci::use(limit));
    stmt.execute();

    int64_t key_value;
    stmt.exchange(soci::into(key_value));
    while (stmt.fetch()) {
      keys.push_back(key_value);
    }

    modyn::storage::GetDataPerWorkerResponse response;
    for (auto key : keys) {
      response.add_keys(key);
      if (response.keys_size() % sample_batch_size_ == 0) {
        writer->Write(response);
        response.clear_keys();
      }
    }

    if (response.keys_size() > 0) {
      writer->Write(response, ::grpc::WriteOptions().set_last_message());
    }

    return {::grpc::StatusCode::OK, "Data retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDataPerWorker: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in GetDataPerWorker: {}", e.what())};
  }
}

::grpc::Status StorageServiceImpl::GetDatasetSize(  // NOLINT readability-identifier-naming
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetDatasetSizeRequest* request,
    modyn::storage::GetDatasetSizeResponse* response) {  // NOLINT misc-const-correctness
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      return {::grpc::StatusCode::OK, "Dataset does not exist."};
    }

    int64_t total_keys = 0;
    session << "SELECT COALESCE(SUM(number_of_samples), 0) FROM files WHERE dataset_id = :dataset_id", soci::into(total_keys),
        soci::use(dataset_id);

    response->set_num_keys(total_keys);
    response->set_success(true);
    return {::grpc::StatusCode::OK, "Dataset size retrieved."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in GetDatasetSize: {}", e.what());
    return {::grpc::StatusCode::OK, fmt::format("Error in GetDatasetSize: {}", e.what())};
  }
}

// ------- Helper functions -------

template <typename T>
void StorageServiceImpl::send_file_ids_and_labels(::grpc::ServerWriter<T>* writer, int64_t dataset_id,
                                                  int64_t start_timestamp, int64_t end_timestamp) {
  soci::session session = storage_database_connection_.get_session();

  const std::vector<int64_t> file_ids = get_file_ids(dataset_id, session, start_timestamp, end_timestamp);

  if (disable_multithreading_) {
    for (const int64_t file_id : file_ids) {
      send_samples_synchronous_retrieval<T>(writer, file_id, session);
    }
  } else {
    for (const int64_t file_id : file_ids) {
      send_samples_asynchronous_retrieval<T>(writer, file_id, session);
    }
  }
}

template <typename T>
void StorageServiceImpl::send_samples_synchronous_retrieval(::grpc::ServerWriter<T>* writer, int64_t file_id,
                                                            soci::session& session) {
  const int64_t number_of_samples = get_number_of_samples_in_file(file_id, session);
  if (number_of_samples > 0) {
    soci::rowset<soci::row> rs =  // NOLINT misc-const-correctness
        (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id", soci::use(file_id));
    T response;
    for (auto& row : rs) {
      response.add_keys(row.get<long long>(0));  // NOLINT google-runtime-int
      response.add_labels(row.get<long long>(1));  // NOLINT google-runtime-int
      if (response.keys_size() == sample_batch_size_) {
        writer->Write(response);
        response.Clear();
      }
    }

    if (response.keys_size() > 0) {
      writer->Write(response);
    }
  }
}

template <typename T>
void StorageServiceImpl::send_samples_asynchronous_retrieval(::grpc::ServerWriter<T>* writer, int64_t file_id,
                                                             soci::session& session) {
  const int64_t number_of_samples = get_number_of_samples_in_file(file_id, session);
  if (number_of_samples <= sample_batch_size_) {
    // If the number of samples is less than the sample batch size, retrieve all of the samples in one go.
    soci::rowset<soci::row> rs =  // NOLINT misc-const-correctness
        (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id", soci::use(file_id));
    T response;
    for (auto& row : rs) {
      response.add_keys(row.get<long long>(0));    // NOLINT google-runtime-int
      response.add_labels(row.get<long long>(1));  // NOLINT google-runtime-int
    }
    writer->Write(response);
  } else {
    // If the number of samples is greater than the sample batch size, retrieve the samples in batches of size
    // sample_batch_size_. The batches are retrieved asynchronously and the futures are stored in a queue. When the
    // queue is full, the first future is waited for and the response is sent to the client. This is repeated until all
    // of the futures have been waited for.
    std::queue<std::future<SampleData>> sample_ids_futures_queue;

    for (int64_t i = 0; i < number_of_samples; i += sample_batch_size_) {
      if (static_cast<int64_t>(sample_ids_futures_queue.size()) == retrieval_threads_) {
        // The queue is full, wait for the first future to finish and send the response.
        T response;

        SampleData sample_data = sample_ids_futures_queue.front().get();
        sample_ids_futures_queue.pop();

        for (size_t i = 0; i < sample_data.ids.size(); i++) {
          response.add_keys(sample_data.ids[i]);
          response.add_labels(sample_data.labels[i]);
        }

        writer->Write(response);
      }

      // Start a new future to retrieve the next batch of samples.
      std::future<SampleData> sample_ids_future =
          std::async(std::launch::async, get_sample_subset, file_id, i, i + sample_batch_size_ - 1,  // NOLINT
                     std::ref(storage_database_connection_));
      sample_ids_futures_queue.push(std::move(sample_ids_future));
    }

    // Wait for all of the futures to finish executing before returning.
    while (!sample_ids_futures_queue.empty()) {
      T response;

      SampleData sample_data = sample_ids_futures_queue.front().get();
      sample_ids_futures_queue.pop();

      for (size_t i = 0; i < sample_data.ids.size(); i++) {
        response.add_keys(sample_data.ids[i]);
        response.add_labels(sample_data.labels[i]);
      }

      writer->Write(response);
    }
  }
}

SampleData StorageServiceImpl::get_sample_subset(
    int64_t file_id, int64_t start_index, int64_t end_index,
    const storage::database::StorageDatabaseConnection& storage_database_connection) {
  soci::session session = storage_database_connection.get_session();
  const int64_t number_of_samples = end_index - start_index + 1;
  std::vector<int64_t> sample_ids(number_of_samples + 1);
  std::vector<int64_t> sample_labels(number_of_samples + 1);
  session << "SELECT sample_id, label FROM samples WHERE file_id = :file_id AND sample_index >= :start_index AND "
             "sample_index "
             "<= :end_index",
      soci::into(sample_ids), soci::into(sample_labels), soci::use(file_id), soci::use(start_index),
      soci::use(end_index);
  return {sample_ids, {}, sample_labels};
}

int64_t StorageServiceImpl::get_number_of_samples_in_file(int64_t file_id, soci::session& session) {
  int64_t number_of_samples;
  session << "SELECT number_of_samples FROM files WHERE file_id = :file_id", soci::into(number_of_samples),
      soci::use(file_id);
  return number_of_samples;
}

std::tuple<int64_t, int64_t> StorageServiceImpl::get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                                          int64_t total_num_elements) {
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
  int64_t dataset_id = -1;  // NOLINT misc-const-correctness
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

  return dataset_id;
}

std::vector<int64_t> StorageServiceImpl::get_file_ids(int64_t dataset_id, soci::session& session,
                                                      int64_t start_timestamp, int64_t end_timestamp) {
  int64_t number_of_files = -1;  // NOLINT misc-const-correctness
  std::vector<int64_t> file_ids;

  if (start_timestamp >= 0 && end_timestamp == -1) {
    session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp",
        soci::into(number_of_files), soci::use(dataset_id), soci::use(start_timestamp);
    if (number_of_files == 0) {
      return file_ids;
    }
    file_ids = std::vector<int64_t>(number_of_files + 1);
    session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp",
        soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp);
  } else if (start_timestamp == -1 && end_timestamp >= 0) {
    session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at <= :end_timestamp",
        soci::into(number_of_files), soci::use(dataset_id), soci::use(end_timestamp);
    if (number_of_files == 0) {
      return file_ids;
    }
    file_ids = std::vector<int64_t>(number_of_files + 1);

    session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at <= :end_timestamp",
        soci::into(file_ids), soci::use(dataset_id), soci::use(end_timestamp);
  } else if (start_timestamp >= 0 && end_timestamp >= 0) {
    session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp AND "
               "updated_at <= :end_timestamp",
        soci::into(number_of_files), soci::use(dataset_id), soci::use(start_timestamp), soci::use(end_timestamp);
    if (number_of_files == 0) {
      return file_ids;
    }
    file_ids = std::vector<int64_t>(number_of_files + 1);

    session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp AND "
               "updated_at <= :end_timestamp",
        soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp), soci::use(end_timestamp);
  } else {
    session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
        soci::use(dataset_id);
    if (number_of_files == 0) {
      return file_ids;
    }
    file_ids = std::vector<int64_t>(number_of_files + 1);

    session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id", soci::into(file_ids), soci::use(dataset_id);
  }

  return file_ids;
}
