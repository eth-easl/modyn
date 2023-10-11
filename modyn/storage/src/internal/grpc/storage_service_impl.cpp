#include "internal/grpc/storage_service_impl.hpp"

#include "internal/database/storage_database_connection.hpp"
#include "internal/file_wrapper/file_wrapper_utils.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"

using namespace storage::grpc;

::grpc::Status StorageServiceImpl::Get(::grpc::ServerContext* /*context*/, const modyn::storage::GetRequest* request,
                                       ::grpc::ServerWriter<modyn::storage::GetResponse>* writer) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }
  std::string base_path;
  int64_t filesystem_wrapper_type;
  int64_t file_wrapper_type;
  std::string file_wrapper_config;

  session << "SELECT base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM datasets WHERE "
             "name = :name",
      soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->dataset_id());

  uint64_t keys_size = request->keys_size();
  std::vector<int64_t> request_keys(keys_size);
  for (uint64_t i = 0; i < keys_size; i++) {
    request_keys[i] = request->keys(i);
  }

  if (disable_multithreading_) {
    // Group the samples and indices by file
    std::map<int64_t, SampleData> file_id_to_sample_data;

    get_sample_data(session, dataset_id, request_keys, file_id_to_sample_data);

    auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
        base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));
    const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

    if (file_id_to_sample_data.size() == 0) {
      SPDLOG_ERROR("No samples found in dataset {}.", request->dataset_id());
      return {::grpc::StatusCode::NOT_FOUND, "No samples found."};
    }
    for (auto& [file_id, sample_data] : file_id_to_sample_data) {
      send_get_response(writer, file_id, sample_data, file_wrapper_config_node, filesystem_wrapper, file_wrapper_type);
    }
  } else {
    std::vector<std::thread> threads(retrieval_threads_);

    for (uint64_t i = 0; i < retrieval_threads_; i++) {
      threads[i] = std::thread([&, i, keys_size, request_keys]() {
        std::map<int64_t, SampleData> file_id_to_sample_data;
        // Get the sample data for the current thread
        uint64_t start_index = i * (keys_size / retrieval_threads_);
        uint64_t end_index = (i + 1) * (keys_size / retrieval_threads_);
        if (end_index > keys_size) {
          end_index = keys_size;
        }
        uint64_t samples_prepared = 0;
        auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
            base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));
        const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

        for (uint64_t j = start_index; j < end_index; j++) {
          if (samples_prepared == sample_batch_size_) {
            for (auto& [file_id, sample_data] : file_id_to_sample_data) {
              send_get_response(writer, file_id, sample_data, file_wrapper_config_node, filesystem_wrapper,
                                file_wrapper_type);
            }
            file_id_to_sample_data.clear();
            samples_prepared = 0;
          }
          get_sample_data(session, dataset_id, {request_keys[j]}, file_id_to_sample_data);
          samples_prepared++;
        }

        if (samples_prepared > 0) {
          for (auto& [file_id, sample_data] : file_id_to_sample_data) {
            send_get_response(writer, file_id, sample_data, file_wrapper_config_node, filesystem_wrapper,
                              file_wrapper_type);
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
  return ::grpc::Status::OK;
}

void StorageServiceImpl::get_sample_data(soci::session& session, int64_t dataset_id,
                                         const std::vector<int64_t>& sample_ids,
                                         std::map<int64_t, SampleData>& file_id_to_sample_data) {
  std::vector<int64_t> sample_ids_found(sample_ids.size());
  std::vector<int64_t> sample_file_ids(sample_ids.size());
  std::vector<int64_t> sample_indices(sample_ids.size());
  std::vector<int64_t> sample_labels(sample_ids.size());

  session << "SELECT sample_id, file_id, sample_index, label FROM samples WHERE dataset_id = :dataset_id AND sample_id "
             "IN :sample_ids",
      soci::into(sample_ids_found), soci::into(sample_file_ids), soci::into(sample_indices), soci::into(sample_labels),
      soci::use(dataset_id), soci::use(sample_ids);

  for (uint64_t i = 0; i < sample_ids_found.size(); i++) {
    file_id_to_sample_data[sample_file_ids[i]].ids.push_back(sample_ids_found[i]);
    file_id_to_sample_data[sample_file_ids[i]].indices.push_back(sample_indices[i]);
    file_id_to_sample_data[sample_file_ids[i]].labels.push_back(sample_labels[i]);
  }
}

void StorageServiceImpl::send_get_response(
    ::grpc::ServerWriter<modyn::storage::GetResponse>* writer, int64_t file_id, const SampleData sample_data,
    const YAML::Node& file_wrapper_config,
    const std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper>& filesystem_wrapper,
    int64_t file_wrapper_type) {
  soci::session session = storage_database_connection_.get_session();
  // Get the file path
  std::string file_path;
  session << "SELECT path FROM files WHERE file_id = :file_id", soci::into(file_path), soci::use(file_id);

  auto file_wrapper = storage::file_wrapper::get_file_wrapper(
      file_path, static_cast<storage::file_wrapper::FileWrapperType>(file_wrapper_type), file_wrapper_config,
      filesystem_wrapper);

  std::vector<std::vector<unsigned char>> samples = file_wrapper->get_samples_from_indices(sample_data.indices);

  // Send the data to the client
  modyn::storage::GetResponse response;
  for (uint64_t i = 0; i < samples.size(); i++) {
    response.add_keys(sample_data.ids[i]);
    std::vector<uint8_t> sample_bytes(samples[i].begin(), samples[i].end());
    response.add_samples(std::string(sample_bytes.begin(), sample_bytes.end()));
    response.add_labels(sample_data.labels[i]);
  }
  writer->Write(response);
}

::grpc::Status StorageServiceImpl::GetNewDataSince(
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetNewDataSinceRequest* request,
    ::grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  uint64_t number_of_files = get_number_of_files(dataset_id, session);

  // Get the file ids
  std::vector<int64_t> file_ids(number_of_files);
  std::vector<int64_t> timestamps(number_of_files);
  session << "SELECT file_id, timestamp FROM files WHERE dataset_id = :dataset_id AND timestamp > :timestamp",
      soci::into(file_ids), soci::into(timestamps), soci::use(dataset_id), soci::use(request->timestamp());

  if (disable_multithreading_) {
    for (int64_t file_id : file_ids) {
      send_get_new_data_since_response(writer, file_id);
    }
  } else {
    std::vector<std::thread> threads(retrieval_threads_);

    for (uint64_t i = 0; i < retrieval_threads_; i++) {
      threads[i] = std::thread([&, i, number_of_files, file_ids]() {
        uint64_t start_index = i * (number_of_files / retrieval_threads_);
        uint64_t end_index = (i + 1) * (number_of_files / retrieval_threads_);
        if (end_index > number_of_files) {
          end_index = number_of_files;
        }
        for (uint64_t j = start_index; j < end_index; j++) {
          send_get_new_data_since_response(writer, file_ids[j]);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
  return ::grpc::Status::OK;
}

void StorageServiceImpl::send_get_new_data_since_response(
    ::grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer, int64_t file_id) {
  soci::session session = storage_database_connection_.get_session();
  int64_t number_of_samples;
  session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id", soci::into(number_of_samples), soci::use(file_id);
  soci::rowset<soci::row> rs =
      (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id",
       soci::use(file_id));

  modyn::storage::GetNewDataSinceResponse response;
  for (auto it = rs.begin(); it != rs.end(); ++it) {
    response.add_keys(it->get<int64_t>(0));
    response.add_labels(it->get<int64_t>(1));
  }
  writer->Write(response);
}

::grpc::Status StorageServiceImpl::GetDataInInterval(
    ::grpc::ServerContext* /*context*/, const modyn::storage::GetDataInIntervalRequest* request,
    ::grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  uint64_t number_of_files = get_number_of_files(dataset_id, session);

  // Get the file ids
  std::vector<int64_t> file_ids(number_of_files);
  std::vector<int64_t> timestamps(number_of_files);
  session << "SELECT file_id, timestamp FROM files WHERE dataset_id = :dataset_id AND timestamp >= :start_timestamp "
             "AND timestamp <= :end_timestamp ",
      soci::into(file_ids), soci::into(timestamps), soci::use(dataset_id), soci::use(request->start_timestamp()),
      soci::use(request->end_timestamp());

  if (disable_multithreading_) {
    for (int64_t file_id : file_ids) {
      send_get_new_data_in_interval_response(writer, file_id);
    }
  } else {
    std::vector<std::thread> threads(retrieval_threads_);

    for (uint64_t i = 0; i < retrieval_threads_; i++) {
      threads[i] = std::thread([&, i, number_of_files, file_ids]() {
        uint64_t start_index = i * (number_of_files / retrieval_threads_);
        uint64_t end_index = (i + 1) * (number_of_files / retrieval_threads_);
        if (end_index > number_of_files) {
          end_index = number_of_files;
        }
        for (uint64_t j = start_index; j < end_index; j++) {
          send_get_new_data_in_interval_response(writer, file_ids[j]);
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
  return ::grpc::Status::OK;
}

void StorageServiceImpl::send_get_new_data_in_interval_response(
    ::grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer, int64_t file_id) {
  soci::session session = storage_database_connection_.get_session();
  int64_t number_of_samples;
  session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id", soci::into(number_of_samples), soci::use(file_id);
  soci::rowset<soci::row> rs =
      (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id", soci::use(file_id));

  modyn::storage::GetDataInIntervalResponse response;
  for (auto it = rs.begin(); it != rs.end(); ++it) {
    response.add_keys(it->get<int64_t>(0));
    response.add_labels(it->get<int64_t>(1));
  }
  writer->Write(response);
}

::grpc::Status StorageServiceImpl::CheckAvailability(::grpc::ServerContext* /*context*/,
                                                     const modyn::storage::DatasetAvailableRequest* request,
                                                     modyn::storage::DatasetAvailableResponse* response) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  SPDLOG_INFO("Dataset {} exists: {}", request->dataset_id(), dataset_id != -1);

  ::grpc::Status status;
  if (dataset_id == -1) {
    response->set_available(false);
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    status = ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
  } else {
    response->set_available(true);
    status = ::grpc::Status::OK;
  }
  return status;
}

::grpc::Status StorageServiceImpl::RegisterNewDataset(::grpc::ServerContext* /*context*/,
                                                      const modyn::storage::RegisterNewDatasetRequest* request,
                                                      modyn::storage::RegisterNewDatasetResponse* response) {
  bool success = storage_database_connection_.add_dataset(
      request->dataset_id(), request->base_path(),
      storage::filesystem_wrapper::FilesystemWrapper::get_filesystem_wrapper_type(request->filesystem_wrapper_type()),
      storage::file_wrapper::FileWrapper::get_file_wrapper_type(request->file_wrapper_type()), request->description(),
      request->version(), request->file_wrapper_config(), request->ignore_last_timestamp(),
      static_cast<int>(request->file_watcher_interval()));
  response->set_success(success);
  ::grpc::Status status;
  if (success) {
    status = ::grpc::Status::OK;
  } else {
    status = ::grpc::Status(::grpc::StatusCode::INTERNAL, "Could not register dataset.");
  }
  return status;
}

::grpc::Status StorageServiceImpl::GetCurrentTimestamp(::grpc::ServerContext* /*context*/,
                                                       const modyn::storage::GetCurrentTimestampRequest* /*request*/,
                                                       modyn::storage::GetCurrentTimestampResponse* response) {
  response->set_timestamp(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
  return ::grpc::Status::OK;
}

::grpc::Status StorageServiceImpl::DeleteDataset(::grpc::ServerContext* /*context*/,
                                                 const modyn::storage::DatasetAvailableRequest* request,
                                                 modyn::storage::DeleteDatasetResponse* response) {
  std::string base_path;
  int64_t filesystem_wrapper_type;

  soci::session session = storage_database_connection_.get_session();
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);
  session << "SELECT base_path, filesystem_wrapper_type FROM datasets WHERE name = :name", soci::into(base_path),
      soci::into(filesystem_wrapper_type), soci::use(request->dataset_id());

  auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
      base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));

  int64_t number_of_files = get_number_of_files(dataset_id, session);

  if (number_of_files >= 0) {
    std::vector<std::string> file_paths(number_of_files);
    session << "SELECT path FROM files WHERE dataset_id = :dataset_id", soci::into(file_paths), soci::use(dataset_id);

    for (const auto& file_path : file_paths) {
      filesystem_wrapper->remove(file_path);
    }
  }

  bool success = storage_database_connection_.delete_dataset(request->dataset_id());
  response->set_success(success);
  ::grpc::Status status;
  if (success) {
    status = ::grpc::Status::OK;
  } else {
    status = ::grpc::Status(::grpc::StatusCode::INTERNAL, "Could not delete dataset.");
  }
  return status;
}

::grpc::Status StorageServiceImpl::DeleteData(::grpc::ServerContext* /*context*/,
                                              const modyn::storage::DeleteDataRequest* request,
                                              modyn::storage::DeleteDataResponse* response) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = -1;
  std::string base_path;
  int64_t filesystem_wrapper_type;
  int64_t file_wrapper_type;
  std::string file_wrapper_config;
  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->dataset_id());

  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  if (request->keys_size() == 0) {
    SPDLOG_ERROR("No keys provided.");
    return {::grpc::StatusCode::INVALID_ARGUMENT, "No keys provided."};
  }

  std::vector<int64_t> sample_ids;
  uint64_t keys_size = request->keys_size();
  for (uint64_t i = 0; i < keys_size; i++) {
    sample_ids.push_back(request->keys(i));
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
    return {::grpc::StatusCode::NOT_FOUND, "No samples found."};
  }

  // Get the file ids
  std::vector<int64_t> file_ids =
      std::vector<int64_t>(number_of_files + 1);  // There is some undefined behaviour if number_of_files is 1
  sql = fmt::format("SELECT DISTINCT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN {}",
                    sample_placeholders);
  session << sql, soci::into(file_ids), soci::use(dataset_id);

  if (file_ids.size() == 0) {
    SPDLOG_ERROR("No files found in dataset {}.", dataset_id);
    return {::grpc::StatusCode::NOT_FOUND, "No files found."};
  }

  auto filesystem_wrapper = storage::filesystem_wrapper::get_filesystem_wrapper(
      base_path, static_cast<storage::filesystem_wrapper::FilesystemWrapperType>(filesystem_wrapper_type));
  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);
  std::string file_placeholders = fmt::format("({})", fmt::join(file_ids, ","));
  std::string index_placeholders;

  try {
    std::vector<std::string> file_paths(number_of_files + 1);
    sql = fmt::format("SELECT path FROM files WHERE file_id IN {}", file_placeholders);
    session << sql, soci::into(file_paths);
    if (file_paths.size() != file_ids.size()) {
      SPDLOG_ERROR("Error deleting data: Could not find all files.");
      return {::grpc::StatusCode::INTERNAL, "Error deleting data."};
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

      std::vector<int64_t> sample_ids_to_delete_indices(samples_to_delete + 1);
      sql = fmt::format("SELECT sample_id FROM samples WHERE file_id = :file_id AND sample_id IN {}",
                        sample_placeholders);
      session << sql, soci::into(sample_ids_to_delete_indices), soci::use(file_id);

      file_wrapper->delete_samples(sample_ids_to_delete_indices);

      index_placeholders = fmt::format("({})", fmt::join(sample_ids_to_delete_indices, ","));
      sql = fmt::format("DELETE FROM samples WHERE file_id = :file_id AND sample_id IN {}", index_placeholders);
      session << sql, soci::use(file_id);

      int64_t number_of_samples_in_file;
      session << "SELECT number_of_samples FROM files WHERE file_id = :file_id", soci::into(number_of_samples_in_file),
          soci::use(file_id);

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
    return {::grpc::StatusCode::INTERNAL, "Error deleting data."};
  }
  response->set_success(true);
  return ::grpc::Status::OK;
}

::grpc::Status StorageServiceImpl::GetDataPerWorker(
    ::grpc::ServerContext* context, const modyn::storage::GetDataPerWorkerRequest* request,
    ::grpc::ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  int64_t total_keys = 0;
  soci::statement count_stmt = (session.prepare << "SELECT COUNT(*) FROM Sample WHERE dataset_id = :dataset_id",
                                soci::into(total_keys), soci::use(dataset_id));
  count_stmt.execute();

  int64_t start_index, limit;
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
      response.Clear();
    }
  }

  if (response.keys_size() > 0) {
    writer->Write(response);
  }

  return ::grpc::Status::OK;
}

std::tuple<int64_t, int64_t> StorageServiceImpl::get_partition_for_worker(int64_t worker_id, int64_t total_workers,
                                                                          int64_t total_num_elements) {
  if (worker_id < 0 || worker_id >= total_workers) {
    FAIL("Worker id must be between 0 and total_workers - 1.");
  }

  int64_t subset_size = total_num_elements / total_workers;
  int64_t worker_subset_size = subset_size;

  int64_t threshold = total_num_elements % total_workers;
  if (threshold > 0) {
    if (worker_id < threshold) {
      worker_subset_size += 1;
      int64_t start_index = worker_id * (subset_size + 1);
      return {start_index, worker_subset_size};
    } else {
      int64_t start_index = threshold * (subset_size + 1) + (worker_id - threshold) * subset_size;
      return {start_index, worker_subset_size};
    }
  } else {
    int64_t start_index = worker_id * subset_size;
    return {start_index, worker_subset_size};
  }
}

::grpc::Status StorageServiceImpl::GetDatasetSize(::grpc::ServerContext* context,
                                                  const modyn::storage::GetDatasetSizeRequest* request,
                                                  modyn::storage::GetDatasetSizeResponse* response) {
  soci::session session = storage_database_connection_.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == -1) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {::grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  int64_t total_keys = 0;
  soci::statement count_stmt = (session.prepare << "SELECT COUNT(*) FROM Sample WHERE dataset_id = :dataset_id",
                                soci::into(total_keys), soci::use(dataset_id));

  count_stmt.execute();
  response->set_num_keys(total_keys);
  return ::grpc::Status::OK;
}

int64_t StorageServiceImpl::get_dataset_id(const std::string& dataset_name, soci::session& session) {
  int64_t dataset_id = -1;
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

  return dataset_id;
}

uint64_t StorageServiceImpl::get_number_of_files(int64_t dataset_id, soci::session& session) {
  uint64_t number_of_files = -1;
  session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
      soci::use(dataset_id);

  return number_of_files;
}