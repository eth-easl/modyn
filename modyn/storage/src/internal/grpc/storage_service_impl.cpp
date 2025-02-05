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
    ServerContext* context, const modyn::storage::GetRequest* request,
    ServerWriter<modyn::storage::GetResponse>* writer) {
  return Get_Impl<ServerWriter<modyn::storage::GetResponse>>(context, request, writer);
}

Status StorageServiceImpl::GetNewDataSince(  // NOLINT readability-identifier-naming
    ServerContext* context, const modyn::storage::GetNewDataSinceRequest* request,
    ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {
  return GetNewDataSince_Impl<ServerWriter<modyn::storage::GetNewDataSinceResponse>>(context, request, writer);
}

Status StorageServiceImpl::GetDataInInterval(  // NOLINT readability-identifier-naming
    ServerContext* context, const modyn::storage::GetDataInIntervalRequest* request,
    ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) {
  return GetDataInInterval_Impl<ServerWriter<modyn::storage::GetDataInIntervalResponse>>(context, request, writer);
}

Status StorageServiceImpl::GetDataPerWorker(  // NOLINT readability-identifier-naming
    ServerContext* context, const modyn::storage::GetDataPerWorkerRequest* request,
    ServerWriter<::modyn::storage::GetDataPerWorkerResponse>* writer) {
  return GetDataPerWorker_Impl<ServerWriter<modyn::storage::GetDataPerWorkerResponse>>(context, request, writer);
}

Status StorageServiceImpl::CheckAvailability(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DatasetAvailableResponse* response) {
  try {
    soci::session session = storage_database_connection_.get_session();

    // Check if the dataset exists
    const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
    session.close();
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
    const int64_t dataset_id = get_dataset_id(session, request->dataset_id());
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
    session.close();
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
    // index is int type due to gRPC typing
    for (int index = 0; index < request->keys_size(); ++index) {
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

        std::vector<uint64_t> sample_ids_to_delete_ids(samples_to_delete + 1);
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
    session.close();
    response->set_success(true);
    return {StatusCode::OK, "Data deleted."};
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in DeleteData: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in DeleteData: {}", e.what())};
  }
}

Status StorageServiceImpl::GetDatasetSize(  // NOLINT readability-identifier-naming
    ServerContext* /*context*/, const modyn::storage::GetDatasetSizeRequest* request,
    modyn::storage::GetDatasetSizeResponse* response) {
  soci::session session = storage_database_connection_.get_session();
  try {
    // Check if the dataset exists
    const int64_t dataset_id = get_dataset_id(session, request->dataset_id());

    if (dataset_id == -1) {
      SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
      session.close();
      return {StatusCode::OK, "Dataset does not exist."};
    }

    const int64_t start_timestamp = request->has_start_timestamp() ? request->start_timestamp() : -1;
    const int64_t end_timestamp = request->has_end_timestamp() ? request->end_timestamp() : -1;
    const int64_t total_keys =
        get_number_of_samples_in_dataset_with_range(dataset_id, session, start_timestamp, end_timestamp);

    session.close();

    response->set_num_keys(total_keys);
    response->set_success(true);
    return {StatusCode::OK, "Dataset size retrieved."};
  } catch (const std::exception& e) {
    session.close();
    SPDLOG_ERROR("Error in GetDatasetSize: {}", e.what());
    return {StatusCode::OK, fmt::format("Error in GetDatasetSize: {}", e.what())};
  }
}

// ------- Helper functions -------
std::vector<std::pair<std::vector<int64_t>::const_iterator, std::vector<int64_t>::const_iterator>>
StorageServiceImpl::get_keys_per_thread(const std::vector<int64_t>& keys, uint64_t threads) {
  ASSERT(threads > 0, "This function is only intended for multi-threaded retrieval.");

  std::vector<std::pair<std::vector<int64_t>::const_iterator, std::vector<int64_t>::const_iterator>> keys_per_thread(
      threads);
  try {
    if (keys.empty()) {
      return keys_per_thread;
    }

    auto number_of_keys = static_cast<uint64_t>(keys.size());

    if (number_of_keys < threads) {
      threads = number_of_keys;
    }

    const auto subset_size = number_of_keys / threads;
    for (uint64_t thread_id = 0; thread_id < threads; ++thread_id) {
      // These need to be signed because we add them to iterators.
      const auto start_index = static_cast<int64_t>(thread_id * subset_size);
      const auto end_index = static_cast<int64_t>((thread_id + 1) * subset_size);

      DEBUG_ASSERT(start_index < static_cast<int64_t>(keys.size()),
                   fmt::format("Start Index too big! idx = {}, size = {}, thread_id = {}+1/{}, subset_size = {}",
                               start_index, keys.size(), thread_id, threads, subset_size));
      DEBUG_ASSERT(end_index <= static_cast<int64_t>(keys.size()),
                   fmt::format("End Index too big! idx = {}, size = {}, thread_id = {}+1/{}, subset_size = {}",
                               start_index, keys.size(), thread_id, threads, subset_size));

      if (thread_id == threads - 1) {
        keys_per_thread[thread_id] = std::make_pair(keys.begin() + start_index, keys.end());
      } else {
        keys_per_thread[thread_id] = std::make_pair(keys.begin() + start_index, keys.begin() + end_index);
      }
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in get_keys_per_thread with keys.size() = {}, retrieval_theads = {}: {}", keys.size(), threads,
                 e.what());
    throw;
  }
  return keys_per_thread;
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
  // TODO(#362): We are almost executing the same query twice since we first count and then get the data

  const uint64_t number_of_files = get_file_count(session, dataset_id, start_timestamp, end_timestamp);

  if (number_of_files == 0) {
    return {};
  }

  return get_file_ids_given_number_of_files(session, dataset_id, start_timestamp, end_timestamp, number_of_files);
}

uint64_t StorageServiceImpl::get_file_count(soci::session& session, const int64_t dataset_id,
                                            const int64_t start_timestamp, const int64_t end_timestamp) {
  uint64_t number_of_files = -1;
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
                                                                            const uint64_t number_of_files) {
  std::vector<int64_t> file_ids(number_of_files + 1);

  try {
    if (start_timestamp >= 0 && end_timestamp == -1) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp ORDER BY "
                 "updated_at ASC",
          soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp);
    } else if (start_timestamp == -1 && end_timestamp >= 0) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at <= :end_timestamp ORDER BY "
                 "updated_at ASC",
          soci::into(file_ids), soci::use(dataset_id), soci::use(end_timestamp);
    } else if (start_timestamp >= 0 && end_timestamp >= 0) {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id AND updated_at >= :start_timestamp AND "
                 "updated_at <= :end_timestamp ORDER BY updated_at ASC",
          soci::into(file_ids), soci::use(dataset_id), soci::use(start_timestamp), soci::use(end_timestamp);
    } else {
      session << "SELECT file_id FROM files WHERE dataset_id = :dataset_id ORDER BY updated_at ASC",
          soci::into(file_ids), soci::use(dataset_id);
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
  int has_labels_int = 1;

  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(dataset_name);

  YAML::Node config = YAML::Load(file_wrapper_config);
  if (config["has_labels"]) {
    has_labels_int = config["has_labels"].as<bool>() ? 1 : 0;
  }

  // Convert has_labels_int to bool
  const bool has_labels = (has_labels_int != 0);
  return DatasetData{.dataset_id = dataset_id,
                     .base_path = base_path,
                     .filesystem_wrapper_type = static_cast<FilesystemWrapperType>(filesystem_wrapper_type),
                     .file_wrapper_type = static_cast<FileWrapperType>(file_wrapper_type),
                     .file_wrapper_config = file_wrapper_config,
                     .has_labels = has_labels};
}
