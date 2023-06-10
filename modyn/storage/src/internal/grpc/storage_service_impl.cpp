#include "internal/grpc/storage_service_impl.hpp"

#include "internal/database/storage_database_connection.hpp"
#include "internal/utils/utils.hpp"

using namespace storage;

grpc::Status StorageServiceImpl::Get(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/, const modyn::storage::GetRequest* request,  // NOLINT (misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetResponse>* writer) {                    // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = 0;
  std::string base_path;
  std::string filesystem_wrapper_type;
  std::string file_wrapper_type;
  std::string file_wrapper_config;
  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->dataset_id());
  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  std::vector<int64_t> sample_ids = std::vector<int64_t>(request->keys_size());
  for (int i = 0; i < request->keys_size(); i++) {
    sample_ids[i] = request->keys(i);
  }

  // Group the samples and indices by file
  std::map<int64_t, SampleData> file_id_to_sample_data;

  std::vector<int64_t> sample_ids_found(sample_ids.size());
  std::vector<int64_t> sample_file_ids(sample_ids.size());
  std::vector<int64_t> sample_indices(sample_ids.size());
  std::vector<int64_t> sample_labels(sample_ids.size());

  session << "SELECT sample_id, file_id, sample_index, label FROM samples WHERE dataset_id = :dataset_id AND sample_id "
             "IN :sample_ids",
      soci::into(sample_ids_found), soci::into(sample_file_ids), soci::into(sample_indices), soci::into(sample_labels),
      soci::use(dataset_id), soci::use(sample_ids);

  for (std::size_t i = 0; i < sample_ids_found.size(); i++) {
    file_id_to_sample_data[sample_file_ids[i]].ids.push_back(sample_ids_found[i]);
    file_id_to_sample_data[sample_file_ids[i]].indices.push_back(sample_indices[i]);
    file_id_to_sample_data[sample_file_ids[i]].labels.push_back(sample_labels[i]);
  }

  auto filesystem_wrapper =
      Utils::get_filesystem_wrapper(base_path, FilesystemWrapper::get_filesystem_wrapper_type(filesystem_wrapper_type));
  const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  if (file_id_to_sample_data.size() == 0) {
    SPDLOG_ERROR("No samples found in dataset {}.", request->dataset_id());
    return {grpc::StatusCode::NOT_FOUND, "No samples found."};
  }

  std::string file_path;

  auto& [file_id, sample_data] = *file_id_to_sample_data.begin();

  session << "SELECT path FROM files WHERE file_id = :file_id", soci::into(file_path), soci::use(file_id);

  auto file_wrapper = Utils::get_file_wrapper(file_path, FileWrapper::get_file_wrapper_type(file_wrapper_type),
                                              file_wrapper_config_node, filesystem_wrapper);

  // Get the data from the files
  for (auto& [file_id, sample_data] : file_id_to_sample_data) {
    // Get the file path

    session << "SELECT path FROM files WHERE file_id = :file_id", soci::into(file_path), soci::use(file_id);

    // Get the data from the file
    file_wrapper->set_file_path(file_path);

    std::vector<std::vector<unsigned char>> samples = file_wrapper->get_samples_from_indices(sample_data.indices);

    // Send the data to the client
    modyn::storage::GetResponse response;
    for (std::size_t i = 0; i < samples.size(); i++) {
      response.add_keys(sample_data.ids[i]);
      for (auto sample : samples[i]) {
        response.add_samples(std::string(1, sample));
      }
      response.add_labels(sample_data.labels[i]);

      if (i % sample_batch_size_ == 0) {
        writer->Write(response);
        response.Clear();
      }
    }
    if (response.keys_size() > 0) {
      writer->Write(response);
    }
  }
  return grpc::Status::OK;
}

grpc::Status StorageServiceImpl::GetNewDataSince(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::GetNewDataSinceRequest* request,                  // NOLINT (misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {  // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  int64_t number_of_files;
  session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
      soci::use(dataset_id);

  // Get the file ids
  std::vector<int64_t> file_ids = std::vector<int64_t>(number_of_files);
  std::vector<int64_t> timestamps = std::vector<int64_t>(number_of_files);
  session << "SELECT file_id, timestamp FROM files WHERE dataset_id = :dataset_id AND timestamp > :timestamp",
      soci::into(file_ids), soci::into(timestamps), soci::use(dataset_id), soci::use(request->timestamp());

  for (int64_t file_id : file_ids) {
    int64_t number_of_samples;
    session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id", soci::into(number_of_samples),
        soci::use(file_id);
    std::vector<int64_t> sample_ids = std::vector<int64_t>(number_of_samples);
    std::vector<int64_t> sample_labels = std::vector<int64_t>(number_of_samples);
    soci::rowset<soci::row> rs = (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id",
                                  soci::into(sample_ids), soci::into(sample_labels), soci::use(file_id));

    modyn::storage::GetNewDataSinceResponse response;
    int64_t count = 0;
    for (auto it = rs.begin(); it != rs.end(); ++it) {
      response.add_keys(sample_ids[count]);
      response.add_labels(sample_labels[count]);
      count++;
      if (count % sample_batch_size_ == 0) {
        writer->Write(response);
        response.Clear();
      }
    }
    if (response.keys_size() > 0) {
      writer->Write(response);
    }
  }
  return grpc::Status::OK;
}

grpc::Status StorageServiceImpl::GetDataInInterval(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::GetDataInIntervalRequest* request,                  // NOLINT (misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) {  // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  int64_t number_of_files;
  session << "SELECT COUNT(*) FROM files WHERE dataset_id = :dataset_id", soci::into(number_of_files),
      soci::use(dataset_id);

  // Get the file ids
  std::vector<int64_t> file_ids = std::vector<int64_t>(number_of_files);
  std::vector<int64_t> timestamps = std::vector<int64_t>(number_of_files);
  session << "SELECT file_id, timestamp FROM files WHERE dataset_id = :dataset_id AND timestamp >= :start_timestamp "
             "AND timestamp <= :end_timestamp ",
      soci::into(file_ids), soci::into(timestamps), soci::use(dataset_id), soci::use(request->start_timestamp()),
      soci::use(request->end_timestamp());

  for (int64_t file_id : file_ids) {
    int64_t number_of_samples;
    session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id", soci::into(number_of_samples),
        soci::use(file_id);
    std::vector<int64_t> sample_ids = std::vector<int64_t>(number_of_samples);
    std::vector<int64_t> sample_labels = std::vector<int64_t>(number_of_samples);
    soci::rowset<soci::row> rs = (session.prepare << "SELECT sample_id, label FROM samples WHERE file_id = :file_id",
                                  soci::into(sample_ids), soci::into(sample_labels), soci::use(file_id));

    modyn::storage::GetDataInIntervalResponse response;
    int64_t count = 0;
    for (auto it = rs.begin(); it != rs.end(); ++it) {
      response.add_keys(sample_ids[count]);
      response.add_labels(sample_labels[count]);
      count++;
      if (count % sample_batch_size_ == 0) {
        writer->Write(response);
        response.Clear();
      }
    }
    if (response.keys_size() > 0) {
      writer->Write(response);
    }
  }
  return grpc::Status::OK;
}

grpc::Status StorageServiceImpl::CheckAvailability(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::DatasetAvailableRequest* request,  // NOLINT (misc-unused-parameters)
    modyn::storage::DatasetAvailableResponse* response) {    // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  grpc::Status status;
  if (dataset_id == 0) {
    response->set_available(false);
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    status = grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
  } else {
    response->set_available(true);
    status = grpc::Status::OK;
  }
  return status;
}

grpc::Status StorageServiceImpl::RegisterNewDataset(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::RegisterNewDatasetRequest* request,  // NOLINT (misc-unused-parameters)
    modyn::storage::RegisterNewDatasetResponse* response) {    // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);

  bool success = storage_database_connection.add_dataset(
      request->dataset_id(), request->base_path(),
      FilesystemWrapper::get_filesystem_wrapper_type(request->filesystem_wrapper_type()),
      FileWrapper::get_file_wrapper_type(request->file_wrapper_type()), request->description(), request->version(),
      request->file_wrapper_config(), request->ignore_last_timestamp(),
      static_cast<int>(request->file_watcher_interval()));
  response->set_success(success);
  grpc::Status status;
  if (success) {
    status = grpc::Status::OK;
  } else {
    status = grpc::Status(grpc::StatusCode::INTERNAL, "Could not register dataset.");
  }
  return status;
}

grpc::Status StorageServiceImpl::GetCurrentTimestamp(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/, const modyn::storage::GetCurrentTimestampRequest* /*request*/,
    modyn::storage::GetCurrentTimestampResponse* response) {  // NOLINT (misc-unused-parameters)
  response->set_timestamp(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
  return grpc::Status::OK;
}

grpc::Status StorageServiceImpl::DeleteDataset(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::DatasetAvailableRequest* request,  // NOLINT (misc-unused-parameters)
    modyn::storage::DeleteDatasetResponse* response) {       // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  bool success = storage_database_connection.delete_dataset(request->dataset_id());
  response->set_success(success);
  grpc::Status status;
  if (success) {
    status = grpc::Status::OK;
  } else {
    status = grpc::Status(grpc::StatusCode::INTERNAL, "Could not delete dataset.");
  }
  return status;
}

grpc::Status StorageServiceImpl::DeleteData(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* /*context*/,
    const modyn::storage::DeleteDataRequest* request,  // NOLINT (misc-unused-parameters)
    modyn::storage::DeleteDataResponse* response) {    // NOLINT (misc-unused-parameters)
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = 0;
  std::string base_path;
  std::string filesystem_wrapper_type;
  std::string file_wrapper_type;
  std::string file_wrapper_config;
  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->dataset_id());

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return {grpc::StatusCode::NOT_FOUND, "Dataset does not exist."};
  }

  std::vector<int64_t> sample_ids;
  for (int i = 0; i < request->keys_size(); i++) {
    sample_ids.push_back(request->keys(i));
  }

  int64_t number_of_files;
  session << "SELECT COUNT(file_id) FROM samples WHERE dataset_id = :dataset_id AND sample_id IN :sample_ids GROUP "
             "BY file_id",
      soci::into(number_of_files), soci::use(dataset_id), soci::use(sample_ids);

  // Get the file ids
  std::vector<int64_t> file_ids = std::vector<int64_t>(number_of_files);
  session << "SELECT file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN :sample_ids GROUP BY "
             "file_id",
      soci::into(file_ids), soci::use(dataset_id), soci::use(sample_ids);

  auto filesystem_wrapper =
      Utils::get_filesystem_wrapper(base_path, FilesystemWrapper::get_filesystem_wrapper_type(filesystem_wrapper_type));
  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  try {
    std::vector<std::string> file_paths;
    session << "SELECT path FROM files WHERE file_id IN :file_ids", soci::into(file_paths), soci::use(file_ids);

    if (file_paths.size() != file_ids.size()) {
      SPDLOG_ERROR("Error deleting data: Could not find all files.");
      return {grpc::StatusCode::INTERNAL, "Error deleting data."};
    }

    auto file_wrapper =
        Utils::get_file_wrapper(file_paths.front(), FileWrapper::get_file_wrapper_type(file_wrapper_type),
                                file_wrapper_config_node, filesystem_wrapper);
    for (size_t i = 0; i < file_paths.size(); ++i) {
      const auto& file_id = file_ids[i];
      const auto& path = file_paths[i];
      file_wrapper->set_file_path(path);

      int64_t samples_to_delete;
      session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id AND sample_id IN :sample_ids",
          soci::into(samples_to_delete), soci::use(file_id), soci::use(sample_ids);

      std::vector<int64_t> sample_ids_to_delete_indices = std::vector<int64_t>(samples_to_delete);
      session << "SELECT sample_id FROM samples WHERE file_id = :file_id AND sample_id IN :sample_ids",
          soci::into(sample_ids_to_delete_indices), soci::use(file_id), soci::use(sample_ids);

      file_wrapper->delete_samples(sample_ids_to_delete_indices);

      session << "DELETE FROM samples WHERE file_id = :file_id AND index IN :index", soci::use(file_id),
          soci::use(sample_ids_to_delete_indices);

      int64_t number_of_samples_in_file;
      session << "SELECT number_of_samples FROM files WHERE file_id = :file_id", soci::into(number_of_samples_in_file),
          soci::use(file_id);

      if (number_of_samples_in_file - samples_to_delete == 0) {
        session << "DELETE FROM files WHERE file_id = :file_id", soci::use(file_id);
      } else {
        session << "UPDATE files SET number_of_samples = :number_of_samples WHERE file_id = :file_id",
            soci::use(number_of_samples_in_file - samples_to_delete), soci::use(file_id);
      }
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error deleting data: {}", e.what());
    return {grpc::StatusCode::INTERNAL, "Error deleting data."};
  }
  response->set_success(true);
  return grpc::Status::OK;
}
