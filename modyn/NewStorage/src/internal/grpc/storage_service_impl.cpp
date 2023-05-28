#include "internal/grpc/storage_service_impl.hpp"

#include <spdlog/spdlog.h>

#include "internal/database/storage_database_connection.hpp"
#include "internal/utils/utils.hpp"

using namespace storage;

grpc::Status StorageServiceImpl::Get(
    grpc::ServerContext* context,
    const modyn::storage::GetRequest* request,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetResponse>* writer) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id;
  std::string base_path;
  std::string filesystem_wrapper_type;
  std::string file_wrapper_type;
  std::string file_wrapper_config;
  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->name());
  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
  }

  vector<int64_t> sample_ids = vector<int64_t>(request->keys_size());
  for (int i = 0; i < request->keys_size(); i++) {
    sample_ids[i] = request->keys(i);
  }

  vector<int64_t> sample_ids_found = vector<int64_t>(request->keys_size());
  vector<int64_t> sample_file_ids = vector<int64_t>(request->keys_size());
  vector<int64_t> sample_indices = vector<int64_t>(request->keys_size());
  vector<int64_t> sample_labels = vector<int64_t>(request->keys_size());
  session << "SELECT sample_id, file_id, sample_index, label FROM samples WHERE dataset_id = :dataset_id AND sample_id "
             "IN :sample_ids",
      soci::into(sample_ids_found), soci::into(sample_file_ids), soci::into(sample_indices), soci::into(sample_labels),
      soci::use(dataset_id), soci::use(sample_ids);

  for (int i = 0; i < sample_ids_found.size(); i++) {
    if (sample_ids_found[i] == 0) {
      SPDLOG_ERROR("Sample {} does not exist in dataset {}.", sample_ids[i], request->dataset_id());
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Sample does not exist.");
    }
  }

  // Group the samples and indices by file
  std::map < int64_t, std::tuple < std::vector<int64_t>, std::vector<int64_t>,
      std::vector < int64_t >>>> file_id_to_sample_ids;
  for (int i = 0; i < sample_ids_found.size(); i++) {
    file_id_to_sample_ids[sample_file_ids[i]].first.push_back(sample_ids_found[i]);
    file_id_to_sample_ids[sample_file_ids[i]].second.push_back(sample_indices[i]);
    file_id_to_sample_ids[sample_file_ids[i]].third.push_back(sample_labels[i]);
  }

  auto filesystem_wrapper = Utils::get_filesystem_wrapper(base_path, filesystem_wrapper_type);
  const YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  // Get the data from the files
  for (auto& [file_id, sample_ids_and_indices] : file_id_to_sample_ids) {
    // Get the file path
    std::string file_path;
    session << "SELECT path FROM files WHERE file_id = :file_id", soci::into(file_path), soci::use(file_id);

    // Get the data from the file
    auto file_wrapper = Utils::get_file_wrapper(file_path, get_file_wrapper_type_map()[file_wrapper_type],
                                                file_wrapper_config_node, &filesystem_wrapper);

    std::vector<std::vector<unsigned char>> samples =
        file_wrapper->get_get_samples_from_indices(std::get<1>(sample_ids_and_indices));

    // Send the data to the client
    modyn::storage::GetResponse response;
    for (int i = 0; i < samples.size(); i++) {
      response.add_keys(std::get<0>(sample_ids_and_indices)[i]);
      response.add_samples(samples[i]);
      response.add_labels(std::get<2>(sample_ids_and_indices)[i]);

      if (i % sample_batch_size_ == 0) {
        writer->Write(response);
        response.Clear();
      }
    }
    if (response.keys_size() > 0) {
      writer->Write(response);
    }
  }
}
grpc::Status StorageServiceImpl::GetNewDataSince(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* context,
    const modyn::storage::GetNewDataSinceRequest* request,  // NOLINT (misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(session, request->dataset_id());

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
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
    extract_and_write_samples_from_file_id(file_id, writer);
  }
}

grpc::Status StorageServiceImpl::GetDataInInterval(  // NOLINT (readability-identifier-naming)
    grpc::ServerContext* context,
    const modyn::storage::GetDataInIntervalRequest* request,  // NOLINT (misc-unused-parameters)
    grpc::ServerWriter<modyn::storage::GetDataInIntervalResponse>* writer) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (get_dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
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
    extract_and_write_samples_from_file_id(file_id, writer);
  }
}

grpc::Status StorageServiceImpl::CheckAvailability(
    grpc::ServerContext* context,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    const modyn::storage::DatasetAvailableRequest* request,
    modyn::storage::DatasetAvailableResponse* response) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id = get_dataset_id(request->dataset_id(), session);

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
  } else {
    response->set_available(true);
    return grpc::Status::OK;
  }
}

grpc::Status StorageServiceImpl::RegisterNewDataset(
    grpc::ServerContext* context,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    const modyn::storage::RegisterNewDatasetRequest* request,
    modyn::storage::RegisterNewDatasetResponse* response) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);

  bool success = storage_database_connection.add_dataset(
      request->dataset_id(), request->base_path(),
      get_filesystem_wrapper_type_map()[request->filesystem_wrapper_type()],
      get_file_wrapper_type_map()[request->file_wrapper_type()], request->description(), request->version(),
      request->file_wrapper_config(), request->ignore_last_timestamp(), request->file_watcher_interval());
  response->set_success(success);
  if (success) {
    return grpc::Status::OK;
  } else {
    return grpc::Status(grpc::StatusCode::ERROR, "Could not register dataset.");
  }
}

grpc::Status StorageServiceImpl::GetCurrentTimestamp(
    grpc::ServerContext* context,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    const modyn::storage::GetCurrentTimestampRequest* request,
    modyn::storage::GetCurrentTimestampResponse* response) override {
  response->set_timestamp(
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count());
  return grpc::Status::OK;
}

grpc::Status StorageServiceImpl::DeleteDataset(
    grpc::ServerContext* context,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    const modyn::storage::DatasetAvailableRequest* request, modyn::storage::DeleteDatasetResponse* response) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  bool success = storage_database_connection.delete_dataset(request->dataset_id());
  response->set_success(success);
  if (success) {
    return grpc::Status::OK;
  } else {
    return grpc::Status(grpc::StatusCode::ERROR, "Could not delete dataset.");
  }
}
grpc::Status StorageServiceImpl::DeleteData(
    grpc::ServerContext* context,  // NOLINT (readability-identifier-naming, misc-unused-parameters)
    const modyn::storage::DeleteDataRequest* request, modyn::storage::DeleteDataResponse* response) override {
  const StorageDatabaseConnection storage_database_connection = StorageDatabaseConnection(config_);
  soci::session session = storage_database_connection.get_session();

  // Check if the dataset exists
  int64_t dataset_id;
  std::string base_path;
  std::string filesystem_wrapper_type;
  std::string file_wrapper_type;
  std::string file_wrapper_config;
  session << "SELECT dataset_id, base_path, filesystem_wrapper_type, file_wrapper_type, file_wrapper_config FROM "
             "datasets WHERE name = :name",
      soci::into(dataset_id), soci::into(base_path), soci::into(filesystem_wrapper_type), soci::into(file_wrapper_type),
      soci::into(file_wrapper_config), soci::use(request->name());

  if (dataset_id == 0) {
    SPDLOG_ERROR("Dataset {} does not exist.", request->dataset_id());
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Dataset does not exist.");
  }

  vector<int64_t> sample_ids = vector<int64_t>(request->keys_size());
  for (int i = 0; i < request->keys_size(); i++) {
    sample_ids[i] = request->keys(i);
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

  FilesystemWrapper filesystem_wrapper =
      get_filesystem_wrapper(base_path, get_filesystem_wrapper_type_map()[filesystem_wrapper_type]);
  YAML::Node file_wrapper_config_node = YAML::Load(file_wrapper_config);

  for (int64_t file_id : file_ids) {
    std::string path;
    session << "SELECT path FROM files WHERE file_id = :file_id", soci::into(path), soci::use(file_id);
    FileWrapper file_wrapper = get_file_wrapper(path, get_file_wrapper_type_map()[file_wrapper_type],
                                                file_wrapper_config_node, &filesystem_wrapper);

    int64_t samples_to_delete;
    session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id AND sample_id IN :sample_ids",
        soci::into(samples_to_delete), soci::use(file_id), soci::use(sample_ids);

    std::vector<int64_t> sample_ids_to_delete_indices = std::vector<int64_t>(samples_to_delete);
    session << "SELECT sample_id FROM samples WHERE file_id = :file_id AND sample_id IN :sample_ids",
        soci::into(sample_ids_to_delete_indices), soci::use(file_id), soci::use(sample_ids);

    file_wrapper.delete_samples(sample_ids_to_delete_indices);

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
}
void extract_and_write_samples_from_file_id(int64_t file_id,
                                            grpc::ServerWriter<modyn::storage::GetNewDataSinceResponse>* writer) {
  int64_t number_of_samples;
  session << "SELECT COUNT(*) FROM samples WHERE file_id = :file_id", soci::into(number_of_samples), soci::use(file_id);
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

int64_t get_dataset_id(const std::string& dataset_name, soci::session& session) {
  int64_t dataset_id;
  session << "SELECT dataset_id FROM datasets WHERE name = :name", soci::into(dataset_id), soci::use(dataset_name);

  return dataset_id;
}