#include "internal/grpc/storage_service_impl.hpp"

#include <gtest/gtest.h>
#include <soci/soci.h>
#include <soci/sqlite3/soci-sqlite3.h>
#include <spdlog/spdlog.h>
#include <utime.h>

#include <cstdint>
#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "internal/database/storage_database_connection.hpp"
#include "internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;
using namespace grpc;

class StorageServiceImplTest : public ::testing::Test {
 protected:
  std::string tmp_dir_;
  int64_t early_sample_id_ = -1;
  int64_t late_sample_id_ = -1;

  StorageServiceImplTest() : tmp_dir_{std::filesystem::temp_directory_path().string() + "/storage_service_impl_test"} {}

  void SetUp() override {
    modyn::test::TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory(tmp_dir_);
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
    connection.create_tables();

    // Add a dataset to the database
    connection.add_dataset("test_dataset", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                           "test description", "0.0.0", StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                           /*ignore_last_timestamp=*/true);

    soci::session session =
        connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)
    std::string sql_expression = fmt::format(
        "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/test_file.txt', 100, "
        "1)",
        tmp_dir_);
    session << sql_expression;

    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 0, 0)";
    long long inserted_id = -1;  // NOLINT google-runtime-int (Linux otherwise complains about the following call)
    if (!session.get_last_insert_id("samples", inserted_id)) {
      FAIL("Failed to insert sample into database");
    }
    late_sample_id_ = static_cast<int64_t>(inserted_id);

    sql_expression = fmt::format(
        "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/test_file2.txt', "
        "1, 1)",
        tmp_dir_);
    session << sql_expression;

    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 2, 0, 1)";
    inserted_id = -1;  // NOLINT google-runtime-int (Linux otherwise complains about the following call)
    if (!session.get_last_insert_id("samples", inserted_id)) {
      FAIL("Failed to insert sample into database");
    }
    early_sample_id_ = static_cast<int64_t>(inserted_id);

    // Create dummy files
    const std::string test_file_path = tmp_dir_ + "/test_file.txt";
    std::ofstream test_file(test_file_path);
    ASSERT(test_file.is_open(), "Could not open test file");
    test_file << "test";
    test_file.close();
    ASSERT(!test_file.is_open(), "Could not close test file");

    const std::string label_file_path = tmp_dir_ + "/test_file.lbl";
    std::ofstream label_file(label_file_path);
    ASSERT(label_file.is_open(), "Could not open label file");
    label_file << "1";
    label_file.close();
    ASSERT(!label_file.is_open(), "Could not close label file");

    const std::string test_file_path2 = tmp_dir_ + "/test_file2.txt";
    std::ofstream test_file2(test_file_path2);
    ASSERT(test_file2.is_open(), "Could not open test file");
    test_file2 << "test";
    test_file2.close();
    ASSERT(!test_file2.is_open(), "Could not close test file");

    const std::string label_file_path2 = tmp_dir_ + "/test_file2.lbl";
    std::ofstream label_file2(label_file_path2);
    ASSERT(label_file2.is_open(), "Could not open label file");
    label_file2 << "2";
    label_file2.close();
    ASSERT(!label_file2.is_open(), "Could not close label file");
  }

  void TearDown() override {
    // Remove temporary directory
    std::filesystem::remove_all(tmp_dir_);
    std::filesystem::remove("config.yaml");
    if (std::filesystem::exists("test.db")) {
      std::filesystem::remove("test.db");
    }
  }
};

TEST_F(StorageServiceImplTest, TestCheckAvailability) {
  ServerContext context;

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DatasetAvailableResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  Status status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(response.available());

  request.set_dataset_id("non_existing_dataset");
  status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_FALSE(response.available());
}

TEST_F(StorageServiceImplTest, TestGetCurrentTimestamp) {
  ServerContext context;

  const modyn::storage::GetCurrentTimestampRequest request;

  modyn::storage::GetCurrentTimestampResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  const Status status = storage_service.GetCurrentTimestamp(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_GE(response.timestamp(), 0);
}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  const StorageDatabaseConnection connection(config);

  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DeleteDatasetResponse response;

  ServerContext context;

  int dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_TRUE(dataset_exists);

  const Status status = storage_service.DeleteDataset(&context, &request, &response);

  ASSERT_TRUE(status.ok());

  ASSERT_TRUE(response.success());

  dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_FALSE(dataset_exists);
}

TEST_F(StorageServiceImplTest, TestDeleteData) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  modyn::storage::DeleteDataRequest request;
  request.set_dataset_id("test_dataset");
  request.add_keys(1);

  // Add an additional sample for file 1 to the database
  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 1, 0)";

  modyn::storage::DeleteDataResponse response;

  ServerContext context;

  Status status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  int number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 2);

  ASSERT_FALSE(std::filesystem::exists(tmp_dir_ + "/test_file.txt"));

  ASSERT_TRUE(std::filesystem::exists(tmp_dir_ + "/test_file2.txt"));

  request.clear_keys();

  status = storage_service.DeleteData(&context, &request, &response);

  request.add_keys(1);

  status = storage_service.DeleteData(&context, &request, &response);

  request.clear_keys();
  request.add_keys(2);

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 1);
}

// NOLINTNEXTLINE (readability-function-cognitive-complexity)
TEST_F(StorageServiceImplTest, TestGetNewDataSince) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness
  grpc::ServerContext context;
  grpc::internal::Call call;
  modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse> writer(&call, &context);

  modyn::storage::GetNewDataSinceRequest request;
  request.set_dataset_id("test_dataset");
  request.set_timestamp(0);

  grpc::Status status =
      storage_service.GetNewDataSince_Impl<modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse>>(
          &context, &request, &writer);

  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetNewDataSinceResponse>& responses = writer.get_responses();
  ASSERT_EQ(responses.size(), 1);
  const modyn::storage::GetNewDataSinceResponse& response = responses[0];

  std::vector<int64_t> keys;
  keys.reserve(response.keys_size());
  for (const auto& key : response.keys()) {
    keys.push_back(key);
  }

  ASSERT_THAT(keys, ::testing::UnorderedElementsAre(early_sample_id_, late_sample_id_));

  // Now try only the second file

  modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse> writer2(&call, &context);
  request.set_timestamp(50);
  status =
      storage_service.GetNewDataSince_Impl<modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse>>(
          &context, &request, &writer2);
  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetNewDataSinceResponse>& responses2 = writer2.get_responses();
  ASSERT_EQ(responses2.size(), 1);
  const modyn::storage::GetNewDataSinceResponse& response2 = responses2[0];
  std::vector<int64_t> keys2;
  keys2.reserve(response2.keys_size());
  for (const auto& key : response2.keys()) {
    keys2.push_back(key);
  }

  ASSERT_THAT(keys2, ::testing::ElementsAre(late_sample_id_));

  // And now no files
  modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse> writer3(&call, &context);
  request.set_timestamp(101);
  status =
      storage_service.GetNewDataSince_Impl<modyn::storage::MockServerWriter<modyn::storage::GetNewDataSinceResponse>>(
          &context, &request, &writer3);
  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetNewDataSinceResponse>& responses3 = writer3.get_responses();
  ASSERT_EQ(responses3.size(), 0);
}

// NOLINT(readability-function-cognitive-complexity)
TEST_F(StorageServiceImplTest, TestGetDataInInterval) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness
  grpc::ServerContext context;
  grpc::internal::Call call;
  modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse> writer(&call, &context);

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)
  const std::string sql_expression = fmt::format(
      "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/non_existing.txt', 200, "
      "1)",
      tmp_dir_);
  session << sql_expression;

  long long inserted_file_id = -1;  // NOLINT google-runtime-int (soci needs ll)
  if (!session.get_last_insert_id("files", inserted_file_id)) {
    FAIL("Failed to insert file into database");
  }

  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, :file, 0, 0)",
      soci::use(inserted_file_id);
  long long inserted_sample_id_ll =  // NOLINT google-runtime-int (soci needs ll)
      -1;
  if (!session.get_last_insert_id("samples", inserted_sample_id_ll)) {
    FAIL("Failed to insert sample into database");
  }

  auto inserted_sample_id = static_cast<uint64_t>(inserted_sample_id_ll);

  modyn::storage::GetDataInIntervalRequest request;
  request.set_dataset_id("test_dataset");
  request.set_start_timestamp(0);
  request.set_end_timestamp(250);

  grpc::Status status =
      storage_service
          .GetDataInInterval_Impl<modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse>>(
              &context, &request, &writer);

  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetDataInIntervalResponse>& responses = writer.get_responses();
  ASSERT_EQ(responses.size(), 1);
  const modyn::storage::GetDataInIntervalResponse& response = responses[0];

  std::vector<int64_t> keys;
  keys.reserve(response.keys_size());
  for (const auto& key : response.keys()) {
    keys.push_back(key);
  }

  ASSERT_THAT(keys, ::testing::UnorderedElementsAre(early_sample_id_, late_sample_id_, inserted_sample_id));

  // Now try only the last 2 files

  modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse> writer2(&call, &context);
  request.set_start_timestamp(50);
  request.set_end_timestamp(250);

  status = storage_service
               .GetDataInInterval_Impl<modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse>>(
                   &context, &request, &writer2);
  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetDataInIntervalResponse>& responses2 = writer2.get_responses();
  ASSERT_EQ(responses2.size(), 1);
  const modyn::storage::GetDataInIntervalResponse& response2 = responses2[0];
  std::vector<int64_t> keys2;
  keys2.reserve(response2.keys_size());
  for (const auto& key : response2.keys()) {
    keys2.push_back(key);
  }
  ASSERT_THAT(keys2, ::testing::UnorderedElementsAre(late_sample_id_, inserted_sample_id));

  // And now no files
  modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse> writer3(&call, &context);
  request.set_start_timestamp(101);
  request.set_end_timestamp(180);
  status = storage_service
               .GetDataInInterval_Impl<modyn::storage::MockServerWriter<modyn::storage::GetDataInIntervalResponse>>(
                   &context, &request, &writer3);
  ASSERT_TRUE(status.ok());
  const std::vector<modyn::storage::GetDataInIntervalResponse>& responses3 = writer3.get_responses();
  ASSERT_EQ(responses3.size(), 0);
}

TEST_F(StorageServiceImplTest, TestDeleteDataErrorHandling) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  modyn::storage::DeleteDataRequest request;
  modyn::storage::DeleteDataResponse response;

  ServerContext context;

  // Test case when dataset does not exist
  request.set_dataset_id("non_existent_dataset");
  request.add_keys(1);
  Status status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_FALSE(response.success());

  // Test case when no samples found for provided keys
  request.set_dataset_id("test_dataset");
  request.clear_keys();
  request.add_keys(99999);  // Assuming no sample with this key
  status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_FALSE(response.success());

  // Test case when no files found for the samples
  // Here we create a sample that doesn't link to a file.
  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)
  session
      << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 99999, 0, 0)";  // Assuming no file
                                                                                                    // with this id
  request.clear_keys();
  request.add_keys(0);
  status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_FALSE(response.success());
}

TEST_F(StorageServiceImplTest, TestGetPartitionForWorker) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  std::tuple<int64_t, int64_t> result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_partition_for_worker(0, 1, 1));
  ASSERT_EQ(std::get<0>(result), 0);
  ASSERT_EQ(std::get<1>(result), 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_partition_for_worker(0, 2, 2));
  ASSERT_EQ(std::get<0>(result), 0);
  ASSERT_EQ(std::get<1>(result), 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_partition_for_worker(1, 2, 2));
  ASSERT_EQ(std::get<0>(result), 1);
  ASSERT_EQ(std::get<1>(result), 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_partition_for_worker(0, 3, 9));
  ASSERT_EQ(std::get<0>(result), 0);
  ASSERT_EQ(std::get<1>(result), 3);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_partition_for_worker(1, 3, 9));
  ASSERT_EQ(std::get<0>(result), 3);
  ASSERT_EQ(std::get<1>(result), 3);
}

TEST_F(StorageServiceImplTest, TestGetNumberOfSamplesInFile) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  int64_t result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_number_of_samples_in_file(1, session, 1));
  ASSERT_EQ(result, 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_number_of_samples_in_file(2, session, 1));
  ASSERT_EQ(result, 1);

  const std::string sql_expression = fmt::format(
      "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/test_file2.txt', "
      "100, 10)",
      tmp_dir_);
  session << sql_expression;

  ASSERT_NO_THROW(result = StorageServiceImpl::get_number_of_samples_in_file(3, session, 1));
  ASSERT_EQ(result, 10);
}

TEST_F(StorageServiceImplTest, TestGetFileIds) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  std::vector<int64_t> result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1, 1, 100));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1, 1, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1, 2, 100));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1, 2));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids(session, 1, 1, 100));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);
}

TEST_F(StorageServiceImplTest, TestGetFileCount) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  int64_t result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_count(session, 1, 1, 100));
  ASSERT_EQ(result, 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_count(session, 1, 1, 1));
  ASSERT_EQ(result, 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_count(session, 1, 2, 100));
  ASSERT_EQ(result, 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_count(session, 1, -1, -1));
  ASSERT_EQ(result, 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_count(session, 1, 2, -1));
  ASSERT_EQ(result, 1);
}

TEST_F(StorageServiceImplTest, TestGetFileIdsGivenNumberOfFiles) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  std::vector<int64_t> result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_given_number_of_files(session, 1, 1, 100, 2));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_given_number_of_files(session, 1, 1, 1, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_given_number_of_files(session, 1, 2, 100, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_given_number_of_files(session, 1, -1, -1, 2));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_given_number_of_files(session, 1, 2, -1, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 1);
}

TEST_F(StorageServiceImplTest, TestGetDatasetId) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  int64_t result;
  ASSERT_NO_THROW(result = StorageServiceImpl::get_dataset_id(session, "test_dataset"));
  ASSERT_EQ(result, 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_dataset_id(session, "non_existent_dataset"));
  ASSERT_EQ(result, -1);
}

TEST_F(StorageServiceImplTest, TestGetFileIdsForSamples) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 2, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 3, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 4, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 4, 0, 1)";

  std::vector<int64_t> result;
  std::vector<int64_t> request_keys = {1, 2, 3};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_for_samples(request_keys, 1, session));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  request_keys = {3, 4, 5, 6};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_for_samples(request_keys, 1, session));
  ASSERT_EQ(result.size(), 4);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);
  ASSERT_EQ(result[2], 3);
  ASSERT_EQ(result[3], 4);

  request_keys = {3, 4};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_for_samples(request_keys, 1, session));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);

  request_keys = {1, 2, 3, 4, 5, 6, 7};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_for_samples(request_keys, 1, session));
  ASSERT_EQ(result.size(), 4);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 2);
  ASSERT_EQ(result[2], 3);
  ASSERT_EQ(result[3], 4);
}

TEST_F(StorageServiceImplTest, TestGetFileIdsPerThread) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);

  std::vector<std::vector<int64_t>> result;
  std::vector<int64_t> file_ids = {1, 2, 3, 4, 5};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_per_thread(file_ids, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 5);
  ASSERT_EQ(result[0][0], 1);
  ASSERT_EQ(result[0][1], 2);
  ASSERT_EQ(result[0][2], 3);
  ASSERT_EQ(result[0][3], 4);
  ASSERT_EQ(result[0][4], 5);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_per_thread(file_ids, 2));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0].size(), 2);
  ASSERT_EQ(result[0][0], 1);
  ASSERT_EQ(result[0][1], 2);
  ASSERT_EQ(result[1].size(), 3);
  ASSERT_EQ(result[1][0], 3);
  ASSERT_EQ(result[1][1], 4);
  ASSERT_EQ(result[1][2], 5);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_per_thread(file_ids, 3));
  ASSERT_EQ(result.size(), 3);
  ASSERT_EQ(result[0].size(), 1);
  ASSERT_EQ(result[0][0], 1);
  ASSERT_EQ(result[1].size(), 1);
  ASSERT_EQ(result[1][0], 2);
  ASSERT_EQ(result[2].size(), 3);
  ASSERT_EQ(result[2][0], 3);
  ASSERT_EQ(result[2][1], 4);
  ASSERT_EQ(result[2][2], 5);

  file_ids = {1};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_per_thread(file_ids, 1));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 1);
  ASSERT_EQ(result[0][0], 1);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_file_ids_per_thread(file_ids, 2));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0].size(), 1);
  ASSERT_EQ(result[0][0], 1);
  ASSERT_EQ(result[1].size(), 0);
}

TEST_F(StorageServiceImplTest, TestGetSamplesCorrespondingToFiles) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 2, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 3, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 4, 0, 1)";
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 4, 0, 1)";

  std::vector<int64_t> result;
  const std::vector<int64_t> request_keys = {1, 2, 3, 4, 5, 6, 7};
  ASSERT_NO_THROW(result = StorageServiceImpl::get_samples_corresponding_to_file(1, 1, request_keys, session));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 1);
  ASSERT_EQ(result[1], 3);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_samples_corresponding_to_file(2, 1, request_keys, session));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 2);
  ASSERT_EQ(result[1], 4);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_samples_corresponding_to_file(3, 1, request_keys, session));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0], 5);

  ASSERT_NO_THROW(result = StorageServiceImpl::get_samples_corresponding_to_file(4, 1, request_keys, session));
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result[0], 6);
  ASSERT_EQ(result[1], 7);
}

TEST_F(StorageServiceImplTest, TestGetDatasetData) {
  const YAML::Node config = YAML::LoadFile("config.yaml");

  const StorageDatabaseConnection connection(config);
  soci::session session =
      connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)

  DatasetData result;
  std::string dataset_name = "test_dataset";
  ASSERT_NO_THROW(result = StorageServiceImpl::get_dataset_data(session, dataset_name));
  ASSERT_EQ(result.dataset_id, 1);
  ASSERT_EQ(result.base_path, tmp_dir_);
  ASSERT_EQ(result.filesystem_wrapper_type, FilesystemWrapperType::LOCAL);
  ASSERT_EQ(result.file_wrapper_type, FileWrapperType::SINGLE_SAMPLE);
  ASSERT_EQ(result.file_wrapper_config, StorageTestUtils::get_dummy_file_wrapper_config_inline());

  dataset_name = "non_existent_dataset";
  ASSERT_NO_THROW(result = StorageServiceImpl::get_dataset_data(session, dataset_name));
  ASSERT_EQ(result.dataset_id, -1);
  ASSERT_EQ(result.base_path, "");
  ASSERT_EQ(result.filesystem_wrapper_type, FilesystemWrapperType::INVALID_FSW);
  ASSERT_EQ(result.file_wrapper_type, FileWrapperType::INVALID_FW);
}