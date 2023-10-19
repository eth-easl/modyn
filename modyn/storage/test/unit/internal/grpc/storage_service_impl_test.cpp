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
#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

using namespace storage::grpcs;
using namespace storage::test;

class StorageServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory("tmp");
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const storage::database::StorageDatabaseConnection connection(config);
    connection.create_tables();

    // Add a dataset to the database
    connection.add_dataset("test_dataset", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                           storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                           TestUtils::get_dummy_file_wrapper_config_inline(), true);

    soci::session session = connection.get_session();  // NOLINT misc-const-correctness
    session << "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, 'tmp/test_file.txt', "
               "0, 1)";

    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 0, 0)";

    session << "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, 'tmp/test_file2.txt', "
               "100, 1)";
    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 2, 0, 1)";

    // Create dummy files
    std::ofstream file("tmp/test_file.txt");
    file << "test";
    file.close();

    file = std::ofstream("tmp/test_file.lbl");
    file << "1";
    file.close();

    file = std::ofstream("tmp/test_file2.txt");
    file << "test";
    file.close();

    file = std::ofstream("tmp/test_file2.lbl");
    file << "2";
    file.close();
  }

  void TearDown() override {
    // Remove temporary directory
    std::filesystem::remove_all("tmp");
    std::filesystem::remove("config.yaml");
    if (std::filesystem::exists("'test.db'")) {
      std::filesystem::remove("'test.db'");
    }
  }
};

TEST_F(StorageServiceImplTest, TestCheckAvailability) {
  ::grpc::ServerContext context;

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DatasetAvailableResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  ::grpc::Status status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(response.available());

  request.set_dataset_id("non_existing_dataset");
  status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(response.available());

  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::NOT_FOUND);
}

TEST_F(StorageServiceImplTest, TestGetCurrentTimestamp) {
  ::grpc::ServerContext context;

  modyn::storage::GetCurrentTimestampRequest request;

  modyn::storage::GetCurrentTimestampResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  ::grpc::Status status = storage_service.GetCurrentTimestamp(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_GE(response.timestamp(), 0);
}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  const storage::database::StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();  // NOLINT misc-const-correctness

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DeleteDatasetResponse response;

  ::grpc::ServerContext context;

  int dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_TRUE(dataset_exists);

  ::grpc::Status status = storage_service.DeleteDataset(&context, &request, &response);

  ASSERT_TRUE(status.ok());

  ASSERT_TRUE(response.success());

  dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_FALSE(dataset_exists);
}

TEST_F(StorageServiceImplTest, TestDeleteData) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  modyn::storage::DeleteDataRequest request;
  request.set_dataset_id("test_dataset");
  request.add_keys(1);

  // Add an additional sample for file 1 to the database
  const storage::database::StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();  // NOLINT misc-const-correctness
  session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 1, 0)";

  modyn::storage::DeleteDataResponse response;

  ::grpc::ServerContext context;

  ::grpc::Status status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  int number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 2);

  ASSERT_FALSE(std::filesystem::exists("tmp/test_file.txt"));

  ASSERT_TRUE(std::filesystem::exists("tmp/test_file2.txt"));

  request.clear_keys();

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);

  request.add_keys(1);

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::NOT_FOUND);

  request.clear_keys();
  request.add_keys(2);

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 1);
}

TEST_F(StorageServiceImplTest, TestDeleteDataErrorHandling) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

  modyn::storage::DeleteDataRequest request;
  modyn::storage::DeleteDataResponse response;

  ::grpc::ServerContext context;

  // Test case when dataset does not exist
  request.set_dataset_id("non_existent_dataset");
  request.add_keys(1);
  ::grpc::Status status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::NOT_FOUND);
  ASSERT_FALSE(response.success());

  // Test case when no samples found for provided keys
  request.set_dataset_id("test_dataset");
  request.clear_keys();
  request.add_keys(99999);  // Assuming no sample with this key
  status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::NOT_FOUND);
  ASSERT_FALSE(response.success());

  // Test case when no files found for the samples
  // Here we create a sample that doesn't link to a file.
  const storage::database::StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();  // NOLINT misc-const-correctness
  session
      << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 99999, 0, 0)";  // Assuming no file
                                                                                                    // with this id
  request.clear_keys();
  request.add_keys(0);
  status = storage_service.DeleteData(&context, &request, &response);
  ASSERT_EQ(status.error_code(), ::grpc::StatusCode::NOT_FOUND);
  ASSERT_FALSE(response.success());
}
