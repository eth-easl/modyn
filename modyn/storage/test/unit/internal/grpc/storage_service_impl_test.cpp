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

using namespace storage;

class StorageServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory("tmp");
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
    connection.create_tables();

    // Add a dataset to the database
    connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                           "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

    soci::session session = connection.get_session();
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
  grpc::ServerContext context;

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DatasetAvailableResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  grpc::Status status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(response.available());

  request.set_dataset_id("non_existing_dataset");
  status = storage_service.CheckAvailability(&context, &request, &response);

  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(response.available());

  ASSERT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(StorageServiceImplTest, TestGetCurrentTimestamp) {
  grpc::ServerContext context;

  modyn::storage::GetCurrentTimestampRequest request;

  modyn::storage::GetCurrentTimestampResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  grpc::Status status = storage_service.GetCurrentTimestamp(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_GE(response.timestamp(), 0);
}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DeleteDatasetResponse response;

  grpc::ServerContext context;

  int dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_TRUE(dataset_exists);

  grpc::Status status = storage_service.DeleteDataset(&context, &request, &response);

  ASSERT_TRUE(status.ok());

  ASSERT_TRUE(response.success());

  dataset_exists = 0;
  session << "SELECT COUNT(*) FROM datasets WHERE name = 'test_dataset'", soci::into(dataset_exists);

  ASSERT_FALSE(dataset_exists);
}

TEST_F(StorageServiceImplTest, TestDeleteData) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl storage_service(config);

  modyn::storage::DeleteDataRequest request;
  request.set_dataset_id("test_dataset");
  request.add_keys(1);

  modyn::storage::DeleteDataResponse response;

  grpc::ServerContext context;

  grpc::Status status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  int number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 1);

  ASSERT_FALSE(std::filesystem::exists("tmp/test_file"));

  ASSERT_TRUE(std::filesystem::exists("tmp/test_file2"));

  request.clear_keys();

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);

  request.add_keys(1);

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);

  request.clear_keys();
  request.add_keys(2);

  status = storage_service.DeleteData(&context, &request, &response);

  ASSERT_TRUE(status.ok());
  ASSERT_TRUE(response.success());

  number_of_samples = 0;
  session << "SELECT COUNT(*) FROM samples WHERE dataset_id = 1", soci::into(number_of_samples);

  ASSERT_EQ(number_of_samples, 0);
}
