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
  }

  void TearDown() override {}
};

TEST_F(StorageServiceImplTest, TestGet) {}

TEST_F(StorageServiceImplTest, TestGetNewDataSince) {}

TEST_F(StorageServiceImplTest, TestGetDataInInterval) {}

TEST_F(StorageServiceImplTest, TestCheckAvailability) {
  // Set up server context
  grpc::ServerContext context;

  // Set up request
  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  // Set up response
  modyn::storage::DatasetAvailableResponse response;

  // Set up service
  const YAML::Node config = YAML::LoadFile("config.yaml");
  StorageServiceImpl service(config);

  // Test the CheckAvailability method
  grpc::Status status = service.CheckAvailability(&context, &request, &response);

  // Check the status and the response
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(response.available());

  // Test the CheckAvailability method with a non-existing dataset
  request.set_dataset_id("non_existing_dataset");
  status = service.CheckAvailability(&context, &request, &response);

  // Check the status and the response
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(response.available());

  ASSERT_EQ(status.error_code(), grpc::StatusCode::NOT_FOUND);
}

TEST_F(StorageServiceImplTest, TestGetCurrentTimestamp) {}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {}

TEST_F(StorageServiceImplTest, TestDeleteData) {}
