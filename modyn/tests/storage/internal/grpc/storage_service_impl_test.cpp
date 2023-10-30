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
                           "test description", "0.0.0", StorageTestUtils::get_dummy_file_wrapper_config_inline(), true);

    soci::session session =
        connection.get_session();  // NOLINT misc-const-correctness  (the soci::session cannot be const)
    std::string sql_expression = fmt::format(
        "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/test_file.txt', 100, "
        "1)",
        tmp_dir_);
    session << sql_expression;

    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 1, 0, 0)";

    sql_expression = fmt::format(
        "INSERT INTO files (dataset_id, path, updated_at, number_of_samples) VALUES (1, '{}/test_file2.txt', "
        "100, 1)",
        tmp_dir_);
    session << sql_expression;

    session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, 2, 0, 1)";

    // Create dummy files
    std::ofstream file(tmp_dir_ + "/test_file.txt");
    file << "test";
    file.close();

    file = std::ofstream(tmp_dir_ + "/test_file.lbl");
    file << "1";
    file.close();

    file = std::ofstream(tmp_dir_ + "/test_file2.txt");
    file << "test";
    file.close();

    file = std::ofstream(tmp_dir_ + "/test_file2.lbl");
    file << "2";
    file.close();
  }

  void TearDown() override {
    // Remove temporary directory
    std::filesystem::remove_all(tmp_dir_);
    std::filesystem::remove("config.yaml");
    if (std::filesystem::exists("'test.db'")) {
      std::filesystem::remove("'test.db'");
    }
  }
};

TEST_F(StorageServiceImplTest, TestCheckAvailability) {
  ServerContext context;

  modyn::storage::DatasetAvailableRequest request;
  request.set_dataset_id("test_dataset");

  modyn::storage::DatasetAvailableResponse response;

  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);

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
  ::StorageServiceImpl storage_service(config);

  const Status status = storage_service.GetCurrentTimestamp(&context, &request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_GE(response.timestamp(), 0);
}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);

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
  ::StorageServiceImpl storage_service(config);  // NOLINT misc-const-correctness

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

TEST_F(StorageServiceImplTest, TestDeleteDataErrorHandling) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ::StorageServiceImpl storage_service(config);

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
