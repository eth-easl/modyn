#include "internal/database/storage_database_connection.hpp"

#include <gtest/gtest.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "test_utils.hpp"

using namespace storage;

class StorageDatabaseConnectionTest : public ::testing::Test {
 protected:
  void TearDown() override {
    if (std::filesystem::exists("'test.db'")) {
      std::filesystem::remove("'test.db'");
    }
  }
};

TEST_F(StorageDatabaseConnectionTest, TestGetSession) {
  YAML::Node config = TestUtils::get_dummy_config();  // NOLINT
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.get_session());
}

TEST_F(StorageDatabaseConnectionTest, TestWrongParameterGetSession) {
  YAML::Node config = TestUtils::get_dummy_config();  // NOLINT
  config["storage"]["database"]["drivername"] = "invalid";
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_THROW(connection.get_session(), std::runtime_error);
}

TEST_F(StorageDatabaseConnectionTest, TestCreateTables) {
  const YAML::Node config = TestUtils::get_dummy_config();
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session session = connection2.get_session();

  const soci::rowset<soci::row> tables = (session.prepare << "SELECT name FROM sqlite_master WHERE type='table';");

  // Assert datasets, files and samples tables exist
  int number_of_tables = 0;  // NOLINT
  session << "SELECT COUNT(*) FROM sqlite_master WHERE type='table';", soci::into(number_of_tables);
  ASSERT_EQ(number_of_tables, 4);  // 3 tables + 1
                                   // sqlite_sequence
                                   // table
}

TEST_F(StorageDatabaseConnectionTest, TestCreateTablesInvalidDriver) {
  YAML::Node config = TestUtils::get_dummy_config();  // NOLINT
  config["storage"]["database"]["drivername"] = "invalid";
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_THROW(connection.create_tables(), std::runtime_error);
}

TEST_F(StorageDatabaseConnectionTest, TestAddSampleDatasetPartitionInvalidDriver) {
  YAML::Node config = TestUtils::get_dummy_config();  // NOLINT
  config["storage"]["database"]["drivername"] = "invalid";
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_THROW(connection.add_sample_dataset_partition("test_dataset"), std::runtime_error);
}

TEST_F(StorageDatabaseConnectionTest, TestAddDataset) {
  const YAML::Node config = TestUtils::get_dummy_config();
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session session = connection2.get_session();

  // Assert no datasets exist
  int number_of_datasets = 0;  // NOLINT
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);

  // Add dataset
  ASSERT_TRUE(connection2.add_dataset("test_dataset", "test_base_path", FilesystemWrapperType::LOCAL,
                                      FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                      "test_file_wrapper_config", false, 0));

  // Assert dataset exists
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 1);
  std::string dataset_name;  // NOLINT
  session << "SELECT name FROM datasets;", soci::into(dataset_name);
  ASSERT_EQ(dataset_name, "test_dataset");
}

TEST_F(StorageDatabaseConnectionTest, TestAddExistingDataset) {
  const YAML::Node config = TestUtils::get_dummy_config();
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  // Add dataset
  ASSERT_TRUE(connection.add_dataset("test_dataset", "test_base_path", FilesystemWrapperType::LOCAL,
                                     FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                     "test_file_wrapper_config", false, 0));

  // Add existing dataset
  ASSERT_TRUE(connection.add_dataset("test_dataset", "test_base_path2", FilesystemWrapperType::LOCAL,
                                     FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                     "test_file_wrapper_config", false, 0));

  soci::session session = connection.get_session();
  std::string base_path;
  session << "SELECT base_path FROM datasets where name='test_dataset';", soci::into(base_path);
  ASSERT_EQ(base_path, "test_base_path2");
}

TEST_F(StorageDatabaseConnectionTest, TestDeleteDataset) {
  const YAML::Node config = TestUtils::get_dummy_config();
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session session = connection2.get_session();

  // Assert no datasets exist
  int number_of_datasets = 0;  // NOLINT
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);

  // Add dataset
  ASSERT_NO_THROW(connection2.add_dataset("test_dataset", "test_base_path", FilesystemWrapperType::LOCAL,
                                          FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                          "test_file_wrapper_config", false, 0));

  // Assert dataset exists
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 1);

  std::string dataset_name;  // NOLINT
  session << "SELECT name FROM datasets;", soci::into(dataset_name);
  ASSERT_EQ(dataset_name, "test_dataset");

  // Delete dataset
  ASSERT_TRUE(connection2.delete_dataset("test_dataset"));

  // Assert no datasets exist
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);
}

TEST_F(StorageDatabaseConnectionTest, TestDeleteNonExistingDataset) {
  const YAML::Node config = TestUtils::get_dummy_config();
  const storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  // Delete non-existing dataset
  ASSERT_FALSE(connection.delete_dataset("non_existing_dataset"));
}
