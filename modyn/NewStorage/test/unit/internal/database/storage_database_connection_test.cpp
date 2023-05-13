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
  YAML::Node config = TestUtils::get_dummy_config();
  storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.get_session());

  config["storage"]["database"]["drivername"] = "invalid";
  storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);

  ASSERT_THROW(connection2.get_session(), std::runtime_error);
}

TEST_F(StorageDatabaseConnectionTest, TestCreateTables) {
  YAML::Node config = TestUtils::get_dummy_config();
  storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session* sql = connection2.get_session();

  soci::rowset<soci::row> tables = (sql->prepare << "SELECT name FROM sqlite_master WHERE type='table';");

  // Assert datasets, files and samples tables exist
  int number_of_tables = 0;
  *sql << "SELECT COUNT(*) FROM sqlite_master WHERE type='table';", soci::into(number_of_tables);
  ASSERT_EQ(number_of_tables, 4);  // 3 tables + 1
                                   // sqlite_sequence
                                   // table
}

TEST_F(StorageDatabaseConnectionTest, TestAddDataset) {
  YAML::Node config = TestUtils::get_dummy_config();
  storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session* sql = connection2.get_session();

  // Assert no datasets exist
  int number_of_datasets = 0;
  *sql << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);

  // Add dataset
  ASSERT_TRUE(connection2.add_dataset("test_dataset", "test_base_path", "test_filesystem_wrapper_type",
                                      "test_file_wrapper_type", "test_description", "test_version",
                                      "test_file_wrapper_config", false, 0));

  // Assert dataset exists
  *sql << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 1);
  std::string dataset_name;
  *sql << "SELECT name FROM datasets;", soci::into(dataset_name);
  ASSERT_EQ(dataset_name, "test_dataset");
}

TEST_F(StorageDatabaseConnectionTest, TestDeleteDataset) {
  YAML::Node config = TestUtils::get_dummy_config();
  storage::StorageDatabaseConnection connection = storage::StorageDatabaseConnection(config);
  ASSERT_NO_THROW(connection.create_tables());

  storage::StorageDatabaseConnection connection2 = storage::StorageDatabaseConnection(config);
  soci::session* sql = connection2.get_session();

  // Assert no datasets exist
  int number_of_datasets = 0;
  *sql << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);

  // Add dataset
  ASSERT_NO_THROW(connection2.add_dataset("test_dataset", "test_base_path", "test_filesystem_wrapper_type",
                                          "test_file_wrapper_type", "test_description", "test_version",
                                          "test_file_wrapper_config", false, 0));

  // Assert dataset exists
  *sql << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 1);

  std::string dataset_name;
  *sql << "SELECT name FROM datasets;", soci::into(dataset_name);
  ASSERT_EQ(dataset_name, "test_dataset");

  // Delete dataset
  ASSERT_TRUE(connection2.delete_dataset("test_dataset"));

  // Assert no datasets exist
  *sql << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);
}
