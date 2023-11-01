#include "internal/database/storage_database_connection.hpp"

#include <gtest/gtest.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "modyn/utils/utils.hpp"
#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

class StorageDatabaseConnectionTest : public ::testing::Test {
 protected:
  void TearDown() override {
    if (std::filesystem::exists("test.db")) {
      std::filesystem::remove("test.db");
    }
  }
};

TEST_F(StorageDatabaseConnectionTest, TestGetSession) {
  YAML::Node config = modyn::test::TestUtils::get_dummy_config();  // NOLINT
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.get_session());
}

TEST_F(StorageDatabaseConnectionTest, TestInvalidDriver) {
  YAML::Node config = modyn::test::TestUtils::get_dummy_config();  // NOLINT
  config["storage"]["database"]["drivername"] = "invalid";
  ASSERT_THROW(const StorageDatabaseConnection connection(config), modyn::utils::ModynException);
}

TEST_F(StorageDatabaseConnectionTest, TestCreateTables) {
  const YAML::Node config = modyn::test::TestUtils::get_dummy_config();
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const StorageDatabaseConnection connection2(config);
  soci::session session = connection2.get_session();

  const soci::rowset<soci::row> tables = (session.prepare << "SELECT name FROM sqlite_master WHERE type='table';");

  // Assert datasets, files and samples tables exist
  int number_of_tables = 0;  // NOLINT
  session << "SELECT COUNT(*) FROM sqlite_master WHERE type='table';", soci::into(number_of_tables);
  ASSERT_EQ(number_of_tables, 4);  // 3 tables + 1
                                   // sqlite_sequence
                                   // table
}

TEST_F(StorageDatabaseConnectionTest, TestAddDataset) {
  const YAML::Node config = modyn::test::TestUtils::get_dummy_config();
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const StorageDatabaseConnection connection2(config);
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
  const YAML::Node config = modyn::test::TestUtils::get_dummy_config();
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.create_tables());

  // Add dataset
  ASSERT_TRUE(connection.add_dataset("test_dataset", "test_base_path", FilesystemWrapperType::LOCAL,
                                     FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                     "test_file_wrapper_config", false, 0));

  // Add existing dataset
  ASSERT_FALSE(connection.add_dataset("test_dataset", "test_base_path2", FilesystemWrapperType::LOCAL,
                                      FileWrapperType::SINGLE_SAMPLE, "test_description", "test_version",
                                      "test_file_wrapper_config", false, 0));

  soci::session session = connection.get_session();
  std::string base_path;
  session << "SELECT base_path FROM datasets where name='test_dataset';", soci::into(base_path);
  ASSERT_EQ(base_path, "test_base_path");
}

TEST_F(StorageDatabaseConnectionTest, TestDeleteDataset) {
  const YAML::Node config = modyn::test::TestUtils::get_dummy_config();
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.create_tables());

  const StorageDatabaseConnection connection2(config);
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
  std::int64_t dataset_id;   // NOLINT
  session << "SELECT name, dataset_id FROM datasets;", soci::into(dataset_name), soci::into(dataset_id);
  ASSERT_EQ(dataset_name, "test_dataset");

  // Delete dataset
  ASSERT_TRUE(connection2.delete_dataset("test_dataset", dataset_id));

  // Assert no datasets exist
  session << "SELECT COUNT(*) FROM datasets;", soci::into(number_of_datasets);
  ASSERT_EQ(number_of_datasets, 0);
}

TEST_F(StorageDatabaseConnectionTest, TestDeleteNonExistingDataset) {
  const YAML::Node config = modyn::test::TestUtils::get_dummy_config();
  const StorageDatabaseConnection connection(config);
  ASSERT_NO_THROW(connection.create_tables());
}
