#include "../../../src/internal/database/StorageDatabaseConnection.hpp"
#include "../../../src/internal/file_watcher/FileWatcher.hpp"
#include "../../TestUtils.hpp"
#include <boost/filesystem.hpp>
#include <filesystem>
#include <gtest/gtest.h>
#include <soci/soci.h>
#include <soci/sqlite3/soci-sqlite3.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

using namespace storage;

class FileWatcherTest : public ::testing::Test {
protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory("tmp");
    YAML::Node config = YAML::LoadFile("config.yaml");
    StorageDatabaseConnection connection(config);
    connection.create_tables();
  }

  void TearDown() override {
    TestUtils::delete_dummy_yaml();
    if (std::filesystem::exists("'test.db'")) {
      std::filesystem::remove("'test.db'");
    }
    // Remove temporary directory
    std::filesystem::remove_all("tmp");
  }
};

TEST_F(FileWatcherTest, TestConstructor) {
  ASSERT_NO_THROW(FileWatcher watcher("config.yaml", 0, true));
}

TEST_F(FileWatcherTest, TestSeek) {
  FileWatcher watcher("config.yaml", 0, true);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  soci::session *sql = connection.get_session();

  // Add a dataset to the database
  connection.add_dataset(
      "test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);

  // TODO: Add a file to the temporary directory and check if it is added to the
  // database (5)
}

TEST_F(FileWatcherTest, TestSeekDataset) {
  // TODO: Test if dataset is recognized and update_files_in_directory is called
  // (10)
}

TEST_F(FileWatcherTest, TestExtractFilePathsPerThreadToFile) {
  // TODO: Check if the correct number of files is written to the file and if
  // the file is written correctly (10)
}

TEST_F(FileWatcherTest, TestExtractCheckValidFile) {
  // TODO: Check if file validation works (5)
}

TEST_F(FileWatcherTest, TestUpdateFilesInDirectory) {
  // TODO: Check if files are added to the database (15)
}

TEST_F(FileWatcherTest, TestFallbackInsertion) {
  // TODO: Check if fallback insertion works (10)
}

TEST_F(FileWatcherTest, TestHandleFilePaths) {
  // TODO: Check if handle file paths works and fallback_insertion is called
  // (10)
}