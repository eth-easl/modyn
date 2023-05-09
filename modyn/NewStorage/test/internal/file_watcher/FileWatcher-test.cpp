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

  std::string test_config = "file_extension: .txt\nlabel_file_extension: .lbl";

  // Add a dataset to the database
  connection.add_dataset("test_dataset", "tmp", "LOCAL", "MOCK",
                         "test description", "0.0.0", test_config, true);

  
}