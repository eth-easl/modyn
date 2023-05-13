#include "internal/file_watcher/file_watcher.hpp"

#include <gtest/gtest.h>
#include <soci/soci.h>
#include <soci/sqlite3/soci-sqlite3.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "internal/database/storage_database_connection.hpp"
#include "test_utils.hpp"
#include "unit/internal/filesystem_wrapper/mock_filesystem_wrapper.hpp"

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
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  ASSERT_NO_THROW(FileWatcher watcher("config.yaml", 0, true, stop_file_watcher));
}

TEST_F(FileWatcherTest, TestSeek) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  // Add a dataset to the database
  connection.add_dataset("test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  // Add a file to the temporary directory
  std::ofstream file("tmp/test_file.txt");
  file << "test";
  file.close();

  // Seek the temporary directory
  ASSERT_NO_THROW(watcher.seek());

  // Check if the file is added to the database
  std::string file_path = "tmp/test_file.txt";
  *sql << "SELECT * FROM files WHERE path = :path", soci::use(file_path);
  ASSERT_TRUE(sql->got_data());

  // Assert the last timestamp of the dataset is updated
  int last_timestamp;
  *sql << "SELECT last_timestamp FROM datasets WHERE dataset_id = :id", soci::use(1), soci::into(last_timestamp);

  ASSERT_TRUE(last_timestamp > 0);
}

TEST_F(FileWatcherTest, TestSeekDataset) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  connection.add_dataset("test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  ASSERT_NO_THROW(watcher.seek_dataset());
}

TEST_F(FileWatcherTest, TestExtractCheckValidFile) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(1000));

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", false, 0, &filesystem_wrapper));

  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(0));

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 1000, &filesystem_wrapper));

  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(1000));

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", true, 0, &filesystem_wrapper));

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  *sql << "INSERT INTO files (file_id, dataset_id, path, last_modified) VALUES "
          "(1, 1, 'test.txt', 1000)";

  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(1000));

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 0, &filesystem_wrapper));
}

TEST_F(FileWatcherTest, TestUpdateFilesInDirectory) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  std::vector<std::string>* files = new std::vector<std::string>();
  files->push_back("test.txt");
  MockFilesystemWrapper filesystem_wrapper;

  EXPECT_CALL(filesystem_wrapper, list(testing::_, testing::_)).WillOnce(testing::Return(files));

  ASSERT_NO_THROW(watcher.update_files_in_directory(&filesystem_wrapper, "tmp", 0));
}

TEST_F(FileWatcherTest, TestFallbackInsertion) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  std::vector<std::tuple<long long, long long, int, int>> files;

  // Add some files to the vector
  files.push_back(std::make_tuple(1, 1, 1, 1));
  files.push_back(std::make_tuple(2, 2, 2, 2));
  files.push_back(std::make_tuple(3, 3, 3, 3));

  // Insert the files into the database
  ASSERT_NO_THROW(watcher.fallback_insertion(files, sql));

  // Check if the files are added to the database
  int file_id;
  *sql << "SELECT file_id FROM files WHERE file_id = :id", soci::use(1), soci::into(file_id);
  ASSERT_TRUE(sql->got_data());

  *sql << "SELECT file_id FROM files WHERE file_id = :id", soci::use(2), soci::into(file_id);

  ASSERT_TRUE(sql->got_data());

  *sql << "SELECT file_id FROM files WHERE file_id = :id", soci::use(3), soci::into(file_id);

  ASSERT_TRUE(sql->got_data());
}

TEST_F(FileWatcherTest, TestHandleFilePaths) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher("config.yaml", 0, true, stop_file_watcher);

  std::vector<std::string>* file_paths = new std::vector<std::string>();
  file_paths->push_back("test.txt");
  file_paths->push_back("test2.txt");

  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillRepeatedly(testing::Return(1000));
  EXPECT_CALL(filesystem_wrapper, get_created_time(testing::_)).WillOnce(testing::Return(1000));

  // TODO: Also mock the file wrapper
  ASSERT_NO_THROW(watcher.handle_file_paths(file_paths, ".txt", "MOCK", &filesystem_wrapper, 0));
}