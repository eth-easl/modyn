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
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
    connection.create_tables();

    // Add a dataset to the database
    connection.add_dataset("test_dataset", "tmp", "LOCAL", "SINGLE_SAMPLE", "test description", "0.0.0",
                           TestUtils::get_dummy_file_wrapper_config_inline(), true);
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
  ASSERT_NO_THROW(const FileWatcher watcher(YAML::LoadFile("config.yaml"), 1, stop_file_watcher));
}

TEST_F(FileWatcherTest, TestSeek) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher(config, 1, stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  // Add a file to the temporary directory
  std::ofstream file("tmp/test_file.txt");
  file << "test";
  file.close();

  file = std::ofstream("tmp/test_file.lbl");
  file << "1";
  file.close();

  // Seek the temporary directory
  ASSERT_NO_THROW(watcher.seek());

  // Check if the file is added to the database
  const std::string file_path = "tmp/test_file.txt";
  std::vector<std::string> file_paths = std::vector<std::string>(1);
  *sql << "SELECT path FROM files", soci::into(file_paths);
  ASSERT_EQ(file_paths[0], file_path);

  // Check if the sample is added to the database
  std::vector<int64_t> sample_ids = std::vector<int64_t>(1);
  *sql << "SELECT sample_id FROM samples", soci::into(sample_ids);
  ASSERT_EQ(sample_ids[0], 1);

  // Assert the last timestamp of the dataset is updated
  int32_t last_timestamp;
  *sql << "SELECT last_timestamp FROM datasets WHERE dataset_id = :id", soci::use(1), soci::into(last_timestamp);

  ASSERT_TRUE(last_timestamp > 0);
}

TEST_F(FileWatcherTest, TestSeekDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher(config, 1, stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  // Add a file to the temporary directory
  std::ofstream file("tmp/test_file.txt");
  file << "test";
  file.close();

  file = std::ofstream("tmp/test_file.lbl");
  file << "1";
  file.close();

  ASSERT_NO_THROW(watcher.seek_dataset());

  // Check if the file is added to the database
  const std::string file_path = "tmp/test_file.txt";
  std::vector<std::string> file_paths = std::vector<std::string>(1);
  soci::session* sql = connection.get_session();
  *sql << "SELECT path FROM files", soci::into(file_paths);
  ASSERT_EQ(file_paths[0], file_path);

  // Check if the sample is added to the database
  std::vector<int64_t> sample_ids = std::vector<int64_t>(1);
  *sql << "SELECT sample_id FROM samples", soci::into(sample_ids);
  ASSERT_EQ(sample_ids[0], 1);
}

TEST_F(FileWatcherTest, TestExtractCheckValidFile) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher(config, 1, stop_file_watcher);

  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(1000));
  watcher.filesystem_wrapper = std::make_shared<MockFilesystemWrapper>(filesystem_wrapper);

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", false, 0));

  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(0));

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 1000));

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", true, 0));

  const StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  *sql << "INSERT INTO files (file_id, dataset_id, path, updated_at) VALUES "
          "(1, 1, 'test.txt', 1000)";

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 0));
}

TEST_F(FileWatcherTest, TestUpdateFilesInDirectory) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher(config, 1, stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  std::vector<std::string> files = std::vector<std::string>();
  files.emplace_back("test.txt");
  files.emplace_back("test.lbl");
  MockFilesystemWrapper filesystem_wrapper;

  EXPECT_CALL(filesystem_wrapper, list(testing::_, testing::_)).WillOnce(testing::Return(files));
  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillRepeatedly(testing::Return(1000));
  EXPECT_CALL(filesystem_wrapper, get_created_time(testing::_)).WillOnce(testing::Return(1000));
  const std::vector<unsigned char> bytes{'1'};
  EXPECT_CALL(filesystem_wrapper, get(testing::_)).WillOnce(testing::Return(bytes));

  watcher.filesystem_wrapper = std::make_shared<MockFilesystemWrapper>(filesystem_wrapper);

  ASSERT_NO_THROW(watcher.update_files_in_directory("tmp", 0));
}

TEST_F(FileWatcherTest, TestFallbackInsertion) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  const FileWatcher watcher(config, 1, stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>> files;

  // Add some files to the vector
  files.emplace_back(1, 1, 1, 1);
  files.emplace_back(2, 2, 2, 2);
  files.emplace_back(3, 3, 3, 3);

  // Insert the files into the database
  ASSERT_NO_THROW(watcher.fallback_insertion(files, sql));

  // Check if the files are added to the database
  int32_t file_id;
  *sql << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(1), soci::into(file_id);
  ASSERT_EQ(file_id, 1);

  *sql << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(2), soci::into(file_id);
  ASSERT_EQ(file_id, 2);

  *sql << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(3), soci::into(file_id);
  ASSERT_EQ(file_id, 3);
}

TEST_F(FileWatcherTest, TestHandleFilePaths) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatcher watcher(config, 1, stop_file_watcher);

  std::vector<std::string> files = std::vector<std::string>();
  files.emplace_back("test.txt");
  files.emplace_back("test.lbl");
  files.emplace_back("test2.txt");
  files.emplace_back("test2.lbl");

  const StorageDatabaseConnection connection(config);

  soci::session* sql = connection.get_session();

  MockFilesystemWrapper filesystem_wrapper;
  EXPECT_CALL(filesystem_wrapper, get_modified_time(testing::_)).WillRepeatedly(testing::Return(1000));
  EXPECT_CALL(filesystem_wrapper, get_created_time(testing::_)).WillRepeatedly(testing::Return(1000));
  std::vector<unsigned char> bytes{'1'};
  EXPECT_CALL(filesystem_wrapper, get("test.lbl")).WillOnce(testing::Return(bytes));
  bytes = {'2'};
  EXPECT_CALL(filesystem_wrapper, get("test2.lbl")).WillOnce(testing::Return(bytes));
  watcher.filesystem_wrapper = std::make_shared<MockFilesystemWrapper>(filesystem_wrapper);

  const YAML::Node file_wrapper_config_node = YAML::Load(TestUtils::get_dummy_file_wrapper_config_inline());

  ASSERT_NO_THROW(watcher.handle_file_paths(files, ".txt", "SINGLE_SAMPLE", 0, file_wrapper_config_node));

  // Check if the samples are added to the database
  int32_t sample_id1;
  int32_t label1;
  *sql << "SELECT sample_id, label FROM samples WHERE file_id = :id", soci::use(1), soci::into(sample_id1),
      soci::into(label1);
  ASSERT_EQ(sample_id1, 1);
  ASSERT_EQ(label1, 1);

  int32_t sample_id2;
  int32_t label2;
  *sql << "SELECT sample_id, label FROM samples WHERE file_id = :id", soci::use(2), soci::into(sample_id2),
      soci::into(label2);
  ASSERT_EQ(sample_id2, 2);
  ASSERT_EQ(label2, 2);

  // Check if the files are added to the database
  int32_t file_id;
  *sql << "SELECT file_id FROM files WHERE file_id = :id", soci::use(1), soci::into(file_id);
  ASSERT_EQ(file_id, 1);

  *sql << "SELECT file_id FROM files WHERE file_id = :id", soci::use(2), soci::into(file_id);
  ASSERT_EQ(file_id, 2);
}