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
    connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                           "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);
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
  std::atomic<bool> stop_file_watcher = false;
  ASSERT_NO_THROW(const FileWatcher watcher(YAML::LoadFile("config.yaml"), 1, &stop_file_watcher));
}

TEST_F(FileWatcherTest, TestSeek) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

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
  std::vector<std::string> file_paths(1);
  session << "SELECT path FROM files", soci::into(file_paths);
  ASSERT_EQ(file_paths[0], file_path);

  // Check if the sample is added to the database
  std::vector<int64_t> sample_ids(1);
  session << "SELECT sample_id FROM samples", soci::into(sample_ids);
  ASSERT_EQ(sample_ids[0], 1);

  // Assert the last timestamp of the dataset is updated
  const int32_t dataset_id = 1;
  int32_t last_timestamp;
  session << "SELECT last_timestamp FROM datasets WHERE dataset_id = :id", soci::use(dataset_id),
      soci::into(last_timestamp);

  ASSERT_TRUE(last_timestamp > 0);
}

TEST_F(FileWatcherTest, TestSeekDataset) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

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
  soci::session session = connection.get_session();
  session << "SELECT path FROM files", soci::into(file_paths);
  ASSERT_EQ(file_paths[0], file_path);

  // Check if the sample is added to the database
  std::vector<int64_t> sample_ids = std::vector<int64_t>(1);
  session << "SELECT sample_id FROM samples", soci::into(sample_ids);
  ASSERT_EQ(sample_ids[0], 1);
}

TEST_F(FileWatcherTest, TestExtractCheckValidFile) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(1000));
  watcher.filesystem_wrapper = filesystem_wrapper;

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", false, 0));

  EXPECT_CALL(*filesystem_wrapper, get_modified_time(testing::_)).WillOnce(testing::Return(0));

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 1000));

  ASSERT_TRUE(watcher.check_valid_file("test.txt", ".txt", true, 0));

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  session << "INSERT INTO files (file_id, dataset_id, path, updated_at) VALUES "
             "(1, 1, 'test.txt', 1000)";

  ASSERT_FALSE(watcher.check_valid_file("test.txt", ".txt", false, 0));
}

TEST_F(FileWatcherTest, TestUpdateFilesInDirectory) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  watcher.filesystem_wrapper = filesystem_wrapper;

  std::vector<std::string> files = std::vector<std::string>();
  files.emplace_back("test.txt");
  files.emplace_back("test.lbl");

  EXPECT_CALL(*filesystem_wrapper, list(testing::_, testing::_)).WillOnce(testing::Return(files));
  EXPECT_CALL(*filesystem_wrapper, get_modified_time(testing::_)).WillRepeatedly(testing::Return(1000));
  ON_CALL(*filesystem_wrapper, exists(testing::_)).WillByDefault(testing::Return(true));
  const std::vector<unsigned char> bytes{'1'};
  EXPECT_CALL(*filesystem_wrapper, get("test.lbl")).WillOnce(testing::Return(bytes));

  ASSERT_NO_THROW(watcher.update_files_in_directory("tmp", 0));

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  std::vector<std::string> file_paths = std::vector<std::string>(1);
  session << "SELECT path FROM files", soci::into(file_paths);
  ASSERT_EQ(file_paths[0], "test.txt");
}

TEST_F(FileWatcherTest, TestFallbackInsertion) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  const FileWatcher watcher(config, 1, &stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>> files;

  // Add some files to the vector
  files.emplace_back(1, 1, 1, 1);
  files.emplace_back(2, 2, 2, 2);
  files.emplace_back(3, 3, 3, 3);

  // Insert the files into the database
  ASSERT_NO_THROW(watcher.fallback_insertion(files));

  // Check if the files are added to the database
  int32_t file_id = 1;
  int32_t sample_id = 0;
  session << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(file_id), soci::into(sample_id);
  ASSERT_EQ(sample_id, 1);

  file_id = 2;
  session << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(file_id), soci::into(sample_id);
  ASSERT_EQ(sample_id, 2);

  file_id = 3;
  session << "SELECT sample_id FROM samples WHERE file_id = :id", soci::use(file_id), soci::into(sample_id);
  ASSERT_EQ(sample_id, 3);
}

TEST_F(FileWatcherTest, TestHandleFilePaths) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  std::vector<std::string> files = std::vector<std::string>();
  files.emplace_back("test.txt");
  files.emplace_back("test.lbl");
  files.emplace_back("test2.txt");
  files.emplace_back("test2.lbl");

  const StorageDatabaseConnection connection(config);

  soci::session session = connection.get_session();

  const std::shared_ptr<MockFilesystemWrapper> filesystem_wrapper = std::make_shared<MockFilesystemWrapper>();
  EXPECT_CALL(*filesystem_wrapper, get_modified_time(testing::_)).WillRepeatedly(testing::Return(1000));
  std::vector<unsigned char> bytes{'1'};
  EXPECT_CALL(*filesystem_wrapper, get("test.lbl")).WillOnce(testing::Return(bytes));
  bytes = {'2'};
  EXPECT_CALL(*filesystem_wrapper, get("test2.lbl")).WillOnce(testing::Return(bytes));
  EXPECT_CALL(*filesystem_wrapper, exists(testing::_)).WillRepeatedly(testing::Return(true));
  watcher.filesystem_wrapper = filesystem_wrapper;

  const YAML::Node file_wrapper_config_node = YAML::Load(TestUtils::get_dummy_file_wrapper_config_inline());

  ASSERT_NO_THROW(
      watcher.handle_file_paths(files, ".txt", FileWrapperType::SINGLE_SAMPLE, 0, file_wrapper_config_node));

  // Check if the samples are added to the database
  int32_t sample_id1;
  int32_t label1;
  int32_t file_id = 1;
  session << "SELECT sample_id, label FROM samples WHERE file_id = :id", soci::use(file_id), soci::into(sample_id1),
      soci::into(label1);
  ASSERT_EQ(sample_id1, 1);
  ASSERT_EQ(label1, 1);

  int32_t sample_id2;
  int32_t label2;
  file_id = 2;
  session << "SELECT sample_id, label FROM samples WHERE file_id = :id", soci::use(file_id), soci::into(sample_id2),
      soci::into(label2);
  ASSERT_EQ(sample_id2, 2);
  ASSERT_EQ(label2, 2);

  // Check if the files are added to the database
  int32_t output_file_id = 0;
  int32_t input_file_id = 1;
  session << "SELECT file_id FROM files WHERE file_id = :id", soci::use(input_file_id), soci::into(output_file_id);
  ASSERT_EQ(output_file_id, 1);

  input_file_id = 2;
  session << "SELECT file_id FROM files WHERE file_id = :id", soci::use(input_file_id), soci::into(output_file_id);
  ASSERT_EQ(output_file_id, 2);
}

TEST_F(FileWatcherTest, TestConstructorWithInvalidInterval) {
  std::atomic<bool> stop_file_watcher = false;
  const FileWatcher watcher(YAML::LoadFile("config.yaml"), -1, &stop_file_watcher);
  ASSERT_TRUE(watcher.stop_file_watcher_->load());
}

TEST_F(FileWatcherTest, TestConstructorWithNullStopFileWatcher) {
  ASSERT_THROW(const FileWatcher watcher(YAML::LoadFile("config.yaml"), 1, nullptr), std::runtime_error);
}

TEST_F(FileWatcherTest, TestSeekWithNonExistentDirectory) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);
  std::filesystem::remove_all("tmp");

  watcher.seek();
  ASSERT_TRUE(watcher.stop_file_watcher_->load());
}

TEST_F(FileWatcherTest, TestSeekDatasetWithNonExistentDirectory) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);
  std::filesystem::remove_all("tmp");

  ASSERT_THROW(watcher.seek_dataset(), std::runtime_error);
}

TEST_F(FileWatcherTest, TestCheckValidFileWithInvalidPath) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  ASSERT_FALSE(watcher.check_valid_file("", ".txt", false, 0));
  ASSERT_FALSE(watcher.check_valid_file("test", ".txt", true, 0));
}

TEST_F(FileWatcherTest, TestUpdateFilesInDirectoryWithNonExistentDirectory) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);
  std::filesystem::remove_all("tmp");

  ASSERT_THROW(watcher.update_files_in_directory("tmp", 0), std::runtime_error);
}

TEST_F(FileWatcherTest, TestFallbackInsertionWithEmptyVector) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  const FileWatcher watcher(config, 1, &stop_file_watcher);

  std::vector<std::tuple<int64_t, int64_t, int32_t, int32_t>> files;

  ASSERT_NO_THROW(watcher.fallback_insertion(files));
}

TEST_F(FileWatcherTest, TestHandleFilePathsWithEmptyVector) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  std::vector<std::string> files;

  const YAML::Node file_wrapper_config_node = YAML::Load(TestUtils::get_dummy_file_wrapper_config_inline());

  ASSERT_NO_THROW(
      watcher.handle_file_paths(files, ".txt", FileWrapperType::SINGLE_SAMPLE, 0, file_wrapper_config_node));
}

TEST_F(FileWatcherTest, TestMultipleFileHandling) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  int16_t number_of_files = 10;

  // Add several files to the temporary directory
  for (int i = 0; i < number_of_files; i++) {
    std::ofstream file("tmp/test_file" + std::to_string(i) + ".txt");
    file << "test";
    file.close();

    file = std::ofstream("tmp/test_file" + std::to_string(i) + ".lbl");
    file << i;
    file.close();
  }

  // Seek the temporary directory
  ASSERT_NO_THROW(watcher.seek());

  const StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  // Check if the files are added to the database
  std::vector<std::string> file_paths(number_of_files);
  session << "SELECT path FROM files", soci::into(file_paths);

  // Make sure all files were detected and processed
  for (int i = 0; i < number_of_files; i++) {
    ASSERT_TRUE(std::find(file_paths.begin(), file_paths.end(), "tmp/test_file" + std::to_string(i) + ".txt") !=
                file_paths.end());
  }
}

TEST_F(FileWatcherTest, TestDirectoryUpdateWhileRunning) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcher watcher(config, 1, &stop_file_watcher);

  std::thread watcher_thread([&watcher, &stop_file_watcher]() {
    while (!stop_file_watcher) {
      watcher.seek();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });

  // Add a file to the temporary directory
  std::ofstream file("tmp/test_file1.txt");
  file << "test";
  file.close();
  file = std::ofstream("tmp/test_file1.lbl");
  file << "1";
  file.close();

  std::this_thread::sleep_for(std::chrono::seconds(2));  // wait for the watcher to process

  const StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  // Check if the file is added to the database
  std::string file_path;
  session << "SELECT path FROM files WHERE file_id=1", soci::into(file_path);
  ASSERT_EQ(file_path, "tmp/test_file1.txt");

  // Add another file to the temporary directory
  file = std::ofstream("tmp/test_file2.txt");
  file << "test";
  file.close();
  file = std::ofstream("tmp/test_file2.lbl");
  file << "2";
  file.close();

  std::this_thread::sleep_for(std::chrono::seconds(2));  // wait for the watcher to process

  // Check if the second file is added to the database
  session << "SELECT path FROM files WHERE file_id=2", soci::into(file_path);
  ASSERT_EQ(file_path, "tmp/test_file2.txt");

  stop_file_watcher = true;
  watcher_thread.join();
}

TEST_F(FileWatcherTest, TestMultithreadedInsertion) {
  // Define test directory and files
  const std::string directory_path = "tmp/test_directory";
  const int num_files = 20;

  // Create test directory
  ASSERT_TRUE(std::filesystem::create_directory(directory_path));

  // Create several files in the directory
  for (int i = 0; i < num_files; i++) {
    std::ofstream file(directory_path + "/test_file" + std::to_string(i) + ".txt");
    file << "test";
    file.close();

    file = std::ofstream(directory_path + "/test_file" + std::to_string(i) + ".lbl");
    file << i;
    file.close();
  }

  // Create a configuration with multiple insertion threads
  YAML::Node config = YAML::LoadFile("config.yaml");
  config["storage"]["insertion_threads"] = 2;

  // Create a FileWatcher instance with the multithreaded configuration
  std::atomic<bool> stop_file_watcher = false;
  storage::FileWatcher watcher(config, 1, &stop_file_watcher, 2);

  // Call the FileWatcher's seek function
  watcher.seek();

  // Check that all files have been processed and inserted into the database
  const storage::StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  std::vector<std::string> file_paths(num_files);
  session << "SELECT path FROM files", soci::into(file_paths);
  
  for (const auto& file_path : file_paths) {
    ASSERT_TRUE(std::filesystem::exists(file_path));
  }

  // Clean up test directory
  std::filesystem::remove_all(directory_path);
}
