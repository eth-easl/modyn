#include "internal/file_watcher/file_watcher_watchdog.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "storage_test_utils.hpp"
#include "test_utils.hpp"

using namespace modyn::storage;

class FileWatcherWatchdogTest : public ::testing::Test {
 protected:
  std::string tmp_dir_;

  FileWatcherWatchdogTest() : tmp_dir_{modyn::test::TestUtils::get_tmp_testdir("file_watcher_watchdog_test")} {}

  void SetUp() override {
    modyn::test::TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory(tmp_dir_);
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
    connection.create_tables();
  }

  void TearDown() override {
    modyn::test::TestUtils::delete_dummy_yaml();
    if (std::filesystem::exists("test.db")) {
      std::filesystem::remove("test.db");
    }
    // Remove temporary directory
    std::filesystem::remove_all(tmp_dir_);
  }
};

TEST_F(FileWatcherWatchdogTest, TestConstructor) {
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ASSERT_NO_THROW(const FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown));
}

TEST_F(FileWatcherWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;

  const std::shared_ptr<FileWatcherWatchdog> watchdog =
      std::make_shared<FileWatcherWatchdog>(config, &stop_file_watcher, &request_shutdown);

  std::thread th(&FileWatcherWatchdog::run, watchdog);
  std::this_thread::sleep_for(std::chrono::milliseconds(2));

  stop_file_watcher = true;
  th.join();

  // Check if the watchdog has stopped
  ASSERT_FALSE(th.joinable());
}

TEST_F(FileWatcherWatchdogTest, TestStartFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);

  const StorageDatabaseConnection connection(config);

  // Add two dataset to the database
  connection.add_dataset("test_dataset1", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);
  connection.add_dataset("test_dataset2", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  watchdog.start_file_watcher_thread(1);
  std::vector<int64_t> file_watcher_threads;
  file_watcher_threads = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_threads.size(), 1);

  // Test if the file watcher process is still running
  file_watcher_threads = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_threads.size(), 1);

  watchdog.stop_file_watcher_thread(1);
  watchdog.start_file_watcher_thread(1);
  file_watcher_threads = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_threads.size(), 1);

  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestStopFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);

  const StorageDatabaseConnection connection(config);

  connection.add_dataset("test_dataset", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  watchdog.start_file_watcher_thread(1);

  std::vector<int64_t> file_watcher_threads;
  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 0);
}

TEST_F(FileWatcherWatchdogTest, TestWatchFileWatcherThreads) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);

  const StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  connection.add_dataset("test_dataset1", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_threads;
  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);

  watchdog.watch_file_watcher_threads();

  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);
  ASSERT_EQ(file_watcher_threads[0], 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 0);

  watchdog.watch_file_watcher_threads();

  file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);

  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestFileWatcherWatchdogWithNoDataset) {
  // This test ensures that the watchdog handles correctly the situation where there is no dataset in the database
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);
  const StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  // Assert that there are no running FileWatcher threads as there are no datasets
  const std::vector<int64_t> file_watcher_threads = watchdog.get_running_file_watcher_threads();
  ASSERT_TRUE(file_watcher_threads.empty());
}

TEST_F(FileWatcherWatchdogTest, TestRestartFailedFileWatcherProcess) {
  // This test checks that the watchdog successfully restarts a failed FileWatcher process
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);
  const StorageDatabaseConnection connection(config);

  connection.add_dataset("test_dataset", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  watchdog.start_file_watcher_thread(1);
  // Simulate a failure of the FileWatcher process
  watchdog.stop_file_watcher_thread(1);

  // The watchdog should detect the failure and restart the process
  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);
  ASSERT_EQ(file_watcher_threads[0], 1);
  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestAddingNewDataset) {
  // This test checks that the watchdog successfully starts a FileWatcher process for a new dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);
  const StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  // The watchdog should start a FileWatcher process for the new dataset
  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_threads.size(), 1);
  ASSERT_EQ(file_watcher_threads[0], 1);
  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestRemovingDataset) {
  // This test checks that the watchdog successfully stops a FileWatcher process for a removed dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);
  const StorageDatabaseConnection connection(config);

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", tmp_dir_, FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0",
                         modyn::storage::StorageTestUtils::get_dummy_file_wrapper_config_inline(),
                         /*ignore_last_timestamp=*/true);

  watchdog.watch_file_watcher_threads();

  // The watchdog should start a FileWatcher process for the new dataset
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Now remove the dataset from the database
  connection.delete_dataset("test_dataset", 1);

  // The watchdog should stop the FileWatcher process for the removed dataset
  watchdog.watch_file_watcher_threads();

  const std::vector<int64_t> file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_TRUE(file_watcher_threads.empty());
}

TEST_F(FileWatcherWatchdogTest, TestNoDatasetsInDB) {
  // This test checks that the watchdog does not start any FileWatcher threads if there are no datasets
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  std::atomic<bool> request_shutdown = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher, &request_shutdown);
  const StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  const std::vector<int64_t> file_watcher_threads = watchdog.get_running_file_watcher_threads();

  ASSERT_TRUE(file_watcher_threads.empty());
}
