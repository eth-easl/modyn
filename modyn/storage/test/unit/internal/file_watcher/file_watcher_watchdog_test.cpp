#include "internal/file_watcher/file_watcher_watchdog.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "test_utils.hpp"

using namespace storage::file_watcher;
using namespace storage::test;

class FileWatcherWatchdogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory("tmp");
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const storage::database::StorageDatabaseConnection connection(config);
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

TEST_F(FileWatcherWatchdogTest, TestConstructor) {
  std::atomic<bool> stop_file_watcher = false;
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ASSERT_NO_THROW(const FileWatcherWatchdog watchdog(config, &stop_file_watcher));
}

TEST_F(FileWatcherWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;

  const std::shared_ptr<FileWatcherWatchdog> watchdog =
      std::make_shared<FileWatcherWatchdog>(config, &stop_file_watcher);

  std::thread th(&FileWatcherWatchdog::run, watchdog);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  stop_file_watcher = true;
  th.join();

  // Check if the watchdog has stopped
  ASSERT_FALSE(th.joinable());
}

TEST_F(FileWatcherWatchdogTest, TestStartFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);

  const storage::database::StorageDatabaseConnection connection(config);

  // Add two dataset to the database
  connection.add_dataset("test_dataset1", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_thread(1, 0);
  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  // Test if the file watcher process is still running
  file_watcher_processes = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_thread(1);
  watchdog.start_file_watcher_thread(1, 0);
  file_watcher_processes = watchdog.get_running_file_watcher_threads();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestStopFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);

  const storage::database::StorageDatabaseConnection connection(config);

  connection.add_dataset("test_dataset", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_thread(1, 0);

  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatcherWatchdogTest, TestWatchFileWatcherProcesses) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);

  storage::database::StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  connection.add_dataset("test_dataset1", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.watch_file_watcher_threads();

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_threads();

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_threads();

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_thread(1);

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_threads();

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  watchdog.stop_file_watcher_thread(1);

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_threads();

  file_watcher_processes = watchdog.get_running_file_watcher_threads();

  // Restarted more than 3 times, should not be restarted again
  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatcherWatchdogTest, TestFileWatcherWatchdogWithNoDataset) {
  // This test ensures that the watchdog handles correctly the situation where there is no dataset in the database
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  // Assert that there are no running FileWatcher processes as there are no datasets
  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();
  ASSERT_TRUE(file_watcher_processes.empty());
}

TEST_F(FileWatcherWatchdogTest, TestRestartFailedFileWatcherProcess) {
  // This test checks that the watchdog successfully restarts a failed FileWatcher process
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  connection.add_dataset("test_dataset", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_thread(1, 0);
  // Simulate a failure of the FileWatcher process
  watchdog.stop_file_watcher_thread(1);

  // The watchdog should detect the failure and restart the process
  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);
  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestAddingNewDataset) {
  // This test checks that the watchdog successfully starts a FileWatcher process for a new dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  // The watchdog should start a FileWatcher process for the new dataset
  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);
  watchdog.stop_file_watcher_thread(1);
}

TEST_F(FileWatcherWatchdogTest, TestRemovingDataset) {
  // This test checks that the watchdog successfully stops a FileWatcher process for a removed dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_threads();

  // Wait for the FileWatcher process to start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Now remove the dataset from the database
  connection.delete_dataset("test_dataset");

  // The watchdog should stop the FileWatcher process for the removed dataset
  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_TRUE(file_watcher_processes.size() == 1);
  ASSERT_EQ(file_watcher_processes[0], 2);

  watchdog.stop_file_watcher_thread(2);
}

TEST_F(FileWatcherWatchdogTest, TestNoDatasetsInDB) {
  // This test checks that the watchdog does not start any FileWatcher processes if there are no datasets
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_TRUE(file_watcher_processes.empty());
}

TEST_F(FileWatcherWatchdogTest, TestMultipleDatasets) {
  // This test checks that the watchdog correctly manages multiple FileWatcher processes for multiple datasets
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatcherWatchdog watchdog(config, &stop_file_watcher);
  storage::database::StorageDatabaseConnection connection(config);

  // Add multiple datasets to the database
  connection.add_dataset("test_dataset1", "tmp1", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description1", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp2", storage::filesystem_wrapper::FilesystemWrapperType::LOCAL,
                         storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE, "test description2", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_threads();

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_threads();

  ASSERT_EQ(file_watcher_processes.size(), 2);
  watchdog.stop_file_watcher_thread(1);
  watchdog.stop_file_watcher_thread(2);
}