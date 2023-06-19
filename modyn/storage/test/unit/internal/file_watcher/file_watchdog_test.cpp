#include "internal/file_watcher/file_watchdog.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <filesystem>

#include "test_utils.hpp"

using namespace storage;

class FileWatchdogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    // Create temporary directory
    std::filesystem::create_directory("tmp");
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
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

TEST_F(FileWatchdogTest, TestConstructor) {
  std::atomic<bool> stop_file_watcher = false;
  const YAML::Node config = YAML::LoadFile("config.yaml");
  ASSERT_NO_THROW(const FileWatchdog watchdog(config, &stop_file_watcher));
}

TEST_F(FileWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;

  const std::shared_ptr<FileWatchdog> watchdog = std::make_shared<FileWatchdog>(config, &stop_file_watcher);

  std::thread th(&FileWatchdog::run, watchdog);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  stop_file_watcher = true;
  th.join();

  // Check if the watchdog has stopped
  ASSERT_FALSE(th.joinable());
}

TEST_F(FileWatchdogTest, TestStartFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);

  const StorageDatabaseConnection connection(config);

  // Add two dataset to the database
  connection.add_dataset("test_dataset1", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(1, 0);
  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  // Test if the file watcher process is still running
  file_watcher_processes = watchdog.get_running_file_watcher_processes();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1);
  watchdog.start_file_watcher_process(1, 0);
  file_watcher_processes = watchdog.get_running_file_watcher_processes();
  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1);
}

TEST_F(FileWatchdogTest, TestStopFileWatcherProcess) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);

  const StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(1, 0);

  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatchdogTest, TestWatchFileWatcherProcesses) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);

  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  watchdog.watch_file_watcher_processes(&connection);

  connection.add_dataset("test_dataset1", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.watch_file_watcher_processes(&connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);

  watchdog.stop_file_watcher_process(1, /*is_test=*/true);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(&connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1, /*is_test=*/true);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(&connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1, /*is_test=*/true);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(&connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  watchdog.stop_file_watcher_process(1, /*is_test=*/true);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(&connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  // Restarted more than 3 times, should not be restarted again
  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatchdogTest, TestFileWatchdogWithNoDataset) {
  // This test ensures that the watchdog handles correctly the situation where there is no dataset in the database
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  watchdog.watch_file_watcher_processes(&connection);

  // Assert that there are no running FileWatcher processes as there are no datasets
  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();
  ASSERT_TRUE(file_watcher_processes.empty());
}

TEST_F(FileWatchdogTest, TestWatchdogStopWhenNoDatabaseConnection) {
  // This test checks the case when the database connection is lost in the middle of the watchdog operation
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);

  // Let's say we lose the database connection here (simulated by setting the pointer to nullptr)
  StorageDatabaseConnection* connection = nullptr;

  ASSERT_THROW(watchdog.watch_file_watcher_processes(connection), std::runtime_error);
}

TEST_F(FileWatchdogTest, TestRestartFailedFileWatcherProcess) {
  // This test checks that the watchdog successfully restarts a failed FileWatcher process
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(1, 0);
  // Simulate a failure of the FileWatcher process
  watchdog.stop_file_watcher_process(1, /*is_test=*/true);

  // The watchdog should detect the failure and restart the process
  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);
  watchdog.stop_file_watcher_process(1, /*is_test=*/false);
}

TEST_F(FileWatchdogTest, TestAddingNewDataset) {
  // This test checks that the watchdog successfully starts a FileWatcher process for a new dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  watchdog.watch_file_watcher_processes(&connection);

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  // The watchdog should start a FileWatcher process for the new dataset
  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 1);
  watchdog.stop_file_watcher_process(1, /*is_test=*/false);
}

TEST_F(FileWatchdogTest, TestRemovingDataset) {
  // This test checks that the watchdog successfully stops a FileWatcher process for a removed dataset
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  // Add a new dataset to the database
  connection.add_dataset("test_dataset", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(&connection);

  // Wait for the FileWatcher process to start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Now remove the dataset from the database
  connection.delete_dataset("test_dataset");

  // The watchdog should stop the FileWatcher process for the removed dataset
  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_TRUE(file_watcher_processes.size() == 1);
  ASSERT_EQ(file_watcher_processes[0], 2);

  watchdog.stop_file_watcher_process(2, /*is_test=*/false);
}

TEST_F(FileWatchdogTest, TestNoDatasetsInDB) {
  // This test checks that the watchdog does not start any FileWatcher processes if there are no datasets
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_TRUE(file_watcher_processes.empty());
}

TEST_F(FileWatchdogTest, TestMultipleDatasets) {
  // This test checks that the watchdog correctly manages multiple FileWatcher processes for multiple datasets
  const YAML::Node config = YAML::LoadFile("config.yaml");
  std::atomic<bool> stop_file_watcher = false;
  FileWatchdog watchdog(config, &stop_file_watcher);
  StorageDatabaseConnection connection = StorageDatabaseConnection(config);

  // Add multiple datasets to the database
  connection.add_dataset("test_dataset1", "tmp1", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description1", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp2", FilesystemWrapperType::LOCAL, FileWrapperType::SINGLE_SAMPLE,
                         "test description2", "0.0.0", TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(&connection);

  std::vector<int64_t> file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 2);
  watchdog.stop_file_watcher_process(1, /*is_test=*/false);
  watchdog.stop_file_watcher_process(2, /*is_test=*/false);
}