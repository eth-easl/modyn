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

TEST_F(FileWatchdogTest, TestConstructor) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  ASSERT_NO_THROW(FileWatchdog watchdog("config.yaml", stop_file_watcher));
}

TEST_F(FileWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);

  FileWatchdog* watchdog = new FileWatchdog("config.yaml", stop_file_watcher);

  std::thread th(&FileWatchdog::run, watchdog);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  *stop_file_watcher = true;
  th.join();

  // Check if the watchdog has stopped
  ASSERT_FALSE(th.joinable());
}

TEST_F(FileWatchdogTest, TestStartFileWatcherProcess) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection connection(config);

  // Add two dataset to the database
  connection.add_dataset("test_dataset1", "tmp", "LOCAL", "SINGLE_SAMPLE", "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);
  connection.add_dataset("test_dataset2", "tmp", "LOCAL", "SINGLE_SAMPLE", "test description", "0.0.0",
                         TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(1);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  // Test if the file watcher process is still running
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1);

  watchdog.start_file_watcher_process(2);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(2);
}

TEST_F(FileWatchdogTest, TestStopFileWatcherProcess) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection* connection = new StorageDatabaseConnection(config);

  connection->add_dataset("test_dataset", "tmp", "LOCAL", "SINGLE_SAMPLE", "test description", "0.0.0",
                          TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(1);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(1);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatchdogTest, TestWatchFileWatcherProcesses) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher = std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection* connection = new StorageDatabaseConnection(config);

  watchdog.watch_file_watcher_processes(connection);

  soci::session* sql = connection->get_session();

  connection->add_dataset("test_dataset1", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
                          TestUtils::get_dummy_file_wrapper_config_inline(), true);

  connection->add_dataset("test_dataset2", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
                          TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(connection);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 2);

  *sql << "DELETE FROM datasets WHERE name = 'test_dataset1'";

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);
  ASSERT_EQ(file_watcher_processes[0], 2);

  watchdog.stop_file_watcher_process(2);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(2);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.stop_file_watcher_process(2);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  // Restarted more than 3 times, should not be restarted again
  ASSERT_EQ(file_watcher_processes.size(), 0);
}