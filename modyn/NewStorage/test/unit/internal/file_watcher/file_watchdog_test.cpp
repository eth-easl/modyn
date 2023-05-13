#include "internal/file_watcher/file_watchdog.hpp"
#include "test_utils.hpp"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

using namespace storage;

class FileWatchdogTest : public ::testing::Test {
protected:
  void SetUp() override {
    TestUtils::create_dummy_yaml();
    YAML::Node config = YAML::LoadFile("config.yaml");
    StorageDatabaseConnection connection(config);
    connection.create_tables();
  }

  void TearDown() override {
    TestUtils::delete_dummy_yaml();
    if (std::filesystem::exists("'test.db'")) {
      std::filesystem::remove("'test.db'");
    }
  }
};

TEST_F(FileWatchdogTest, TestConstructor) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher =
      std::make_shared<std::atomic<bool>>(false);
  ASSERT_NO_THROW(FileWatchdog watchdog("config.yaml", stop_file_watcher));
}

TEST_F(FileWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  std::shared_ptr<std::atomic<bool>> stop_file_watcher =
      std::make_shared<std::atomic<bool>>(false);

  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  std::stringstream ss;
  std::streambuf *old_cout = std::cout.rdbuf(ss.rdbuf());

  std::thread th(&FileWatchdog::run, &watchdog);
  std::this_thread::sleep_for(std::chrono::seconds(2));

  *stop_file_watcher = true;
  th.join();

  std::cout.rdbuf(old_cout);
  std::string output = ss.str();

  // Assert that the watchdog has run
  ASSERT_NE(output.find("FileWatchdog running"), std::string::npos);
}

TEST_F(FileWatchdogTest, TestStartFileWatcherProcess) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher =
      std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");

  watchdog.start_file_watcher_process(0);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.start_file_watcher_process(0);

  // Test if the file watcher process is not started again and still running

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.start_file_watcher_process(1);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 2);
}

TEST_F(FileWatchdogTest, TestStopFileWatcherProcess) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher =
      std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection *connection = new StorageDatabaseConnection(config);

  connection->add_dataset(
      "test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.start_file_watcher_process(0);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.stop_file_watcher_process(0);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);
}

TEST_F(FileWatchdogTest, Test) {
  std::shared_ptr<std::atomic<bool>> stop_file_watcher =
      std::make_shared<std::atomic<bool>>(false);
  FileWatchdog watchdog("config.yaml", stop_file_watcher);

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection *connection = new StorageDatabaseConnection(config);

  soci::session *sql = connection->get_session();

  connection->add_dataset(
      "test_dataset1", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);

  connection->add_dataset(
      "test_dataset2", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
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