#include "../../../src/internal/file_watcher/FileWatchdog.hpp"
#include "../../TestUtils.hpp"
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

using namespace storage;
namespace bp = boost::process;

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
  ASSERT_NO_THROW(FileWatchdog watchdog("config.yaml"));
}

TEST_F(FileWatchdogTest, TestRun) {
  // Collect the output of the watchdog
  bp::ipstream is;
  std::string exec = std::filesystem::current_path() / "executables" /
                     "FileWatchdog" / "FileWatchdog";

  bp::child subprocess(exec, bp::args({"config.yaml"}), bp::std_out > is);
  subprocess.wait_for(std::chrono::seconds(1));
  subprocess.terminate();

  std::string line;
  std::string output;
  while (std::getline(is, line)) {
    output += line;
  }

  // Assert that the watchdog has run
  ASSERT_NE(output.find("FileWatchdog running"), std::string::npos);
}

TEST_F(FileWatchdogTest, TestStartFileWatcherProcess) {
  FileWatchdog watchdog("config.yaml");

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection *connection = new StorageDatabaseConnection(config);

  soci::session *sql = connection->get_session();

  connection->add_dataset(
      "test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(connection);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  watchdog.watch_file_watcher_processes(connection);

  // Test if the file watcher process is not started again and still running

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  connection->add_dataset(
      "test_dataset2", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);
  
  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 2);

  *sql << "DELETE FROM datasets WHERE name = 'test_dataset'";

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);
}

TEST_F(FileWatchdogTest, TestStopFileWatcherProcess) {
  FileWatchdog watchdog("config.yaml");

  YAML::Node config = YAML::LoadFile("config.yaml");
  StorageDatabaseConnection *connection = new StorageDatabaseConnection(config);

  soci::session *sql = connection->get_session();

  connection->add_dataset(
      "test_dataset", "tmp", "LOCAL", "MOCK", "test description", "0.0.0",
      TestUtils::get_dummy_file_wrapper_config_inline(), true);

  watchdog.watch_file_watcher_processes(connection);

  std::vector<long long> file_watcher_processes;
  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 1);

  *sql << "DELETE FROM datasets WHERE name = 'test_dataset'";

  watchdog.watch_file_watcher_processes(connection);

  file_watcher_processes = watchdog.get_running_file_watcher_processes();

  ASSERT_EQ(file_watcher_processes.size(), 0);
}
