#include "../../../src/internal/file_watcher/FileWatchdog.hpp"
#include "../../TestUtils.hpp"
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

using namespace storage;
namespace bp = boost::process;

class FileWatchdogTest : public ::testing::Test {
protected:
  void SetUp() override { TestUtils::create_dummy_yaml(); }

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
  bp::child subprocess(bp::search_path(exec), bp::args({"config.yaml"}),
                       bp::std_out > is);
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

// TODO: Figure out how to test the file watcher (60)