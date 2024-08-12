#include "internal/database/cursor_handler.hpp"

#include <gtest/gtest.h>
#include <soci/soci.h>

#include "test_utils.hpp"

using namespace modyn::storage;

class CursorHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    modyn::test::TestUtils::create_dummy_yaml();
    const YAML::Node config = YAML::LoadFile("config.yaml");
    const StorageDatabaseConnection connection(config);
    connection.create_tables();

    soci::session session = connection.get_session();

    for (int64_t i = 0; i < 1000; i++) {
      session << "INSERT INTO samples (dataset_id, file_id, sample_index, label) VALUES (1, :file_id, :sample_index, "
                 ":label)",
          soci::use(i, "file_id"), soci::use(i, "sample_index"), soci::use(i, "label");
    }
  }
  void TearDown() override {
    if (std::filesystem::exists("test.db")) {
      std::filesystem::remove("test.db");
    }
  }
};

TEST_F(CursorHandlerTest, TestCheckCursorInitialized) {
  const YAML::Node config = YAML::LoadFile("config.yaml");
  const StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  CursorHandler cursor_handler(session, connection.get_drivername(), "SELECT * FROM samples", "test_cursor");

  ASSERT_NO_THROW(cursor_handler.close_cursor());
}

TEST_F(CursorHandlerTest, TestYieldPerSQLite3ThreeColumns) {  // NOLINT (readability-function-cognitive-complexity)
  const YAML::Node config = YAML::LoadFile("config.yaml");
  const StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  CursorHandler cursor_handler(session, connection.get_drivername(),
                               "SELECT sample_id, label, sample_index FROM samples", "test_cursor");

  std::vector<SampleRecord> record(100);
  for (int64_t i = 0; i < 10; i++) {
    ASSERT_NO_THROW(record = cursor_handler.yield_per(100));
    ASSERT_EQ(record.size(), 100);
    for (int64_t j = 0; j < 100; j++) {
      ASSERT_EQ(record[j].id, j + i * 100 + 1);
      ASSERT_EQ(record[j].column_1, j + i * 100);
      ASSERT_EQ(record[j].column_2, j + i * 100);
    }
  }
  cursor_handler.close_cursor();
}

TEST_F(CursorHandlerTest, TestYieldPerSQLite3TwoColumns) {  // NOLINT (readability-function-cognitive-complexity)
  const YAML::Node config = YAML::LoadFile("config.yaml");
  const StorageDatabaseConnection connection(config);
  soci::session session = connection.get_session();

  CursorHandler cursor_handler(session, connection.get_drivername(), "SELECT sample_id, label FROM samples",
                               "test_cursor", 2);

  std::vector<SampleRecord> record(100);
  for (int64_t i = 0; i < 10; i++) {
    ASSERT_NO_THROW(record = cursor_handler.yield_per(100));
    ASSERT_EQ(record.size(), 100);
    for (int64_t j = 0; j < 100; j++) {
      ASSERT_EQ(record[j].id, j + i * 100 + 1);
      ASSERT_EQ(record[j].column_1, j + i * 100);
    }
  }
  cursor_handler.close_cursor();
}
