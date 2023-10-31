#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>

#include <iostream>

#include "internal/database/storage_database_connection.hpp"

namespace modyn::storage {

struct SampleRecord {
  int64_t id;
  int64_t label;
  int64_t index;
};

class CursorHandler {
 public:
  CursorHandler(soci::session& session, DatabaseDriver driver, const std::string& query, const std::string& cursorName,
                int16_t number_of_columns = 3)
      : driver_{driver},
        session_{session},
        query_{query},
        cursorName_{cursorName},
        number_of_columns_{number_of_columns} {
    rs_ = nullptr;
    postgresql_conn_ = nullptr;
    switch (driver_) {
      case DatabaseDriver::POSTGRESQL: {
        auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session_.get_backend());
        PGconn* conn = postgresql_session_backend->conn_;

        std::string declareCursor = fmt::format("DECLARE {} CURSOR FOR {}", cursorName, query);
        PGresult* result = PQexec(conn, declareCursor.c_str());

        if (PQresultStatus(result) != PGRES_COMMAND_OK) {
          SPDLOG_ERROR("Cursor declaration failed: {}", PQerrorMessage(conn));
          PQclear(result);
          break;
        }

        PQclear(result);

        postgresql_conn_ = conn;
        break;
      }
      case DatabaseDriver::SQLITE3: {
        rs_ = new soci::rowset<soci::row>((session_.prepare << query));
        break;
      }
      default:
        FAIL("Unsupported database driver");
    }
  }
  ~CursorHandler() { close_cursor(); }
  CursorHandler(const CursorHandler&) = delete;
  CursorHandler& operator=(const CursorHandler&) = delete;
  CursorHandler(CursorHandler&&) = delete;
  CursorHandler& operator=(CursorHandler&&) = delete;
  std::vector<SampleRecord> yield_per(int64_t number_of_rows_to_fetch);
  void close_cursor();

 private:
  DatabaseDriver driver_;
  soci::session& session_;
  std::string query_;
  std::string cursorName_;
  int16_t number_of_columns_;
  soci::rowset<soci::row>* rs_;
  PGconn* postgresql_conn_;
};
}  // namespace modyn::storage