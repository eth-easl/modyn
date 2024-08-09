#pragma once

#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <soci/soci.h>

#include <iostream>

#include "internal/database/storage_database_connection.hpp"

namespace modyn::storage {

struct SampleRecord {
  int64_t id;
  int64_t column_1;
  int64_t column_2;
};

/*
Implements a server-side cursor on Postgres and emulates it for sqlite.
For a given query, results are returned (using the yield_per function) buffered, to avoid filling up memory.
*/
class CursorHandler {
 public:
  CursorHandler(soci::session& session, DatabaseDriver driver, const std::string& query, std::string cursor_name,
                int16_t number_of_columns = 3)
      : driver_{driver},
        session_{session},
        query_{query},
        cursor_name_{std::move(cursor_name)},
        number_of_columns_{number_of_columns} {
    // ncol = 0 or = 1 means that we only return the first column in the result of the query (typically, the ID)
    // ncol = 2 returns the second column as well (typically what you want if you want an id + some property)
    // ncol = 3 returns the third as well
    // This could be generalized but currently is hardcoded.
    // A SampleRecord is populated and (as can be seen above) only has three properties per row.
    ASSERT(number_of_columns <= 3 && number_of_columns >= 0, "We currently only support 0 - 3 columns.");

    switch (driver_) {
      case DatabaseDriver::POSTGRESQL: {
        auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session_.get_backend());
        PGconn* conn = postgresql_session_backend->conn_;

        const std::string declare_cursor = fmt::format("DECLARE {} CURSOR WITH HOLD FOR {}", cursor_name_, query);
        PGresult* result = PQexec(conn, declare_cursor.c_str());

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
        rs_ = std::make_unique<soci::rowset<soci::row>>(session_.prepare << query);
        break;
      }
      default:
        FAIL("Unsupported database driver");
    }

    open_ = true;
  }
  ~CursorHandler() { close_cursor(); }
  CursorHandler(const CursorHandler&) = delete;
  CursorHandler& operator=(const CursorHandler&) = delete;
  CursorHandler(CursorHandler&&) = delete;
  CursorHandler& operator=(CursorHandler&&) = delete;
  std::vector<SampleRecord> yield_per(uint64_t number_of_rows_to_fetch);
  void close_cursor();

 private:
  void check_cursor_initialized();
  DatabaseDriver driver_;
  soci::session& session_;
  std::string query_;
  std::string cursor_name_;
  int16_t number_of_columns_;
  std::unique_ptr<soci::rowset<soci::row>> rs_{nullptr};
  PGconn* postgresql_conn_{nullptr};
  bool open_{false};
};
}  // namespace modyn::storage
