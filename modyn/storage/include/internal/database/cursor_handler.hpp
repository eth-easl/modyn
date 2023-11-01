#pragma once

#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <soci/soci.h>

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
  CursorHandler(soci::session& session, DatabaseDriver driver, const std::string& query, std::string cursor_name,
                int16_t number_of_columns = 3)
      : driver_{driver},
        session_{session},
        query_{query},
        cursor_name_{std::move(cursor_name)},
        number_of_columns_{number_of_columns} {
    switch (driver_) {
      case DatabaseDriver::POSTGRESQL: {
        auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session_.get_backend());
        PGconn* conn = postgresql_session_backend->conn_;

        const std::string declare_cursor = fmt::format("DECLARE {} CURSOR FOR {}", cursor_name_, query);
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
  }
  ~CursorHandler() { close_cursor(); }
  CursorHandler(const CursorHandler&) = delete;
  CursorHandler& operator=(const CursorHandler&) = delete;
  CursorHandler(CursorHandler&&) = delete;
  CursorHandler& operator=(CursorHandler&&) = delete;
  std::vector<SampleRecord> yield_per(int64_t number_of_rows_to_fetch);
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
};
}  // namespace modyn::storage