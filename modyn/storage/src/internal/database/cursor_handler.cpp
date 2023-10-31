#include "internal/database/cursor_handler.hpp"

#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <soci/soci.h>
#include <spdlog/spdlog.h>

using namespace modyn::storage;

std::vector<SampleRecord> CursorHandler::yield_per(const int64_t number_of_rows_to_fetch) {
  std::vector<SampleRecord> records(number_of_rows_to_fetch);

  switch (driver_) {
    case DatabaseDriver::POSTGRESQL: {
      if (postgresql_conn_ == nullptr) {
        FAIL("Cursor not initialized");
      }
      std::string fetchQuery = fmt::format("FETCH {} FROM {}", number_of_rows_to_fetch, cursorName_);

      PGresult* result = PQexec(postgresql_conn_, fetchQuery.c_str());

      if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        PQclear(result);
        FAIL("Cursor fetch failed");
        return records;
      }

      int64_t rows = PQntuples(result);

      for (int64_t i = 0; i < rows; i++) {
        SampleRecord record;
        record.id = std::stoll(PQgetvalue(result, i, 0));
        if (number_of_columns_ > 1) {
          record.label = std::stoll(PQgetvalue(result, i, 1));
        }
        if (number_of_columns_ == 3) {
          record.index = std::stoll(PQgetvalue(result, i, 2));
        }
        records[i] = record;
      }

      PQclear(result);
      return records;
      break;
    }
    case DatabaseDriver::SQLITE3: {
      if (rs_ == nullptr) {
        FAIL("Cursor not initialized");
      }
      int64_t retrieved_rows = 0;
      for (auto& row : *rs_) {
        if (retrieved_rows >= number_of_rows_to_fetch) {
          break;
        }
        SampleRecord record;
        record.id = row.get<int64_t>(0);
        if (number_of_columns_ > 1) {
          record.label = row.get<int64_t>(1);
        }
        if (number_of_columns_ == 3) {
          record.index = row.get<int64_t>(2);
        }
        records[retrieved_rows] = record;
        retrieved_rows++;
      }
      return records;
      break;
    }
    default:
      FAIL("Unsupported database driver");
  }
}

void CursorHandler::close_cursor() {
  switch (driver_) {
    case DatabaseDriver::POSTGRESQL: {
      auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session_.get_backend());
      PGconn* conn = postgresql_session_backend->conn_;

      std::string closeQuery = "CLOSE " + cursorName_;
      PGresult* result = PQexec(conn, closeQuery.c_str());

      if (PQresultStatus(result) != PGRES_COMMAND_OK) {
        std::cerr << "Cursor closure failed: " << PQerrorMessage(conn) << std::endl;
      }

      PQclear(result);
      break;
    }
    case DatabaseDriver::SQLITE3:
      break;
    default:
      FAIL("Unsupported database driver");
  }
}