#include "internal/database/cursor_handler.hpp"

#include <fmt/format.h>
#include <libpq/libpq-fs.h>
#include <soci/soci.h>

using namespace modyn::storage;

std::vector<SampleRecord> CursorHandler::yield_per(const int64_t number_of_rows_to_fetch) {
  std::vector<SampleRecord> records;
  check_cursor_initialized();

  switch (driver_) {
    case DatabaseDriver::POSTGRESQL: {
      const std::string fetch_query = fmt::format("FETCH {} FROM {}", number_of_rows_to_fetch, cursor_name_);

      PGresult* result = PQexec(postgresql_conn_, fetch_query.c_str());

      if (PQresultStatus(result) != PGRES_TUPLES_OK) {
        SPDLOG_ERROR("Cursor fetch failed: {}", PQerrorMessage(postgresql_conn_));
        PQclear(result);
        return records;
      }

      const uint64_t rows = static_cast<uint64_t>(PQntuples(result));
      records.resize(rows);

      for (uint64_t i = 0; i < rows; ++i) {
        SampleRecord record{};
        record.id = std::stoll(PQgetvalue(result, i, 0));
        if (number_of_columns_ > 1) {
          record.column_1 = std::stoll(PQgetvalue(result, i, 1));
        }
        if (number_of_columns_ == 3) {
          record.column_2 = std::stoll(PQgetvalue(result, i, 2));
        }

        records[i] = record;
      }

      PQclear(result);
      return records;
      break;
    }
    case DatabaseDriver::SQLITE3: {
      int64_t retrieved_rows = 0;
      for (auto& row : *rs_) {
        SampleRecord record{};
        record.id = StorageDatabaseConnection::get_from_row<int64_t>(row, 0);
        if (number_of_columns_ > 1) {
          record.column_1 = StorageDatabaseConnection::get_from_row<int64_t>(row, 1);
        }
        if (number_of_columns_ == 3) {
          record.column_2 = StorageDatabaseConnection::get_from_row<int64_t>(row, 2);
        }
        records[retrieved_rows] = record;
        retrieved_rows++;
        if (retrieved_rows >= number_of_rows_to_fetch) {
          break;
        }
      }
      return records;
      break;
    }
    default:
      FAIL("Unsupported database driver");
  }
}

void CursorHandler::check_cursor_initialized() {
  if (rs_ == nullptr && postgresql_conn_ == nullptr) {
    SPDLOG_ERROR("Cursor not initialized");
    throw std::runtime_error("Cursor not initialized");
  }
}

void CursorHandler::close_cursor() {
  switch (driver_) {
    case DatabaseDriver::POSTGRESQL: {
      auto* postgresql_session_backend = static_cast<soci::postgresql_session_backend*>(session_.get_backend());
      PGconn* conn = postgresql_session_backend->conn_;

      const std::string close_query = "CLOSE " + cursor_name_;
      PGresult* result = PQexec(conn, close_query.c_str());

      if (PQresultStatus(result) != PGRES_COMMAND_OK) {
        SPDLOG_ERROR(fmt::format("Cursor close failed: {}", PQerrorMessage(conn)));
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