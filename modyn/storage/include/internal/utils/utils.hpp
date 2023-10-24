#pragma once

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#define FAIL(msg)                                                                                          \
  SPDLOG_ERROR(msg);                                                                                       \
  throw storage::utils::ModynException("ERROR at " __FILE__ ":" + std::to_string(__LINE__) + " " + (msg) + \
                                       "\nExecution failed.")

#define ASSERT(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    FAIL((msg));                  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

namespace storage::utils {

class ModynException : public std::exception {
 public:
  explicit ModynException(std::string msg) : msg_{std::move(msg)} {}
  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  const std::string msg_;
};

}  // namespace storage::utils
