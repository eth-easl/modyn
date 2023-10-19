#pragma once

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>


#define FAIL(msg)                                                                                        \
  throw modyn::utils::ModynException("ERROR at " __FILE__ ":" + std::to_string(__LINE__) + " " + (msg) + \
                                     "\nExecution failed.")

#define ASSERT(expr, msg)         \
  if (!static_cast<bool>(expr)) { \
    FAIL((msg));                  \
  }                               \
  static_assert(true, "End call of macro with a semicolon")

namespace modyn::utils {

bool is_power_of_two(uint64_t value);

class ModynException : public std::exception {
 public:
  // If we make msg a const-reference, clang-tidy asks us to move instead...
  explicit ModynException(std::string msg) : msg_{std::move(msg)} {} 
  const char* what() const noexcept override { return msg_.c_str(); }

 private:
  const std::string msg_;
};

}  // namespace modyn::utils
