#ifndef MOCK_UTILS_HPP
#define MOCK_UTILS_HPP

#include "../../../src/internal/utils/Utils.hpp"
#include "gmock/gmock.h"
#include <gtest/gtest.h>

namespace storage {
class MockUtils : public storage::Utils {
public:
  MockUtils() : Utils(){};
  MOCK_METHOD(AbstractFilesystemWrapper *, get_filesystem_wrapper, (),
              (override));
  MOCK_METHOD(AbstractFileWrapper *, get_file_wrapper,
              (std::string path, YAML::Node file_wrapper_config,
               AbstractFilesystemWrapper *filesystem_wrapper),
              (override));
  MOCK_METHOD(std::string, join_string_list,
              (std::vector<std::string> list, std::string delimiter),
              (override));
  MOCK_METHOD(std::string, get_tmp_filename, (std::string base_name),
              (override));
};
} // namespace storage

#endif