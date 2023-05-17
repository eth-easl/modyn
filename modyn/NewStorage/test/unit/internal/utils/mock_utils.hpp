#pragma once

#include <gtest/gtest.h>

#include "gmock/gmock.h"
#include "internal/utils/Utils.hpp"

namespace storage {
class MockUtils : public storage::Utils {
 public:
  MockUtils() : Utils(){};
  MOCK_METHOD(std::unique_ptr<FilesystemWrapper>, get_filesystem_wrapper, (), (override));
  MOCK_METHOD(std::unique_ptr<FileWrapper>, get_file_wrapper,
              (std::string path, YAML::Node file_wrapper_config, std::unique_ptr<FilesystemWrapper> filesystem_wrapper),
              (override));
  MOCK_METHOD(std::string, join_string_list, (std::vector<std::string> list, std::string delimiter), (override));
  MOCK_METHOD(std::string, get_tmp_filename, (std::string base_name), (override));
};
}  // namespace storage
