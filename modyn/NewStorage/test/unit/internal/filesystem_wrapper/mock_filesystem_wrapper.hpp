#pragma once

#include <gtest/gtest.h>

#include <fstream>

#include "gmock/gmock.h"
#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {
class MockFilesystemWrapper : public storage::AbstractFilesystemWrapper {
 public:
  MockFilesystemWrapper() : AbstractFilesystemWrapper("") {}  // NOLINT
  MOCK_METHOD(std::vector<unsigned char>, get, (const std::string& path), (override));
  MOCK_METHOD(bool, exists, (const std::string& path), (override));
  MOCK_METHOD(std::vector<std::string>, list, (const std::string& path, bool recursive), (override));
  MOCK_METHOD(bool, is_directory, (const std::string& path), (override));
  MOCK_METHOD(bool, is_file, (const std::string& path), (override));
  MOCK_METHOD(int64_t, get_file_size, (const std::string& path), (override));
  MOCK_METHOD(int64_t, get_modified_time, (const std::string& path), (override));
  MOCK_METHOD(int64_t, get_created_time, (const std::string& path), (override));
  MOCK_METHOD(std::string, join, (const std::vector<std::string>& paths), (override));
  MOCK_METHOD(bool, is_valid_path, (const std::string& path), (override));
  MOCK_METHOD(std::string, get_name, (), (override));
  ~MockFilesystemWrapper() override = default;
  MockFilesystemWrapper(const MockFilesystemWrapper& other) : AbstractFilesystemWrapper(other.base_path_) {}
};
}  // namespace storage
