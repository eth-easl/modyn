#pragma once

#include <gtest/gtest.h>

#include <fstream>

#include "gmock/gmock.h"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "storage_test_utils.hpp"

namespace storage::test {
class MockFilesystemWrapper : public storage::filesystem_wrapper::FilesystemWrapper {
 public:
  MockFilesystemWrapper() : storage::filesystem_wrapper::FilesystemWrapper("") {}  // NOLINT
  MOCK_METHOD(std::vector<unsigned char>, get, (const std::string& path), (override));
  MOCK_METHOD(bool, exists, (const std::string& path), (override));
  MOCK_METHOD(std::vector<std::string>, list, (const std::string& path, bool recursive), (override));
  MOCK_METHOD(bool, is_directory, (const std::string& path), (override));
  MOCK_METHOD(bool, is_file, (const std::string& path), (override));
  MOCK_METHOD(int64_t, get_file_size, (const std::string& path), (override));
  MOCK_METHOD(int64_t, get_modified_time, (const std::string& path), (override));
  MOCK_METHOD(bool, is_valid_path, (const std::string& path), (override));
  MOCK_METHOD(std::ifstream&, get_stream, (const std::string& path), (override));
  MOCK_METHOD(storage::filesystem_wrapper::FilesystemWrapperType, get_type, (), (override));
  MOCK_METHOD(bool, remove, (const std::string& path), (override));
  ~MockFilesystemWrapper() override = default;
  MockFilesystemWrapper(const MockFilesystemWrapper&) = delete;
  MockFilesystemWrapper& operator=(const MockFilesystemWrapper&) = delete;
  MockFilesystemWrapper(MockFilesystemWrapper&&) = delete;
  MockFilesystemWrapper& operator=(MockFilesystemWrapper&&) = delete;
};
}  // namespace storage::test
