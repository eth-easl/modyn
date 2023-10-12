#pragma once

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "gmock/gmock.h"
#include "internal/file_wrapper/FileWrapper.hpp"

namespace storage::test {
class MockFileWrapper : public storage::file_wrapper::FileWrapper {
 public:
  MockFileWrapper(const std::string& path, const YAML::Node& fw_config, std::shared_ptr<FilesystemWrapper>& fs_wrapper)
      : storage::file_wrapper::FileWrapper(path, fw_config, fs_wrapper) {}
  MOCK_METHOD(int64_t, get_number_of_samples, (), (override));
  MOCK_METHOD(std::vector<std::vector<unsigned char>>*, get_samples, (int64_t start, int64_t end), (override));
  MOCK_METHOD(int64_t, get_label, (int64_t index), (override));
  MOCK_METHOD(std::vector<int32_t>*, get_all_labels, (), (override));
  MOCK_METHOD(std::vector<unsigned char>*, get_sample, (int64_t index), (override));
  MOCK_METHOD(std::vector<std::vector<unsigned char>>*, get_samples_from_indices, (std::vector<int64_t> * indices),
              (override));
  MOCK_METHOD(storage::file_wrapper::FileWrapperType, get_type, (), (override));
  MOCK_METHOD(void, validate_file_extension, (), (override));
  MOCK_METHOD(void, delete_samples, (std::vector<int64_t> * indices), (override));
  MOCK_METHOD(void, set_file_path, (const std::string& path), (override));
  ~MockFileWrapper() override = default;
  MockFileWrapper(const MockFileWrapper& other) : FileWrapper(other) {}
}
}  // namespace storage::test
