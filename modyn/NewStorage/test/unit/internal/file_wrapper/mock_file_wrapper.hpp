#pragma once

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

#include "gmock/gmock.h"
#include "internal/file_wrapper/AbstractFileWrapper.hpp"

namespace storage {
class MockFileWrapper : public AbstractFileWrapper {
 public:
  MockFileWrapper() : AbstractFileWrapper("", YAML::Node(), nullptr){};
  MOCK_METHOD(int, get_number_of_samples, (), (override));
  MOCK_METHOD(std::vector<std::vector<unsigned char>>*, get_samples, (int start, int end), (override));
  MOCK_METHOD(int, get_label, (int index), (override));
  MOCK_METHOD(std::vector<int>*, get_all_labels, (), (override));
  MOCK_METHOD(std::vector<unsigned char>*, get_sample, (int index), (override));
  MOCK_METHOD(std::vector<std::vector<unsigned char>>*, get_samples_from_indices, (std::vector<int> * indices),
              (override));
  MOCK_METHOD(std::string, get_name, (), (override));
  MOCK_METHOD(void, validate_file_extension, (), (override));
  ~MockFileWrapper() {}
}
}  // namespace storage
