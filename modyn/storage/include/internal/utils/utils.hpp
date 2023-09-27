#pragma once

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#define FAIL(msg)                                                                                          \
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

static std::shared_ptr<FilesystemWrapper> get_filesystem_wrapper(const std::string& path,
                                                                 const FilesystemWrapperType& type) {
  std::shared_ptr<FilesystemWrapper> filesystem_wrapper;
  if (type == FilesystemWrapperType::LOCAL) {
    filesystem_wrapper = std::make_shared<LocalFilesystemWrapper>(path);
  } else {
    FAIL("Unknown filesystem wrapper type");
  }
  return filesystem_wrapper;
}

static std::unique_ptr<FileWrapper> get_file_wrapper(const std::string& path, const FileWrapperType& type,
                                                     const YAML::Node& file_wrapper_config,
                                                     const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
  assert(filesystem_wrapper != nullptr);
  assert(!path.empty());
  assert(filesystem_wrapper->exists(path));

  std::unique_ptr<FileWrapper> file_wrapper;
  if (type == FileWrapperType::BINARY) {
    file_wrapper = std::make_unique<BinaryFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == FileWrapperType::SINGLE_SAMPLE) {
    file_wrapper = std::make_unique<SingleSampleFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else {
    FAIL("Unknown file wrapper type");
  }
  return file_wrapper;
}

}  // namespace storage::utils
