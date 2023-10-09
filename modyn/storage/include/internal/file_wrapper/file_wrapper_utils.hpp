#pragma once

#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "internal/utils/utils.hpp"

namespace storage::file_wrapper {

static std::unique_ptr<storage::file_wrapper::FileWrapper> get_file_wrapper(const std::string& path, const storage::file_wrapper::FileWrapperType& type,
                                                     const YAML::Node& file_wrapper_config,
                                                     const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
  ASSERT(filesystem_wrapper != nullptr, "Filesystem wrapper is nullptr");
  ASSERT(!path.empty(), "Path is empty");
  ASSERT(filesystem_wrapper->exists(path), "Path does not exist");

  std::unique_ptr<storage::file_wrapper::FileWrapper> file_wrapper;
  if (type == storage::file_wrapper::FileWrapperType::BINARY) {
    file_wrapper = std::make_unique<storage::file_wrapper::BinaryFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == storage::file_wrapper::FileWrapperType::SINGLE_SAMPLE) {
    file_wrapper = std::make_unique<storage::file_wrapper::SingleSampleFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else {
    FAIL("Unknown file wrapper type");
  }
  return file_wrapper;
}
} // namespace storage::file_wrapper