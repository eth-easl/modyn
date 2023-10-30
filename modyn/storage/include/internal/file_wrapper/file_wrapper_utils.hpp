#pragma once

#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

static std::unique_ptr<FileWrapper> get_file_wrapper(
    const std::string& path, const FileWrapperType& type, const YAML::Node& file_wrapper_config,
    const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
  ASSERT(filesystem_wrapper != nullptr, "Filesystem wrapper is nullptr");
  ASSERT(!path.empty(), "Path is empty");
  ASSERT(filesystem_wrapper->exists(path), "Path does not exist");

  std::unique_ptr<FileWrapper> file_wrapper;
  if (type == FileWrapperType::BINARY) {
    file_wrapper =
        std::make_unique<BinaryFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == FileWrapperType::SINGLE_SAMPLE) {
    file_wrapper =
        std::make_unique<SingleSampleFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else {
    FAIL("Unknown file wrapper type");
  }
  return file_wrapper;
}
}  // namespace modyn::storage