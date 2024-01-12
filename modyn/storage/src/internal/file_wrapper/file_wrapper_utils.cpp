#include "internal/file_wrapper/file_wrapper_utils.hpp"

#include <memory>
#include <string>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"

namespace modyn::storage {

std::unique_ptr<FileWrapper> get_file_wrapper(const std::string& path, const FileWrapperType& type,
                                              const YAML::Node& file_wrapper_config,
                                              const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper) {
  ASSERT(filesystem_wrapper != nullptr, "Filesystem wrapper is nullptr");
  ASSERT(!path.empty(), "Path is empty");
  ASSERT(filesystem_wrapper->exists(path), fmt::format("Path {} does not exist", path));

  std::unique_ptr<FileWrapper> file_wrapper;
  if (type == FileWrapperType::BINARY) {
    file_wrapper = std::make_unique<BinaryFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == FileWrapperType::SINGLE_SAMPLE) {
    file_wrapper = std::make_unique<SingleSampleFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == FileWrapperType::CSV) {
    file_wrapper = std::make_unique<CsvFileWrapper>(path, file_wrapper_config, filesystem_wrapper);
  } else if (type == FileWrapperType::INVALID_FW) {
    FAIL(fmt::format("Trying to instantiate INVALID FileWrapper at path {}", path));
  } else {
    FAIL(fmt::format("Unknown file wrapper type {}", static_cast<int64_t>(type)));
  }
  return file_wrapper;
}

}  // namespace modyn::storage