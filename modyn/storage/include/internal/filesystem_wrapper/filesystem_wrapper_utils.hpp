#pragma once

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

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
}  // namespace modyn::storage