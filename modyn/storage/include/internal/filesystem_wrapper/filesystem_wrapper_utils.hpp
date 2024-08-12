#pragma once

#include "internal/filesystem_wrapper/filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

std::shared_ptr<FilesystemWrapper> get_filesystem_wrapper(const FilesystemWrapperType& type);

}  // namespace modyn::storage
