#pragma once

#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/csv_file_wrapper.hpp"
#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

std::unique_ptr<FileWrapper> get_file_wrapper(const std::string& path, const FileWrapperType& type,
                                              const YAML::Node& file_wrapper_config,
                                              const std::shared_ptr<FilesystemWrapper>& filesystem_wrapper);

}  // namespace modyn::storage
