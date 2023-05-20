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

namespace storage {

class Utils {
 public:
  static std::shared_ptr<FilesystemWrapper> get_filesystem_wrapper(const std::string& path,
                                                                   const FilesystemWrapperType& type) {
    std::shared_ptr<FilesystemWrapper> filesystem_wrapper;
    if (type == FilesystemWrapperType::LOCAL) {
      filesystem_wrapper = std::make_shared<LocalFilesystemWrapper>(path);
    } else {
      throw std::runtime_error("Unknown filesystem wrapper type");
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
      throw std::runtime_error("Unknown file wrapper type");
    }
    return file_wrapper;
  }
  static std::string join_string_list(const std::vector<std::string>& list, const std::string& delimiter) {
    std::string result;
    for (uint32_t i = 0; i < list.size(); i++) {
      result += list[i];
      if (i < list.size() - 1) {
        result += delimiter;
      }
    }
    return result;
  }
  static std::string get_tmp_filename(const std::string& base_name) {
    const int16_t max_num = 10000;
    const int16_t digits = 8;
    const std::string filename;
    std::random_device rd;  // NOLINT
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int16_t> dist(0, max_num);
    const int16_t random_number = dist(mt);
    std::string random_number_string = std::to_string(random_number);
    while (random_number_string.length() < digits) {
      random_number_string += "0";
    }
    return base_name + random_number_string + ".tmp";
  }
};
}  // namespace storage
