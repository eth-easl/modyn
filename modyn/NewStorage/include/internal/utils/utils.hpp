#pragma once

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"
#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

namespace storage {

class Utils {
 public:
  static AbstractFilesystemWrapper* get_filesystem_wrapper(const std::string& path, const std::string& type) {
    AbstractFilesystemWrapper* abstract_filesystem_wrapper;
    if (type == "LOCAL") {
      abstract_filesystem_wrapper=  new LocalFilesystemWrapper(path);
    } else {
      throw std::runtime_error("Unknown filesystem wrapper type: " + type);
    }
    return abstract_filesystem_wrapper;
  }
  static AbstractFileWrapper* get_file_wrapper(const std::string& path, const std::string& type,
                                               const YAML::Node& file_wrapper_config,
                                               AbstractFilesystemWrapper* filesystem_wrapper) {
    AbstractFileWrapper* file_wrapper;
    if (type == "BIN") {
      file_wrapper = new BinaryFileWrapper(path, file_wrapper_config, filesystem_wrapper);
    } else if (type == "SINGLE_SAMPLE") {
      file_wrapper = new SingleSampleFileWrapper(path, file_wrapper_config, filesystem_wrapper);
    } else {
      throw std::runtime_error("Unknown file wrapper type: " + type);
    }
    return file_wrapper;
  }
  static std::string join_string_list(std::vector<std::string> list, const std::string& delimiter) {
    std::string result;
    for (uint64_t i = 0; i < list.size(); i++) {
      result += list[i];
      if (i < list.size() - 1) {
        result += delimiter;
      }
    }
    return result;
  }
  static std::string get_tmp_filename(const std::string& base_name) {
    const int max_num = 10000;
    const int digits = 8;
    std::string filename;
    std::random_device rd;  // NOLINT
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, max_num);
    const int random_number = dist(mt);
    std::string random_number_string = std::to_string(random_number);
    while (random_number_string.length() < digits) {
      random_number_string += "0";
    }
    filename = base_name + random_number_string + ".tmp";
    return filename;
  }
};
}  // namespace storage
