#pragma once

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <string>

#include "internal/file_wrapper/abstract_file_wrapper.hpp"
#include "internal/file_wrapper/binary_file_wrapper.hpp"
#include "internal/file_wrapper/single_sample_file_wrapper.hpp"
#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"
#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

namespace storage {

class Utils {
 public:
  static AbstractFilesystemWrapper* get_filesystem_wrapper(std::string path, std::string type) {
    if (type == "LOCAL") {
      return new LocalFilesystemWrapper(path);
    } else {
      throw std::runtime_error("Unknown filesystem wrapper type");
    }
  }
  static AbstractFileWrapper* get_file_wrapper(std::string path, std::string type, YAML::Node file_wrapper_config,
                                               AbstractFilesystemWrapper* filesystem_wrapper) {
    if (type == "BIN") {
      return new BinaryFileWrapper(path, file_wrapper_config, filesystem_wrapper);
    } else if (type == "SINGLE_SAMPLE") {
      return new SingleSampleFileWrapper(path, file_wrapper_config, filesystem_wrapper);
    } else {
      throw std::runtime_error("Unknown file wrapper type");
    }
  }
  static std::string join_string_list(std::vector<std::string> list, std::string delimiter) {
    std::string result = "";
    for (unsigned long i = 0; i < list.size(); i++) {
      result += list[i];
      if (i < list.size() - 1) {
        result += delimiter;
      }
    }
    return result;
  }
  static std::string get_tmp_filename(std::string base_name) {
    std::srand(std::time(NULL));
    const int MAX_NUM = 10000;
    const int DIGITS = 8;
    std::string filename;
    int randomNumber = std::rand() % MAX_NUM;
    std::string randomNumberString = std::to_string(randomNumber);
    while (randomNumberString.length() < DIGITS) {
      randomNumberString = "0" + randomNumberString;
    }
    filename = base_name + randomNumberString + ".tmp";
    return filename;
  }
};
}  // namespace storage
