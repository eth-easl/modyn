#ifndef UTILS_HPP
#define UTILS_HPP

#include "../file_wrapper/AbstractFileWrapper.hpp"
#include "../file_wrapper/BinaryFileWrapper.hpp"
#include "../file_wrapper/SingleSampleFileWrapper.hpp"
#include "../filesystem_wrapper/AbstractFilesystemWrapper.hpp"
#include "../filesystem_wrapper/LocalFilesystemWrapper.hpp"

namespace storage {

class Utils {
public:
  static AbstractFilesystemWrapper *get_filesystem_wrapper(std::string path,
                                                           std::string type) {
    if (type == "LOCAL") {
      return new LocalFilesystemWrapper(path);
    } else {
      throw std::runtime_error("Unknown filesystem wrapper type");
    }
  }
  static AbstractFileWrapper *
  get_file_wrapper(std::string path, std::string type, YAML::Node config,
                   AbstractFilesystemWrapper *filesystem_wrapper) {
    if (type == "BIN") {
      return new BinaryFileWrapper(path, config, filesystem_wrapper);
    } else if (type == "SINGLE_SAMPLE") {
      return new SingleSampleFileWrapper(path, config, filesystem_wrapper);
    } else {
      throw std::runtime_error("Unknown file wrapper type");
    }
  }
  static std::string joinStringList(std::vector<std::string> list,
                                    std::string delimiter) {
    std::string result = "";
    for (int i = 0; i < list.size(); i++) {
      result += list[i];
      if (i < list.size() - 1) {
        result += delimiter;
      }
    }
    return result;
  }
};
} // namespace storage

#endif