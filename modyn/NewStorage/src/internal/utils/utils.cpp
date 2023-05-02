#include "../file_wrapper/AbstractFileWrapper.hpp"
#include "../file_wrapper/BinaryFileWrapper.hpp"
#include "../file_wrapper/SingleSampleFileWrapper.hpp"
#include "../filesystem_wrapper/AbstractFilesystemWrapper.hpp"
#include "../filesystem_wrapper/LocalFilesystemWrapper.hpp"

using namespace storage;

AbstractFilesystemWrapper *get_filesystem_wrapper(std::string path,
                                                  std::string type) {
  if (type == "LOCAL") {
    return new LocalFilesystemWrapper(path);
  } else {
    throw std::runtime_error("Unknown filesystem wrapper type");
  }
}

AbstractFileWrapper *get_file_wrapper(std::string path, std::string type, YAML::Node config, AbstractFilesystemWrapper *filesystem_wrapper) {
  if (type == "BIN") {
    return new BinaryFileWrapper(path, config, filesystem_wrapper);
  } else if (type == "SINGLE_SAMPLE") {
    return new SingleSampleFileWrapper(path, config, filesystem_wrapper);
  } else {
    throw std::runtime_error("Unknown file wrapper type");
  }
}
