#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "internal/utils/utils.hpp"

using namespace storage::filesystem_wrapper;

std::vector<unsigned char> LocalFilesystemWrapper::get(const std::string& path) {
  std::ifstream file;
  file.open(path, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  return buffer;
}

std::ifstream& LocalFilesystemWrapper::get_stream(const std::string& path) {
  std::unique_ptr<std::ifstream> file = std::make_unique<std::ifstream>();
  file->open(path, std::ios::binary);
  std::ifstream& reference = *file;
  return reference;
}

bool LocalFilesystemWrapper::exists(const std::string& path) { return std::filesystem::exists(path); }

std::vector<std::string> LocalFilesystemWrapper::list(const std::string& path, bool recursive) {
  std::vector<std::string> paths = std::vector<std::string>();

  if (!std::filesystem::exists(path)) {
    return paths;
  }

  if (recursive) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
      if (!std::filesystem::is_directory(entry)) {
        paths.push_back(entry.path());
      }
    }
  } else {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      if (!std::filesystem::is_directory(entry)) {
        paths.push_back(entry.path());
      }
    }
  }

  return paths;
}

bool LocalFilesystemWrapper::is_directory(const std::string& path) { return std::filesystem::is_directory(path); }

bool LocalFilesystemWrapper::is_file(const std::string& path) { return std::filesystem::is_regular_file(path); }

int64_t LocalFilesystemWrapper::get_file_size(const std::string& path) {
  return static_cast<int64_t>(std::filesystem::file_size(path));
}

int64_t LocalFilesystemWrapper::get_modified_time(const std::string& path) {
  ASSERT(is_valid_path(path), fmt::format("Invalid path: {}", path));
  ASSERT(exists(path), fmt::format("Path does not exist: {}", path));

  // For the most system reliable way to get the file timestamp, we use stat
  struct stat file_stat = {};
  if (stat(path.c_str(), &file_stat) != 0) {
    FAIL(fmt::format("File timestamp not readable: {}", path));
  }

  const time_t file_timestamp = file_stat.st_mtime;
  const auto int64_file_timestamp = static_cast<int64_t>(file_timestamp);
  return int64_file_timestamp;
}

bool LocalFilesystemWrapper::is_valid_path(const std::string& path) { return std::filesystem::exists(path); }

bool LocalFilesystemWrapper::remove(const std::string& path) {
  ASSERT(is_valid_path(path), fmt::format("Invalid path: {}", path));
  ASSERT(!std::filesystem::is_directory(path), fmt::format("Path is a directory: {}", path));

  SPDLOG_DEBUG("Removing file: {}", path);

  return std::filesystem::remove(path);
}

FilesystemWrapperType LocalFilesystemWrapper::get_type() { return FilesystemWrapperType::LOCAL; }
