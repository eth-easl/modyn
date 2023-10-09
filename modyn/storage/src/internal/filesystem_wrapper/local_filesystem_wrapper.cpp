#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "internal/filesystem_wrapper/filesystem_wrapper_utils.hpp"

const char path_separator = std::filesystem::path::preferred_separator;

using namespace storage::filesystem_wrapper;

std::vector<unsigned char> LocalFilesystemWrapper::get(const std::string& path) {
  std::ifstream file;
  file.open(path, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  return buffer;
}

bool LocalFilesystemWrapper::exists(const std::string& path) { return std::filesystem::exists(path); }

std::vector<std::string> LocalFilesystemWrapper::list(const std::string& path, bool recursive) {
  std::vector<std::string> paths = std::vector<std::string>();

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

int64_t LocalFilesystemWrapper::get_file_size(const std::string& path) { return std::filesystem::file_size(path); }

int64_t LocalFilesystemWrapper::get_modified_time(const std::string& path) {
  ASSERT(is_valid_path(path), fmt::format("Invalid path: {}", path));
  ASSERT(exists(path), fmt::format("Path does not exist: {}", path));

  std::filesystem::file_time_type time = std::filesystem::last_write_time(path);
  return std::chrono::duration_cast<std::chrono::seconds>(time.time_since_epoch()).count();
}

bool LocalFilesystemWrapper::is_valid_path(const std::string& path) { return std::filesystem::exists(path); }

bool LocalFilesystemWrapper::remove(const std::string& path) {
  ASSERT(is_valid_path(path), fmt::format("Invalid path: {}", path));
  ASSERT(!std::filesystem::is_directory(path), fmt::format("Path is a directory: {}", path));

  return std::filesystem::remove(path);
}

std::string LocalFilesystemWrapper::join(const std::vector<std::string>& paths) {
  return fmt::format("{}", fmt::join(paths, path_separator));
}

FilesystemWrapperType LocalFilesystemWrapper::get_type() { return FilesystemWrapperType::LOCAL; }
