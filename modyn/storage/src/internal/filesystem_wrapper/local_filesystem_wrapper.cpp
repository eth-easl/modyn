#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "modyn/utils/utils.hpp"

using namespace modyn::storage;

std::vector<unsigned char> LocalFilesystemWrapper::get(const std::string& path) {
  std::ifstream file;
  file.open(path, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  return buffer;
}

std::shared_ptr<std::ifstream> LocalFilesystemWrapper::get_stream(const std::string& path) {
  std::shared_ptr<std::ifstream> file = std::make_shared<std::ifstream>();
  file->open(path, std::ios::binary);
  return file;
}

bool LocalFilesystemWrapper::exists(const std::string& path) { return std::filesystem::exists(path); }

std::vector<std::string> LocalFilesystemWrapper::list(const std::string& path, bool recursive, std::string extension) {
  std::vector<std::string> paths;

  if (!std::filesystem::exists(path)) {
    return paths;
  }

  if (recursive) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
      const std::filesystem::path& entry_path = entry.path();
      if (!std::filesystem::is_directory(entry) && entry_path.extension().string() == extension) {
        paths.push_back(entry_path);
      }
    }
  } else {
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
      const std::filesystem::path& entry_path = entry.path();
      if (!std::filesystem::is_directory(entry) && entry_path.extension().string() == extension) {
        paths.push_back(entry_path);
      }
    }
  }

  return paths;
}

bool LocalFilesystemWrapper::is_directory(const std::string& path) { return std::filesystem::is_directory(path); }

bool LocalFilesystemWrapper::is_file(const std::string& path) { return std::filesystem::is_regular_file(path); }

uint64_t LocalFilesystemWrapper::get_file_size(const std::string& path) {
  return static_cast<int64_t>(std::filesystem::file_size(path));
}

int64_t LocalFilesystemWrapper::get_modified_time(const std::string& path) {
  ASSERT(is_valid_path(path), fmt::format("Invalid path: {}", path));
  ASSERT(exists(path), fmt::format("Path does not exist: {}", path));
  static_assert(sizeof(int64_t) >= sizeof(std::time_t), "Cannot cast time_t to int64_t");

  // there is no portable way to get the modified time of a file in C++17 and earlier
  // clang-format off
  struct stat file_attribute {};//This line keeps getting changed by clang format in my machine and not passing the format test when pushed.
  // clang-format on
  stat(path.c_str(), &file_attribute);
  return static_cast<int64_t>(file_attribute.st_mtime);
  /* C++20 version, not supported by compilers yet */
  /*
    const auto modified_time = std::filesystem::last_write_time(path);
    const auto system_time = std::chrono::clock_cast<std::chrono::system_clock>(modified_time);
    const std::time_t time = std::chrono::system_clock::to_time_t(system_time);
    return static_cast<int64_t>(time); */
}

bool LocalFilesystemWrapper::is_valid_path(const std::string& path) { return std::filesystem::exists(path); }

bool LocalFilesystemWrapper::remove(const std::string& path) {
  ASSERT(!std::filesystem::is_directory(path), fmt::format("Path is a directory: {}", path));

  if (!std::filesystem::exists(path)) {
    SPDLOG_WARN(fmt::format("Trying to delete already deleted file {}", path));
    return true;
  }

  SPDLOG_DEBUG("Removing file: {}", path);

  return std::filesystem::remove(path);
}

FilesystemWrapperType LocalFilesystemWrapper::get_type() { return FilesystemWrapperType::LOCAL; }
