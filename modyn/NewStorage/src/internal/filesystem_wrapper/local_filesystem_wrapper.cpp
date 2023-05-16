#include "internal/filesystem_wrapper/local_filesystem_wrapper.hpp"

#include <spdlog/spdlog.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifndef WIN32
#include <unistd.h>
#endif

#ifdef WIN32
#define stat _stat
#endif

const char path_separator =
#ifdef _WIN32
    '\\';
#else
    '/';
#endif

using namespace storage;

std::vector<unsigned char> LocalFilesystemWrapper::get(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  if (not is_file(path)) {
    throw std::runtime_error("Path " + path + " is a directory.");
  }
  std::ifstream file;
  file.open(path, std::ios::binary);
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});
  file.close();
  return buffer;
}

bool LocalFilesystemWrapper::exists(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  std::ifstream file;
  file.open(path);
  const bool exists = file.good();
  file.close();
  return exists;
}

std::vector<std::string> LocalFilesystemWrapper::list(const std::string& path, bool recursive) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  if (not is_directory(path)) {
    throw std::runtime_error("Path " + path + " is a file.");
  }
  std::vector<std::string> files = std::vector<std::string>();
  std::vector<std::string> directories = std::vector<std::string>();
  std::vector<std::string> paths = std::vector<std::string>();
  paths.push_back(path);
  while (!paths.empty()) {
    const std::string current_path = paths.back();
    paths.pop_back();
    auto current_files = std::vector<std::string>();
    auto current_directories = std::vector<std::string>();
    for (const auto& entry : std::filesystem::directory_iterator(current_path)) {
      const std::string entry_path = entry.path();
      if (std::filesystem::is_directory(entry_path)) {
        current_directories.push_back(entry_path);
      } else {
        current_files.push_back(entry_path);
      }
    }
    if (recursive) {
      paths.insert(paths.end(), current_directories.begin(), current_directories.end());
    }
    files.insert(files.end(), current_files.begin(), current_files.end());
    directories.insert(directories.end(), current_directories.begin(), current_directories.end());
  }
  return files;
}

bool LocalFilesystemWrapper::is_directory(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  return std::filesystem::is_directory(path);
}

bool LocalFilesystemWrapper::is_file(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  return std::filesystem::is_regular_file(path);
}

int64_t LocalFilesystemWrapper::get_file_size(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  if (not is_file(path)) {
    throw std::runtime_error("Path " + path + " is a directory.");
  }
  std::ifstream file;
  file.open(path, std::ios::binary);
  file.seekg(0, std::ios::end);
  const int64_t size = file.tellg();
  file.close();
  return size;
}

int64_t LocalFilesystemWrapper::get_modified_time(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  if (not exists(path)) {
    throw std::runtime_error("Path " + path + " does not exist.");
  }
  struct stat result = {};
  int64_t mod_time;
  if (stat(path.c_str(), &result) == 0) {
    mod_time = static_cast<int64_t>(result.st_mtime);
  } else {
    throw std::runtime_error("Path " + path + " does not exist.");
  }
  return mod_time;
}

int64_t LocalFilesystemWrapper::get_created_time(const std::string& path) {
  if (not is_valid_path(path)) {
    throw std::invalid_argument("Path " + path + " is not valid.");
  }
  if (not exists(path)) {
    throw std::runtime_error("Path " + path + " does not exist.");
  }
  struct stat result = {};
  int64_t mod_time;
  if (stat(path.c_str(), &result) == 0) {
    mod_time = static_cast<int64_t>(result.st_mtime);
  } else {
    throw std::runtime_error("Path " + path + " does not exist.");
  }
  return mod_time;
}

bool LocalFilesystemWrapper::is_valid_path(const std::string& path) { return path.find("..") == std::string::npos; }

std::string LocalFilesystemWrapper::join(const std::vector<std::string>& paths) {
  std::string joined_path;
  for (uint64_t i = 0; i < paths.size(); i++) {
    joined_path += paths[i];
    if (i < paths.size() - 1) {
      joined_path += path_separator;
    }
  }
  return joined_path;
}
