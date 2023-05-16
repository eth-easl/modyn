#include "storage.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>

using namespace storage;

Storage::Storage(const std::string& config_file) {
  /* Initialize the storage service. */
  const YAML::Node config = YAML::LoadFile(config_file);
  config_ = config;
}

void Storage::run() {  // NOLINT // TODO: Remove NOLINT after implementation
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  // Create the database tables

  // Create the dataset watcher process in a new thread

  // Start the storage grpc server
}