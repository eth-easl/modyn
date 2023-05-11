#include "../include/Storage.hpp"
#include <filesystem>
#include <iostream>
#include <spdlog/spdlog.h>

using namespace storage;

Storage::Storage(std::string config_file) {
  /* Initialize the storage service. */
  YAML::Node config = YAML::LoadFile(config_file);
  this->config = config;
}

void Storage::run() {
  /* Run the storage service. */
  SPDLOG_INFO("Running storage service.");

  // Create the database tables

  // Create the dataset watcher process in a new thread

  // Start the storage grpc server
}