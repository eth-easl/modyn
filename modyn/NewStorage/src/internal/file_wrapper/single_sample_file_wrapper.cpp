#include "internal/file_wrapper/single_sample_file_wrapper.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <iostream>

using namespace storage;

int SingleSampleFileWrapper::get_number_of_samples() {
  if (this->file_path_.find(this->file_wrapper_config_["file_extension"].as<std::string>()) == std::string::npos) {
    return 0;
  }
  return 1;
}

int SingleSampleFileWrapper::get_label(int index) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (index != 0) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  if (!this->file_wrapper_config_["label_file_extension"]) {
    throw std::runtime_error("No label file extension defined.");
  }
  std::string label_file_extension = this->file_wrapper_config_["label_file_extension"].as<std::string>();
  auto label_path = std::filesystem::path(this->file_path_).replace_extension(label_file_extension);
  auto label = this->filesystem_wrapper_->get(label_path);
  if (label != nullptr) {
    auto label_str = std::string(reinterpret_cast<char*>(label->data()), label->size());
    return std::stoi(label_str);
  }
  throw std::runtime_error("Label file not found.");
}

std::vector<int>* SingleSampleFileWrapper::get_all_labels() {
  std::vector<int>* labels = new std::vector<int>();
  labels->push_back(get_label(0));
  return labels;
}

std::vector<unsigned char>* SingleSampleFileWrapper::get_sample(int index) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (index != 0) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return this->filesystem_wrapper_->get(this->file_path_);
}

std::vector<std::vector<unsigned char>>* SingleSampleFileWrapper::get_samples(int start, int end) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (start != 0 || end != 1) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return new std::vector<std::vector<unsigned char>>{*get_sample(0)};
}

std::vector<std::vector<unsigned char>>* SingleSampleFileWrapper::get_samples_from_indices(std::vector<int>* indices) {
  if (get_number_of_samples() == 0) {
    throw std::runtime_error("File has wrong file extension.");
  }
  if (indices->size() != 1) {
    throw std::runtime_error("SingleSampleFileWrapper contains only one sample.");
  }
  return new std::vector<std::vector<unsigned char>>{*get_sample(0)};
}

void SingleSampleFileWrapper::validate_file_extension() {
  if (!this->file_wrapper_config_["file_extension"]) {
    throw std::runtime_error("file_extension must be specified in the file wrapper config.");
  }
  std::string file_extension = this->file_wrapper_config_["file_extension"].as<std::string>();
  if (this->file_path_.find(file_extension) == std::string::npos) {
    throw std::runtime_error("File has wrong file extension.");
  }
}