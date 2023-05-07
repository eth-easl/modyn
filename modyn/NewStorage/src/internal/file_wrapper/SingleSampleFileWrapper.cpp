#include "SingleSampleFileWrapper.hpp"
#include <filesystem>
#include <iostream>

using namespace storage;

int SingleSampleFileWrapper::get_number_of_samples() {
  if (this->path.find(
          this->file_wrapper_config["file_extension"].as<std::string>()) ==
      std::string::npos) {
    return 0;
  }
  return 1;
}

int SingleSampleFileWrapper::get_label(int index) {
  if (get_number_of_samples() == 0)
    throw std::runtime_error("File has wrong file extension.");
  if (index != 0)
    throw std::runtime_error(
        "SingleSampleFileWrapper contains only one sample.");
  if (!this->file_wrapper_config["label_file_extension"])
    throw std::runtime_error("No label file extension defined.");
  std::string label_file_extension =
      this->file_wrapper_config["label_file_extension"].as<std::string>();
  auto label_path =
      std::filesystem::path(this->path).replace_extension(label_file_extension);
  auto label = this->filesystem_wrapper->get(label_path);
  if (label != nullptr) {
    auto label_str = std::string((char *)label->data(), label->size());
    return std::stoi(label_str);
  }
  throw std::runtime_error("Label file not found.");
}

std::vector<int> *SingleSampleFileWrapper::get_all_labels() {
  std::vector<int> *labels = new std::vector<int>();
  labels->push_back(get_label(0));
  return labels;
}

std::vector<unsigned char> *SingleSampleFileWrapper::get_sample(int index) {
  if (get_number_of_samples() == 0)
    throw std::runtime_error("File has wrong file extension.");
  if (index != 0)
    throw std::runtime_error(
        "SingleSampleFileWrapper contains only one sample.");
  return this->filesystem_wrapper->get(this->path);
}

std::vector<std::vector<unsigned char>> *
SingleSampleFileWrapper::get_samples(int start, int end) {
  if (get_number_of_samples() == 0)
    throw std::runtime_error("File has wrong file extension.");
  if (start != 0 || end != 1)
    throw std::runtime_error(
        "SingleSampleFileWrapper contains only one sample.");
  return new std::vector<std::vector<unsigned char>>{*get_sample(0)};
}

std::vector<std::vector<unsigned char>> *
SingleSampleFileWrapper::get_samples_from_indices(std::vector<int> *indices) {
  if (get_number_of_samples() == 0)
    throw std::runtime_error("File has wrong file extension.");
  if (indices->size() != 1)
    throw std::runtime_error(
        "SingleSampleFileWrapper contains only one sample.");
  return new std::vector<std::vector<unsigned char>>{*get_sample(0)};
}