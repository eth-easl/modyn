#pragma once

#include <string>
#include <vector>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/filesystem_wrapper/abstract_filesystem_wrapper.hpp"

namespace storage {

class CsvFileWrapper : public FileWrapper {
 private:
  char separator_;
  int label_index_;
  bool ignore_first_line_;
  std::string encoding_;

  void validate_file_extension();
  void validate_file_content();
  std::vector<unsigned char> filter_rows_samples(const std::vector<int64_t>& indices);
  std::vector<int64_t> filter_rows_labels(const std::vector<int64_t>& indices);

 public:
  CsvFileWrapper::CsvFileWrapper(std::string file_path, const YAML::Node& file_wrapper_config,
                                 std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : FileWrapper(std::move(file_path), file_wrapper_config, std::move(filesystem_wrapper)) {
    file_wrapper_type_ = FileWrapperType::CsvFileWrapper;

    if (file_wrapper_config_["separator"]) {
      separator_ = file_wrapper_config_["separator"].as<char>();
    } else {
      separator_ = ',';
    }

    if (!file_wrapper_config_["label_index"]) {
      throw std::invalid_argument("Please specify the index of the column that contains the label.");
    }
    label_index_ = file_wrapper_config_["label_index"].as<int>();

    if (label_index_ < 0) {
      throw std::invalid_argument("The label_index must be a non-negative integer.");
    }

    if (file_wrapper_config_["ignore_first_line"]) {
      ignore_first_line_ = file_wrapper_config_["ignore_first_line"].as<bool>();
    } else {
      ignore_first_line_ = false;
    }

    if (file_wrapper_config_["encoding"]) {
      encoding_ = file_wrapper_config_["encoding"].as<std::string>();
    } else {
      encoding_ = "utf-8";
    }

    validate_file_extension();

    // Do not validate the content only if "validate_file_content" is explicitly set to false
    if (!file_wrapper_config_["validate_file_content"] || file_wrapper_config_["validate_file_content"].as<bool>()) {
      validate_file_content();
    }
  }

  std::vector<unsigned char> get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) override;
  int64_t get_label(int64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  int64_t get_number_of_samples() override;
  void delete_samples(const std::vector<int64_t>& indices) override;
};
}  // namespace storage
