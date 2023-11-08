#pragma once

#include <rapidcsv.h>

#include <string>
#include <vector>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "modyn/utils/utils.hpp"

namespace modyn::storage {

class CsvFileWrapper : public FileWrapper {
 public:
  CsvFileWrapper(const std::string& path, const YAML::Node& fw_config,
                 std::shared_ptr<FilesystemWrapper> filesystem_wrapper)
      : FileWrapper{path, fw_config, std::move(filesystem_wrapper)} {
    if (file_wrapper_config_["separator"]) {
      separator_ = file_wrapper_config_["separator"].as<char>();
    } else {
      separator_ = ',';
    }

    if (!file_wrapper_config_["label_index"]) {
      FAIL("Please specify the index of the column that contains the label.");
    }
    label_index_ = file_wrapper_config_["label_index"].as<int64_t>();

    if (label_index_ < 0) {
      FAIL("The label_index must be a non-negative integer.");
    }

    bool ignore_first_line = false;
    if (file_wrapper_config_["ignore_first_line"]) {
      ignore_first_line = file_wrapper_config_["ignore_first_line"].as<bool>();
    } else {
      ignore_first_line = false;
    }

    ASSERT(filesystem_wrapper_->exists(path), "The file does not exist.");

    validate_file_extension();

    label_params_ = rapidcsv::LabelParams(ignore_first_line ? 0 : -1);

    stream_ = filesystem_wrapper_->get_stream(path);

    doc_ = rapidcsv::Document(*stream_, label_params_, rapidcsv::SeparatorParams(separator_));
  }

  ~CsvFileWrapper() override {
    if (stream_->is_open()) {
      stream_->close();
    }
  }

  int64_t get_number_of_samples() override;
  int64_t get_label(int64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  std::vector<unsigned char> get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) override;
  void validate_file_extension() override;
  void delete_samples(const std::vector<int64_t>& indices) override;
  void set_file_path(const std::string& path) override;
  FileWrapperType get_type() override;

 private:
  char separator_;
  int64_t label_index_;
  rapidcsv::Document doc_;
  rapidcsv::LabelParams label_params_;
  std::shared_ptr<std::ifstream> stream_;
};
}  // namespace modyn::storage
