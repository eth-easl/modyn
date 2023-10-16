#pragma once

#include <rapidcsv.h>

#include <string>
#include <vector>

#include "internal/file_wrapper/file_wrapper.hpp"
#include "internal/utils/utils.hpp"

namespace storage::file_wrapper {

class CsvFileWrapper : public storage::file_wrapper::FileWrapper {
 public:
  CsvFileWrapper(const std::string& path, const YAML::Node& fw_config,
                 std::shared_ptr<storage::filesystem_wrapper::FilesystemWrapper> filesystem_wrapper)
      : storage::file_wrapper::FileWrapper{path, fw_config, std::move(filesystem_wrapper)} {
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

    rapidcsv::LabelParams label_params(ignore_first_line ? 0 : -1);

    std::ifstream& stream = filesystem_wrapper_->get_stream(path);

    doc_ = rapidcsv::Document(stream, label_params, rapidcsv::SeparatorParams(separator_));
  }

  std::vector<unsigned char> get_sample(int64_t index) override;
  std::vector<std::vector<unsigned char>> get_samples(int64_t start, int64_t end) override;
  std::vector<std::vector<unsigned char>> get_samples_from_indices(const std::vector<int64_t>& indices) override;
  int64_t get_label(int64_t index) override;
  std::vector<int64_t> get_all_labels() override;
  int64_t get_number_of_samples() override;
  void delete_samples(const std::vector<int64_t>& indices) override;
  FileWrapperType get_type() override;
  ~CsvFileWrapper() override = default;
  void validate_file_extension() override;

 private:
  char separator_;
  int64_t label_index_;
  rapidcsv::Document doc_;
};
}  // namespace storage::file_wrapper
