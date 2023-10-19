#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <rapidcsv.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace storage::file_wrapper;

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of('.') + 1) != "csv") {
    FAIL("The file extension must be .csv");
  }
}

std::vector<unsigned char> CsvFileWrapper::get_sample(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");

  std::vector<std::string> row = doc_.GetRow<std::string>(index);
  row.erase(row.begin() + label_index_);
  std::string row_string;
  for (const auto& cell : row) {
    row_string += cell + separator_;
  }
  row_string.pop_back();
  return {row_string.begin(), row_string.end()};
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(int64_t start, int64_t end) {
  ASSERT(start >= 0 && end >= start && end <= get_number_of_samples(), "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  const size_t start_t = start;
  const size_t end_t = end;
  for (size_t i = start_t; i < end_t; i++) {
    std::vector<std::string> row = doc_.GetRow<std::string>(i);
    row.erase(row.begin() + label_index_);
    std::string row_string;
    for (const auto& cell : row) {
      row_string += cell + separator_;
    }
    row_string.pop_back();
    samples.emplace_back(row_string.begin(), row_string.end());
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  for (const size_t index : indices) {
    std::vector<std::string> row = doc_.GetRow<std::string>(index);
    row.erase(row.begin() + label_index_);
    std::string row_string;
    for (const auto& cell : row) {
      row_string += cell + separator_;
    }
    row_string.pop_back();
    samples.emplace_back(row_string.begin(), row_string.end());
  }
  return samples;
}

int64_t CsvFileWrapper::get_label(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");
  return doc_.GetCell<int64_t>(static_cast<size_t>(label_index_), static_cast<size_t>(index));
}

std::vector<int64_t> CsvFileWrapper::get_all_labels() {
  std::vector<int64_t> labels;
  const int64_t num_samples = get_number_of_samples();
  for (int64_t i = 0; i < num_samples; i++) {
    labels.push_back(get_label(i));
  }
  return labels;
}

int64_t CsvFileWrapper::get_number_of_samples() { return static_cast<int64_t>(doc_.GetRowCount()); }

void CsvFileWrapper::delete_samples(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<int64_t> indices_copy = indices;
  std::sort(indices_copy.begin(), indices_copy.end(), std::greater<>());

  for (const size_t index : indices_copy) {
    doc_.RemoveRow(index);
  }

  doc_.Save(file_path_);
}

void CsvFileWrapper::set_file_path(const std::string& path) {
  file_path_ = path;
  std::ifstream& stream = filesystem_wrapper_->get_stream(path);

  doc_ = rapidcsv::Document(stream, label_params_, rapidcsv::SeparatorParams(separator_));
}

FileWrapperType CsvFileWrapper::get_type() { return FileWrapperType::CSV; }
