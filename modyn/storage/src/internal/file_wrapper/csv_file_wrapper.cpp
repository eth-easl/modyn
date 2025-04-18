#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <rapidcsv.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace modyn::storage;

void CsvFileWrapper::setup_document(const std::string& path) {
  stream_ = filesystem_wrapper_->get_stream(path);
  auto sep_params = rapidcsv::SeparatorParams(separator_);
  sep_params.mQuoteChar = quote_;
  sep_params.mQuotedLinebreaks = allow_quoted_linebreaks_;
  doc_ = rapidcsv::Document(*stream_, label_params_, sep_params);
}

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of('.') + 1) != "csv") {
    FAIL("The file extension must be .csv");
  }
}

std::vector<unsigned char> CsvFileWrapper::get_sample(uint64_t index) {
  ASSERT(index < get_number_of_samples(), "Invalid index");

  std::vector<std::string> row = doc_.GetRow<std::string>(index);
  if (has_labels_) {
    row.erase(row.begin() + static_cast<int64_t>(label_index_));
  }
  if (has_targets_) {
    row.erase(row.begin() + static_cast<int64_t>(target_index_));
  }

  std::string row_string;
  for (const auto& cell : row) {
    row_string += cell + separator_;
  }
  row_string.pop_back();
  return {row_string.begin(), row_string.end()};
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(uint64_t start, uint64_t end) {
  ASSERT(end >= start && end <= get_number_of_samples(), "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  for (uint64_t i = start; i < end; ++i) {
    std::vector<std::string> row = doc_.GetRow<std::string>(i);
    if (has_labels_) {
      row.erase(row.begin() + static_cast<int64_t>(label_index_));
    }
    if (has_targets_) {
      row.erase(row.begin() + static_cast<int64_t>(target_index_));
    }

    std::string row_string;
    for (const auto& cell : row) {
      row_string += cell + separator_;
    }
    row_string.pop_back();
    samples.emplace_back(row_string.begin(), row_string.end());
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(const std::vector<uint64_t>& indices) {
  std::vector<std::vector<unsigned char>> samples;
  for (const uint64_t index : indices) {
    std::vector<std::string> row = doc_.GetRow<std::string>(index);
    if (has_labels_) {
      row.erase(row.begin() + static_cast<int64_t>(label_index_));
    }
    if (has_targets_) {
      row.erase(row.begin() + static_cast<int64_t>(target_index_));
    }
    std::string row_string;
    for (const auto& cell : row) {
      row_string += cell + separator_;
    }
    row_string.pop_back();
    samples.emplace_back(row_string.begin(), row_string.end());
  }
  return samples;
}

int64_t CsvFileWrapper::get_label(uint64_t index) {
  ASSERT(index < get_number_of_samples(), "Invalid index");
  return doc_.GetCell<int64_t>(static_cast<size_t>(label_index_), static_cast<size_t>(index));
}

std::vector<int64_t> CsvFileWrapper::get_all_labels() {
  std::vector<int64_t> labels;
  if (!has_labels_) {
    return {};
  }
  const uint64_t num_samples = get_number_of_samples();
  for (uint64_t i = 0; i < num_samples; i++) {
    labels.push_back(get_label(i));
  }
  return labels;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_targets(uint64_t start, uint64_t end) {
  ASSERT(end >= start && end <= get_number_of_samples(), "Invalid indices");

  std::vector<std::vector<unsigned char>> targets;
  for (uint64_t i = start; i < end; ++i) {
    targets.push_back(get_target(i));
  }
  return targets;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_targets_from_indices(const std::vector<uint64_t>& indices) {
  std::vector<std::vector<unsigned char>> targets;
  for (const uint64_t index : indices) {
    targets.push_back(get_target(index));
  }
  return targets;
}

std::vector<unsigned char> CsvFileWrapper::get_target(uint64_t index) {
  ASSERT(index < get_number_of_samples(), "Invalid index");

  std::vector<std::string> row = doc_.GetRow<std::string>(index);
  if (target_index_ >= row.size()) {
    throw std::runtime_error("Target index out of range for this row");
  }
  std::string target_cell = row[target_index_];
  return {target_cell.begin(), target_cell.end()};
}

uint64_t CsvFileWrapper::get_number_of_samples() { return static_cast<uint64_t>(doc_.GetRowCount()); }

void CsvFileWrapper::delete_samples(const std::vector<uint64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(), [&](uint64_t index) { return index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<uint64_t> indices_copy = indices;
  std::sort(indices_copy.begin(), indices_copy.end(), std::greater<>());

  for (const size_t index : indices_copy) {
    doc_.RemoveRow(index);
  }

  doc_.Save(file_path_);
}

void CsvFileWrapper::set_file_path(const std::string& path) {
  file_path_ = path;

  if (stream_->is_open()) {
    stream_->close();
  }

  setup_document(path);
}

FileWrapperType CsvFileWrapper::get_type() { return FileWrapperType::CSV; }
