#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <rapidcsv.h>

#include <algorithm>
#include <numeric>
#include <span>
#include <stdexcept>

using namespace storage::file_wrapper;

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of(".") + 1) != "csv") {
    FAIL("The file extension must be .csv");
  }
}

std::vector<unsigned char> CsvFileWrapper::get_sample(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");

  return doc_.GetRow<unsigned char>(index);
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(int64_t start, int64_t end) {
  ASSERT(start >= 0 && end >= start && end <= get_number_of_samples(), "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  size_t start_t = start;
  size_t end_t = end;
  for (size_t i = start_t; i < end_t; i++) {
    samples.push_back(doc_.GetRow<unsigned char>(i));
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  for (size_t i : indices) {
    samples.push_back(doc_.GetRow<unsigned char>(i));
  }

  return samples;
}

int64_t CsvFileWrapper::get_label(int64_t index) { return doc_.GetRow<unsigned char>(index)[label_index_]; }

std::vector<int64_t> CsvFileWrapper::get_all_labels() {
  std::vector<int64_t> labels;
  size_t num_samples = get_number_of_samples();
  for (size_t i = 0; i < num_samples; i++) {
    labels.push_back(get_label(i));
  }
  return labels;
}

int64_t CsvFileWrapper::get_number_of_samples() { return doc_.GetRowCount() - (ignore_first_line_ ? 1 : 0); }

void CsvFileWrapper::delete_samples(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  for (size_t i : indices) {
    doc_.RemoveRow(i);
  }
  doc_.Save();
}

FileWrapperType CsvFileWrapper::get_type() { return FileWrapperType::CSV; }
