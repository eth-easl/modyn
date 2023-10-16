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

  std::vector<std::string> row = doc_.GetRow<std::string>(index);
  row.erase(row.begin() + label_index_);
  std::string s = std::accumulate(row.begin(), row.end(), std::string(),
                                  [&](const std::string& a, const std::string& b) { return a + separator_ + b; });
  s.erase(s.begin());
  return std::vector<unsigned char>(s.begin(), s.end());
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(int64_t start, int64_t end) {
  ASSERT(start >= 0 && end >= start && end <= get_number_of_samples(), "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  size_t start_t = start;
  size_t end_t = end;
  for (size_t i = start_t; i < end_t; i++) {
    std::vector<std::string> row = doc_.GetRow<std::string>(i);
    row.erase(row.begin() + label_index_);
    std::string s = std::accumulate(row.begin(), row.end(), std::string(),
                                    [&](const std::string& a, const std::string& b) { return a + separator_ + b; });
    s.erase(s.begin());
    samples.push_back(std::vector<unsigned char>(s.begin(), s.end()));
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  for (size_t i : indices) {
    std::vector<std::string> row = doc_.GetRow<std::string>(i);
    row.erase(row.begin() + label_index_);
    std::string s = std::accumulate(row.begin(), row.end(), std::string(),
                                    [&](const std::string& a, const std::string& b) { return a + separator_ + b; });
    s.erase(s.begin());
    samples.push_back(std::vector<unsigned char>(s.begin(), s.end()));
  }

  return samples;
}

int64_t CsvFileWrapper::get_label(int64_t index) {
  ASSERT(index >= 0 && index < get_number_of_samples(), "Invalid index");
  return doc_.GetCell<int64_t>((size_t)label_index_, (size_t)index);
}

std::vector<int64_t> CsvFileWrapper::get_all_labels() {
  std::vector<int64_t> labels;
  size_t num_samples = get_number_of_samples();
  for (size_t i = 0; i < num_samples; i++) {
    labels.push_back(get_label(i));
  }
  return labels;
}

int64_t CsvFileWrapper::get_number_of_samples() { return doc_.GetRowCount(); }

void CsvFileWrapper::delete_samples(const std::vector<int64_t>& indices) {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<int64_t> indices_copy = indices;
  std::sort(indices_copy.begin(), indices_copy.end(), std::greater<int64_t>());

  for (size_t i : indices_copy) {
    doc_.RemoveRow(i);
  }

  doc_.Save(file_path_);
}

FileWrapperType CsvFileWrapper::get_type() { return FileWrapperType::CSV; }
