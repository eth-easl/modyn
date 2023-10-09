#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <rapidcsv/document.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace storage::file_wrapper;

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of(".") + 1) != "csv") {
    FAIL("The file extension must be .csv");
  }
}

void CsvFileWrapper::validate_file_content() {
  const rapidcsv::Document doc(file_path_, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator_, false, true),
                               rapidcsv::ConverterParams());
  doc.Parse();

  const size_t num_columns = doc.GetRows()[0].size();
  for (const rapidcsv::Row& row : doc.GetRows()) {
    if (row.size() != num_columns) {
      FAIL("CSV file is invalid: All rows must have the same number of columns.");
    }
  }

  const std::string label_column_name = doc.GetLabels()[label_index_];
  if (label_column_name != "label") {
    FAIL("CSV file is invalid: The label column must be named \"label\".");
  }
}

std::vector<std::vector<unsigned char>> read_csv_file(const std::string& file_path) {
  rapidcsv::Document doc(file_path, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator_, false, true),
                         rapidcsv::ConverterParams());
  doc.Parse();

  std::vector<std::vector<unsigned char>> samples;
  for (const rapidcsv::Row& row : doc.GetRows()) {
    samples.push_back(std::vector<unsigned char>(row.begin(), row.end()));
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples() override { return read_csv_file(file_path_); }

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(int64_t start, int64_t end) {
  ASSERT(start >= 0 && end >= start && end <= get_number_of_samples(), "Invalid indices");

  rapidcsv::Document doc(file_path_, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator_, false, true),
                         rapidcsv::ConverterParams());
  doc.Parse();

  std::vector<std::vector<unsigned char>> samples;
  for (int64_t i = start; i < end; i++) {
    const rapidcsv::Row& row = doc.GetRows()[i];
    samples.push_back(std::vector<unsigned char>(row.begin(), row.end()));
  }

  return samples;
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(
    const std::vector<int64_t>& indices) override {
  ASSERT(std::all_of(indices.begin(), indices.end(),
                     [&](int64_t index) { return index >= 0 && index < get_number_of_samples(); }),
         "Invalid indices");

  std::vector<std::vector<unsigned char>> samples;
  samples.reserve(indices.size());

  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  const std::span<unsigned char> file_span(content.data(), content.size());

  for (const int64_t index : indices) {
    samples.push_back(file_span.subspan(record_start(index), record_size));
  }

  return samples;
}

int64_t CsvFileWrapper::get_label(int64_t index) override {
  const rapidcsv::Document doc(file_path_, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator_, false, true),
                               rapidcsv::ConverterParams());
  doc.Parse();

  const rapidcsv::Row& row = doc.GetRows()[index];
  return std::stoi(row[label_index_]);
}

FileWrapperType CsvFileWrapper::get_type() { return FileWrapperType::CSV; }
