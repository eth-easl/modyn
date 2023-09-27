#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <algorithm>
#include <stdexcept>
#include <numeric>

using namespace storage;

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of(".") + 1) != "csv") {
    FAIL("The file extension must be .csv");
  }
}

void CsvFileWrapper::validate_file_content() {
  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  std::string file_content(content.begin(), content.end());

  std::vector<int> number_of_columns;
  int line_number = 0;

  std::istringstream file_stream(file_content);
  std::string line;
  while (std::getline(file_stream, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    int column_count = 0;

    while (std::getline(ss, cell, separator_)) {
      ++column_count;
      if (column_count - 1 == label_index_) {
        // Check if the label is numeric
        try {
          std::stoi(cell);
        } catch (const std::exception&) {
          FAIL("The label must be an integer.");
        }
      }
    }

    number_of_columns.push_back(column_count);
  }

  if (std::set<int>(number_of_columns.begin(), number_of_columns.end()).size() != 1) {
    FAIL("Some rows have different widths.");
  }
}

std::vector<unsigned char> CsvFileWrapper::get_sample(int64_t index) {
  std::vector<int64_t> indices = {index};
  return filter_rows_samples(indices)[0];
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples(int64_t start, int64_t end) {
  std::vector<int64_t> indices(end - start);
  std::iota(indices.begin(), indices.end(), start);
  return filter_rows_samples(indices);
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::get_samples_from_indices(const std::vector<int64_t>& indices) {
  return filter_rows_samples(indices);
}

int64_t CsvFileWrapper::get_label(int64_t index) {
  std::vector<int64_t> indices = {index};
  return filter_rows_labels(indices)[0];
}

std::vector<int64_t> CsvFileWrapper::get_all_labels() {
  std::vector<int64_t> labels;

  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  std::string file_content(content.begin(), content.end());

  int line_number = 0;

  std::istringstream file_stream(file_content);
  std::string line;
  while (std::getline(file_stream, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    int column_count = 0;

    while (std::getline(ss, cell, separator_)) {
      ++column_count;
      if (column_count - 1 == label_index_) {
        try {
          labels.push_back(std::stoi(cell));
        } catch (const std::exception&) {
          FAIL("The label must be an integer.");
        }
      }
    }
  }

  return labels;
}

int64_t CsvFileWrapper::get_number_of_samples() {
  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  std::string file_content(content.begin(), content.end());

  int64_t count = 0;
  int line_number = 0;

  std::istringstream file_stream(file_content);
  std::string line;
  while (std::getline(file_stream, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    ++count;
  }

  return count;
}

void CsvFileWrapper::delete_samples(const std::vector<int64_t>& indices) { 
  FAIL("Not implemented");
}

std::vector<std::vector<unsigned char>> CsvFileWrapper::filter_rows_samples(const std::vector<int64_t>& indices) {
  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  std::string file_content(content.begin(), content.end());

  std::vector<std::vector<unsigned char>> samples;
  int line_number = 0;
  int64_t current_index = 0;

  std::istringstream file_stream(file_content);
  std::string line;
  while (std::getline(file_stream, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    if (std::find(indices.begin(), indices.end(), current_index) != indices.end()) {
      std::vector<unsigned char> sample(line.begin(), line.end());
      samples.push_back(sample);
    }

    ++current_index;
  }

  if (samples.size() != indices.size()) {
    FAIL("Invalid index");
  }

  return samples;
}

std::vector<int64_t> CsvFileWrapper::filter_rows_labels(const std::vector<int64_t>& indices) {
  std::vector<unsigned char> content = filesystem_wrapper_->get(file_path_);
  std::string file_content(content.begin(), content.end());

  std::vector<int64_t> labels;
  int line_number = 0;
  int64_t current_index = 0;

  std::istringstream file_stream(file_content);
  std::string line;
  while (std::getline(file_stream, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    if (std::find(indices.begin(), indices.end(), current_index) != indices.end()) {
      std::istringstream ss(line);
      std::string cell;
      int column_count = 0;
      int64_t label = 0;

      while (std::getline(ss, cell, separator_)) {
        ++column_count;
        if (column_count - 1 == label_index_) {
          try {
            label = std::stoll(cell);
          } catch (const std::exception&) {
            FAIL("The label must be an integer.");
          }
        }
      }

      labels.push_back(label);
    }

    ++current_index;
  }

  if (labels.size() != indices.size()) {
    FAIL("Invalid index");
  }

  return labels;
}
