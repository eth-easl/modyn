#include "internal/file_wrapper/csv_file_wrapper.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

using namespace storage;

void CsvFileWrapper::validate_file_extension() {
  if (file_path_.substr(file_path_.find_last_of(".") + 1) != "csv") {
    throw std::invalid_argument("File has wrong file extension.");
  }
}

void CsvFileWrapper::validate_file_content() {
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for validation: " + file_path_);
  }

  std::string line;
  std::vector<int> number_of_columns;
  int line_number = 0;

  while (std::getline(file, line)) {
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
          throw std::invalid_argument("The label must be an integer.");
        }
      }
    }

    number_of_columns.push_back(column_count);
  }

  file.close();

  if (std::set<int>(number_of_columns.begin(), number_of_columns.end()).size() != 1) {
    throw std::invalid_argument("Some rows have different widths.");
  }
}

std::vector<unsigned char> CsvFileWrapper::get_sample(int64_t index) {
  std::vector<int64_t> indices = {index};
  return filter_rows_samples(indices);
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
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for reading labels: " + file_path_);
  }

  std::string line;
  int line_number = 0;

  while (std::getline(file, line)) {
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
          throw std::runtime_error("Failed to parse label as an integer.");
        }
      }
    }
  }

  file.close();

  return labels;
}

int64_t CsvFileWrapper::get_number_of_samples() {
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for counting samples: " + file_path_);
  }

  int64_t count = 0;
  std::string line;
  int line_number = 0;

  while (std::getline(file, line)) {
    ++line_number;

    // Skip the first line if required
    if (line_number == 1 && ignore_first_line_) {
      continue;
    }

    ++count;
  }

  file.close();

  return count;
}

void CsvFileWrapper::delete_samples(const std::vector<int64_t>& indices) { throw std::logic_error("Not implemented"); }

std::vector<unsigned char> CsvFileWrapper::filter_rows_samples(const std::vector<int64_t>& indices) {
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for filtering rows: " + file_path_);
  }

  std::vector<unsigned char> samples;
  std::string line;
  int line_number = 0;
  int64_t current_index = 0;

  while (std::getline(file, line)) {
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

  file.close();

  if (samples.size() != indices.size()) {
    throw std::out_of_range("Invalid index");
  }

  return samples;
}

std::vector<int64_t> CsvFileWrapper::filter_rows_labels(const std::vector<int64_t>& indices) {
  std::ifstream file(file_path_);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for filtering rows: " + file_path_);
  }

  std::vector<int64_t> labels;
  std::string line;
  int line_number = 0;
  int64_t current_index = 0;

  while (std::getline(file, line)) {
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
            throw std::runtime_error("Failed to parse label as an integer.");
          }
        }
      }

      labels.push_back(label);
    }

    ++current_index;
  }

  file.close();

  if (labels.size() != indices.size()) {
    throw std::out_of_range("Invalid index");
  }

  return labels;
}
