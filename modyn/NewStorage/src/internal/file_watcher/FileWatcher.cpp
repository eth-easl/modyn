#include "FileWatcher.hpp"

using namespace storage;

bool FileWatcher::file_unknown(std::string file_path) {}

void FileWatcher::handle_file_paths(
    std::vector<std::string> file_paths, std::string data_file_extension,
    AbstractFileWrapper *file_wrapper,
    AbstractFilesystemWrapper *filesystem_wrapper, int timestamp) {}

void FileWatcher::update_files_in_directory(
    AbstractFileWrapper *file_wrapper,
    AbstractFilesystemWrapper *filesystem_wrapper, std::string directory_path,
    int timestamp) {}

void FileWatcher::seek_dataset() {}

void FileWatcher::seek() {}

void FileWatcher::get_datasets() {}

void FileWatcher::run() {}