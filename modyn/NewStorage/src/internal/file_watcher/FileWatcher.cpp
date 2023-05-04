#include "FileWatcher.hpp"
#include <csignal>

using namespace storage;


volatile sig_atomic_t file_watcher_sigflag = 0;
void file_watcher_signal_handler(int signal) {
    file_watcher_sigflag = 1;
}

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

void FileWatcher::run() {
    std::signal(SIGTERM, file_watcher_signal_handler);

    while (true) {
        // Do some work
        if (file_watcher_sigflag) {
            // Perform any necessary cleanup
            // before exiting
            break;
        }
    }
}
