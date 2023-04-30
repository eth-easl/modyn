#ifndef LOCAL_FILESYSTEM_WRAPPER_H_
#define LOCAL_FILESYSTEM_WRAPPER_H_

#include "AbstractFileSystemWrapper.h"

namespace storage
{
    class LocalFileSystemWrapper : public AbstractFileSystemWrapper
    {
    public:
        LocalFileSystemWrapper(std::string base_path) : AbstractFileSystemWrapper(base_path) {}
        std::vector<unsigned char> *get(std::string path);
        bool exists(std::string path);
        std::vector<std::string> *list(std::string path, bool recursive = false);
        bool is_directory(std::string path);
        bool is_file(std::string path);
        int get_file_size(std::string path);
        int get_modified_time(std::string path);
        int get_created_time(std::string path);
        std::string join(std::vector<std::string> paths);
        bool is_valid_path(std::string path);
    };
}

#endif