#ifndef MOCK_FILESYSTEM_WRAPPER_HPP
#define MOCK_FILESYSTEM_WRAPPER_HPP

#include "../../../src/internal/filesystem_wrapper/AbstractFileSystemWrapper.hpp"
#include "gmock/gmock.h"
#include <gtest/gtest.h>
#include <fstream>

namespace storage
{
    class MockFileSystemWrapper : public storage::AbstractFileSystemWrapper
    {
    public:
        MockFileSystemWrapper() : AbstractFileSystemWrapper("") {};
        MOCK_METHOD(std::vector<unsigned char> *, get, (std::string path), (override));
        MOCK_METHOD(bool, exists, (std::string path), (override));
        MOCK_METHOD(std::vector<std::string> *, list, (std::string path, bool recursive), (override));
        MOCK_METHOD(bool, is_directory, (std::string path), (override));
        MOCK_METHOD(bool, is_file, (std::string path), (override));
        MOCK_METHOD(int, get_file_size, (std::string path), (override));
        MOCK_METHOD(int, get_modified_time, (std::string path), (override));
        MOCK_METHOD(int, get_created_time, (std::string path), (override));
        MOCK_METHOD(std::string, join, (std::vector<std::string> paths), (override));
        MOCK_METHOD(bool, is_valid_path, (std::string path), (override));
    };
}

#endif