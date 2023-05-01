#include "LocalFileSystemWrapper.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>

using namespace storage;

std::vector<unsigned char> *LocalFileSystemWrapper::get(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    if (not this->is_file(path))
    {
        throw std::runtime_error("Path " + path + " is a directory.");
    }
    std::ifstream file;
    file.open(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    int size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> *buffer = new std::vector<unsigned char>(size);
    file.read((char *)buffer->data(), size);
    file.close();
    return buffer;
}

bool LocalFileSystemWrapper::exists(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    std::ifstream file;
    file.open(path);
    bool exists = file.good();
    file.close();
    return exists;
}

std::vector<std::string> *LocalFileSystemWrapper::list(std::string path, bool recursive)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    if (not this->is_directory(path))
    {
        throw std::runtime_error("Path " + path + " is a file.");
    }
    std::vector<std::string> *files = new std::vector<std::string>();
    std::vector<std::string> *directories = new std::vector<std::string>();
    std::vector<std::string> *paths = new std::vector<std::string>();
    paths->push_back(path);
    while (paths->size() > 0)
    {
        std::string current_path = paths->back();
        paths->pop_back();
        std::vector<std::string> *current_files = new std::vector<std::string>();
        std::vector<std::string> *current_directories = new std::vector<std::string>();
        for (const auto &entry : std::filesystem::directory_iterator(current_path))
        {
            std::string entry_path = entry.path();
            if (std::filesystem::is_directory(entry_path))
            {
                current_directories->push_back(entry_path);
            }
            else
            {
                current_files->push_back(entry_path);
            }
        }
        if (recursive)
        {
            paths->insert(paths->end(), current_directories->begin(), current_directories->end());
        }
        files->insert(files->end(), current_files->begin(), current_files->end());
        directories->insert(directories->end(), current_directories->begin(), current_directories->end());
        delete current_files;
        delete current_directories;
    }
    delete paths;
    delete directories;
    return files;
}

bool LocalFileSystemWrapper::is_directory(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    return std::filesystem::is_directory(path);
}

bool LocalFileSystemWrapper::is_file(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    return std::filesystem::is_regular_file(path);
}

int LocalFileSystemWrapper::get_file_size(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    if (not this->is_file(path))
    {
        throw std::runtime_error("Path " + path + " is a directory.");
    }
    std::ifstream file;
    file.open(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    int size = file.tellg();
    file.close();
    return size;
}

int LocalFileSystemWrapper::get_modified_time(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    if (not this->exists(path))
    {
        throw std::runtime_error("Path " + path + " does not exist.");
    }
    return std::filesystem::last_write_time(path).time_since_epoch().count();
}

int LocalFileSystemWrapper::get_created_time(std::string path)
{
    if (not this->is_valid_path(path))
    {
        throw std::invalid_argument("Path " + path + " is not valid.");
    }
    if (not this->exists(path))
    {
        throw std::runtime_error("Path " + path + " does not exist.");
    }
    return std::filesystem::last_write_time(path).time_since_epoch().count();
}

bool LocalFileSystemWrapper::is_valid_path(std::string path)
{
    return path.find("..") == std::string::npos;
}

std::string LocalFileSystemWrapper::join(std::vector<std::string> paths)
{
    std::string joined_path = "";
    for (int i = 0; i < paths.size(); i++)
    {
        joined_path += paths[i];
        if (i < paths.size() - 1)
        {
            joined_path += "/";
        }
    }
    return joined_path;
}
