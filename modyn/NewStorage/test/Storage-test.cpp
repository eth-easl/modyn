#include <gtest/gtest.h>
#include "../src/Storage.hpp"
#include "Utils.hpp"

using namespace storage;

TEST(StorageTest, TestStorage)
{
    Utils::create_dummy_yaml();
    std::string config_file = "config.yaml";
    storage::Storage storage(config_file);
    storage.run();
    Utils::delete_dummy_yaml();
}
