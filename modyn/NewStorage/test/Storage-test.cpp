#include <gtest/gtest.h>
#include "../src/Storage.h"
#include "Utils.h"

TEST(StorageTest, TestStorage)
{
    create_dummy_yaml();
    std::string config_file = "config.yaml";
    storage::Storage storage(config_file);
    storage.run();
    delete_dummy_yaml();
}