#include "internal/grpc/storage_service_impl.hpp"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <utime.h>

#include <cstdint>
#include <filesystem>
#include <fstream>

#include "gmock/gmock.h"
#include "test_utils.hpp"

using namespace storage;

class StorageServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(StorageServiceImplTest, TestGet) {}

TEST_F(StorageServiceImplTest, TestGetNewDataSince) {}

TEST_F(StorageServiceImplTest, TestGetDataInInterval) {}

TEST_F(StorageServiceImplTest, TestCheckAvailability) {}

TEST_F(StorageServiceImplTest, TestGetCurrentTimestamp) {}

TEST_F(StorageServiceImplTest, TestDeleteDataset) {}

TEST_F(StorageServiceImplTest, TestDeleteData) {}