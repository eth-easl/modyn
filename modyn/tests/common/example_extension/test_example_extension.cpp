#include <gtest/gtest.h>

#include "example_extension.hpp"

using namespace modyn::common::example_extension;

TEST(ExampleExtensionTest, TestSumList) {
  std::vector<uint64_t> empty_list = {};
  EXPECT_EQ(sum_list_impl(empty_list.data(), 0), 0);
  std::vector<uint64_t> one_list = {1};
  EXPECT_EQ(sum_list_impl(one_list.data(), 1), 1);
  std::vector<uint64_t> long_list = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(sum_list_impl(long_list.data(), 10), 55);
}
