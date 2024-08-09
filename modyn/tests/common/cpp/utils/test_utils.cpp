#include <gtest/gtest.h>
#include <stdlib.h>

#include "modyn/utils/utils.hpp"

TEST(CommonUtilsTest, TestIsPowerOfTwo) {
  ASSERT_TRUE(::modyn::utils::is_power_of_two(1));
  ASSERT_TRUE(::modyn::utils::is_power_of_two(2));
  ASSERT_TRUE(::modyn::utils::is_power_of_two(4));
  ASSERT_TRUE(::modyn::utils::is_power_of_two(8));
  ASSERT_TRUE(::modyn::utils::is_power_of_two(16));

  ASSERT_FALSE(::modyn::utils::is_power_of_two(0));
  ASSERT_FALSE(::modyn::utils::is_power_of_two(3));
  ASSERT_FALSE(::modyn::utils::is_power_of_two(5));
  ASSERT_FALSE(::modyn::utils::is_power_of_two(6));
}
