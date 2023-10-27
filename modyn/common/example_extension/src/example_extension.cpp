#include "example_extension.hpp"

#include <cstdint>
#include <numeric>
#include <vector>

#include "modyn/utils/utils.hpp"

namespace modyn::common::example_extension {
uint64_t sum_list_impl(const uint64_t* list, const uint64_t list_len) {
  ::modyn::utils::is_power_of_two(1);
  return std::accumulate(list, list + list_len, static_cast<uint64_t>(0));
}
}  // namespace modyn::common::example_extension