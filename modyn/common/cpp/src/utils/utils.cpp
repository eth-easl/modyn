#include "modyn/utils/utils.hpp"

#include <bit>

namespace modyn::utils {

bool is_power_of_two(uint64_t value) { return std::has_single_bit(value); }

}  // namespace modyn::utils