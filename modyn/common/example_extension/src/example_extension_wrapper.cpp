#include "example_extension_wrapper.hpp"

#include <cstdint>

#include "example_extension.hpp"

extern "C" {
uint64_t sum_list(const uint64_t* list, const uint64_t list_len) {
  return modyn::common::example_extension::sum_list_impl(list, list_len);
}
}
