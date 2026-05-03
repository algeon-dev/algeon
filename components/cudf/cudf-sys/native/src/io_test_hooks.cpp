#include "io_test_hooks.hpp"

#include "shim_common.h"

#include <new>

namespace {
thread_local cudf_test_io_metadata_fault_t g_next_metadata_fault = CUDF_TEST_IO_METADATA_FAULT_NONE;

char** allocate_name_array(int32_t count)
{
  auto** names = new (std::nothrow) char*[count];
  if (names == nullptr) { return nullptr; }
  for (int32_t i = 0; i < count; ++i) {
    names[i] = nullptr;
  }
  return names;
}
}  // namespace

extern "C" {
void cudf_test_io_set_next_metadata_fault(cudf_test_io_metadata_fault_t fault)
{
  g_next_metadata_fault = fault;
}

void cudf_test_io_reset_metadata_fault()
{
  g_next_metadata_fault = CUDF_TEST_IO_METADATA_FAULT_NONE;
}
}  // extern "C"

cudf_test_io_metadata_fault_t cudf_test_io_take_next_metadata_fault()
{
  auto const fault      = g_next_metadata_fault;
  g_next_metadata_fault = CUDF_TEST_IO_METADATA_FAULT_NONE;
  return fault;
}

cudf_status_t cudf_test_io_apply_metadata_fault(cudf_test_io_metadata_fault_t fault,
                                                char*** out_names,
                                                int32_t* out_num_names)
{
  switch (fault) {
    case CUDF_TEST_IO_METADATA_FAULT_NONE: return CUDF_SUCCESS;
    case CUDF_TEST_IO_METADATA_FAULT_NEGATIVE_NAME_COUNT:
      *out_names     = nullptr;
      *out_num_names = -1;
      return CUDF_SUCCESS;
    case CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ARRAY:
      *out_names     = nullptr;
      *out_num_names = 1;
      return CUDF_SUCCESS;
    case CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ENTRY: {
      auto** names = allocate_name_array(1);
      if (names == nullptr) {
        set_error("failed to allocate test metadata name array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      *out_names     = names;
      *out_num_names = 1;
      return CUDF_SUCCESS;
    }
    case CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME: {
      auto** names = allocate_name_array(1);
      if (names == nullptr) {
        set_error("failed to allocate test metadata name array");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      names[0] = new (std::nothrow) char[2];
      if (names[0] == nullptr) {
        delete[] names;
        set_error("failed to allocate test metadata name string");
        return CUDF_ERROR_OUT_OF_MEMORY;
      }
      names[0][0]    = static_cast<char>(0xFF);
      names[0][1]    = '\0';
      *out_names     = names;
      *out_num_names = 1;
      return CUDF_SUCCESS;
    }
    case CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT:
      *out_names     = nullptr;
      *out_num_names = 0;
      return CUDF_SUCCESS;
  }

  set_error("unknown test metadata fault");
  return CUDF_ERROR_INVALID_ARGUMENT;
}
