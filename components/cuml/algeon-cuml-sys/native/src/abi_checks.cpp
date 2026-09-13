#include "cuml_rust/cuml_rust.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(sizeof(cuml_status_t) == sizeof(std::int32_t));
static_assert(CUML_SUCCESS == 0 && CUML_ERROR_NULL == 1 && CUML_ERROR_INVALID_ARGUMENT == 2);
static_assert(CUML_ERROR_INVALID_STATE == 3 && CUML_ERROR_OUT_OF_MEMORY == 4);
static_assert(CUML_ERROR_HOST_OUT_OF_MEMORY == 5 && CUML_ERROR_CUDA == 6);
static_assert(CUML_ERROR_FATAL_CUDA == 7 && CUML_ERROR_SIZE_OVERFLOW == 8);
static_assert(CUML_ERROR_UNSUPPORTED == 9 && CUML_ERROR_UNKNOWN == 10);
static_assert(sizeof(cuml_error_detail_t::message) == CUML_ERROR_MESSAGE_CAPACITY);
static_assert(std::is_pointer_v<cuml_handle_t>);
