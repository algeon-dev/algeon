#include "execution_context_internal.hpp"
#include "shim_common.h"
#include "cudf_rust/validation.h"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace {
template <typename T>
struct column_non_finite_count_transform {
  cudf::column_device_view input;

  __device__ int32_t operator()(cudf::size_type index) const
  {
    if (input.is_null(index)) { return 1; }
    return cuda::std::isfinite(input.element<T>(index)) ? 0 : 1;
  }
};

template <typename T>
struct buffer_non_finite_count_transform {
  T const* input;

  __device__ int32_t operator()(cudf::size_type index) const
  {
    return cuda::std::isfinite(input[index]) ? 0 : 1;
  }
};

__global__ void float32_buffer_add_scalar(float* values, std::size_t element_count, float scalar)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count) { values[index] += scalar; }
}

__global__ void int64_buffer_fill_where_float32_infinite(int64_t* values,
                                                         float const* paired_values,
                                                         std::size_t element_count,
                                                         int64_t fill_value)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count && cuda::std::isinf(paired_values[index])) { values[index] = fill_value; }
}

__global__ void int64_float32_buffer_replace_value_and_set_paired_infinite(int64_t* values,
                                                                            float* paired_values,
                                                                            std::size_t element_count,
                                                                            int64_t from,
                                                                            int64_t to)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < element_count && values[index] == from) {
    values[index]        = to;
    paired_values[index] = cuda::std::numeric_limits<float>::infinity();
  }
}

struct float32_buffer_non_finite_unless_int64_equals_transform {
  float const* values;
  int64_t const* paired_values;
  int64_t excluded_value;

  __device__ int32_t operator()(cudf::size_type index) const
  {
    return paired_values[index] == excluded_value || cuda::std::isfinite(values[index]) ? 0 : 1;
  }
};

struct int64_buffer_out_of_range_count_transform {
  int64_t const* input;
  int64_t min_inclusive;
  int64_t max_exclusive;
  int64_t sentinel;
  bool allow_sentinel;

  __device__ int32_t operator()(cudf::size_type index) const
  {
    auto const value = input[index];
    return (value >= min_inclusive && value < max_exclusive) || (allow_sentinel && value == sentinel) ? 0 : 1;
  }
};

template <typename Transform>
int32_t count_invalid_values(cudf::size_type element_count,
                             Transform transform,
                             cudf_exec_context const& exec)
{
  auto result      = cudf::detail::device_scalar<int32_t>(int32_t{0}, exec.stream, *exec.mr);
  auto const begin = cuda::counting_iterator<cudf::size_type>{0};

  std::size_t temporary_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(nullptr,
                                                   temporary_bytes,
                                                   begin,
                                                   result.data(),
                                                   element_count,
                                                   cuda::std::plus<int32_t>{},
                                                   transform,
                                                   int32_t{0},
                                                   exec.stream.value()));
  rmm::device_buffer temporary(temporary_bytes, exec.stream, *exec.mr);
  CUDF_CUDA_TRY(cub::DeviceReduce::TransformReduce(temporary.data(),
                                                   temporary_bytes,
                                                   begin,
                                                   result.data(),
                                                   element_count,
                                                   cuda::std::plus<int32_t>{},
                                                   transform,
                                                   int32_t{0},
                                                   exec.stream.value()));

  return result.value(exec.stream);
}

cudf_status_t validate_raw_element_count(std::size_t element_count)
{
  if (element_count > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    set_error("raw validation element count exceeds cudf size_type maximum");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  return CUDF_SUCCESS;
}
}  // namespace

extern "C" {
cudf_status_t cudf_float32_non_finite_count_ex(cudf_column_view_handle input,
                                               int32_t* out_non_finite_count,
                                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_non_finite_count, "out_non_finite_count is null");
  *out_non_finite_count = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const* view = reinterpret_cast<cudf::column_view const*>(input);
    if (view->type().id() != cudf::type_id::FLOAT32) {
      set_error("finite validation requires FLOAT32 input");
      return CUDF_ERROR_TYPE;
    }

    auto device_view = cudf::column_device_view::create_with_mr(*view, exec->stream, *exec->mr);
    // The factory's owning pointer is host-resident; copy the device view into
    // the kernel transform. Device code does not receive that pointer.
    auto const transform = column_non_finite_count_transform<float>{*device_view};
    *out_non_finite_count = count_invalid_values(view->size(), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_float64_non_finite_count_ex(cudf_column_view_handle input,
                                               int32_t* out_non_finite_count,
                                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_non_finite_count, "out_non_finite_count is null");
  *out_non_finite_count = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const* view = reinterpret_cast<cudf::column_view const*>(input);
    if (view->type().id() != cudf::type_id::FLOAT64) {
      set_error("finite validation requires FLOAT64 input");
      return CUDF_ERROR_TYPE;
    }
    auto device_view = cudf::column_device_view::create_with_mr(*view, exec->stream, *exec->mr);
    auto const transform = column_non_finite_count_transform<double>{*device_view};
    *out_non_finite_count = count_invalid_values(view->size(), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_float32_buffer_non_finite_count_ex(float const* input,
                                                      std::size_t element_count,
                                                      int32_t* out_non_finite_count,
                                                      cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_non_finite_count, "out_non_finite_count is null");
  *out_non_finite_count = 0;
  if (input == nullptr && element_count != 0) {
    set_error("input is null");
    return CUDF_ERROR_NULL;
  }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const transform = buffer_non_finite_count_transform<float>{input};
    *out_non_finite_count =
      count_invalid_values(static_cast<cudf::size_type>(element_count), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_float64_buffer_non_finite_count_ex(double const* input,
                                                      std::size_t element_count,
                                                      int32_t* out_non_finite_count,
                                                      cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_non_finite_count, "out_non_finite_count is null");
  *out_non_finite_count = 0;
  if (input == nullptr && element_count != 0) {
    set_error("input is null");
    return CUDF_ERROR_NULL;
  }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    auto const transform = buffer_non_finite_count_transform<double>{input};
    *out_non_finite_count = count_invalid_values(static_cast<cudf::size_type>(element_count), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_int64_buffer_out_of_range_count_ex(int64_t const* input,
                                                      std::size_t element_count,
                                                      int64_t min_inclusive,
                                                      int64_t max_exclusive,
                                                      int32_t* out_of_range_count,
                                                      cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_of_range_count, "out_of_range_count is null");
  *out_of_range_count = 0;
  if (input == nullptr && element_count != 0) {
    set_error("input is null");
    return CUDF_ERROR_NULL;
  }
  if (min_inclusive >= max_exclusive) {
    set_error("INT64 validation requires min_inclusive < max_exclusive");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const transform = int64_buffer_out_of_range_count_transform{input, min_inclusive, max_exclusive, 0, false};
    *out_of_range_count = count_invalid_values(static_cast<cudf::size_type>(element_count), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
cudf_status_t cudf_float32_buffer_add_scalar_ex(float* values,
                                                 std::size_t element_count,
                                                 float scalar,
                                                 cudf_exec_options_t const* options)
{
  if (values == nullptr && element_count != 0) { set_error("input is null"); return CUDF_ERROR_NULL; }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  if (!cuda::std::isfinite(scalar)) { set_error("scalar must be finite"); return CUDF_ERROR_INVALID_OPERATION; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    if (element_count != 0) {
      constexpr std::size_t threads = 256;
      auto const blocks = static_cast<std::uint32_t>((element_count + threads - 1) / threads);
      float32_buffer_add_scalar<<<blocks, threads, 0, exec->stream.value()>>>(values, element_count, scalar);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_int64_buffer_fill_where_float32_infinite_ex(int64_t* values,
                                                                float const* paired_values,
                                                                std::size_t element_count,
                                                                int64_t fill_value,
                                                                cudf_exec_options_t const* options)
{
  if ((values == nullptr || paired_values == nullptr) && element_count != 0) { set_error("input is null"); return CUDF_ERROR_NULL; }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    if (element_count != 0) {
      constexpr std::size_t threads = 256;
      auto const blocks = static_cast<std::uint32_t>((element_count + threads - 1) / threads);
      int64_buffer_fill_where_float32_infinite<<<blocks, threads, 0, exec->stream.value()>>>(values, paired_values, element_count, fill_value);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_int64_float32_buffer_replace_value_and_set_paired_infinite_ex(int64_t* values,
                                                                                 float* paired_values,
                                                                                 std::size_t element_count,
                                                                                 int64_t from,
                                                                                 int64_t to,
                                                                                 cudf_exec_options_t const* options)
{
  if ((values == nullptr || paired_values == nullptr) && element_count != 0) {
    set_error("input is null");
    return CUDF_ERROR_NULL;
  }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    if (element_count != 0) {
      constexpr std::size_t threads = 256;
      auto const blocks = static_cast<std::uint32_t>((element_count + threads - 1) / threads);
      int64_float32_buffer_replace_value_and_set_paired_infinite<<<blocks, threads, 0, exec->stream.value()>>>(
        values, paired_values, element_count, from, to);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_float32_buffer_non_finite_unless_int64_equals_count_ex(float const* values,
                                                                          int64_t const* paired_values,
                                                                          std::size_t element_count,
                                                                          int64_t excluded_value,
                                                                          int32_t* out_non_finite_count,
                                                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_non_finite_count, "out_non_finite_count is null");
  *out_non_finite_count = 0;
  if ((values == nullptr || paired_values == nullptr) && element_count != 0) { set_error("input is null"); return CUDF_ERROR_NULL; }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    auto const transform = float32_buffer_non_finite_unless_int64_equals_transform{values, paired_values, excluded_value};
    *out_non_finite_count = count_invalid_values(static_cast<cudf::size_type>(element_count), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_int64_buffer_out_of_range_or_sentinel_count_ex(int64_t const* input,
                                                                   std::size_t element_count,
                                                                   int64_t min_inclusive,
                                                                   int64_t max_exclusive,
                                                                   int64_t sentinel,
                                                                   int32_t* out_of_range_count,
                                                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out_of_range_count, "out_of_range_count is null");
  *out_of_range_count = 0;
  if (input == nullptr && element_count != 0) { set_error("input is null"); return CUDF_ERROR_NULL; }
  if (min_inclusive >= max_exclusive) { set_error("INT64 validation requires min_inclusive < max_exclusive"); return CUDF_ERROR_INVALID_OPERATION; }
  auto const size_status = validate_raw_element_count(element_count);
  if (size_status != CUDF_SUCCESS) { return size_status; }
  CUDF_RUST_TRY
  {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    auto const transform = int64_buffer_out_of_range_count_transform{input, min_inclusive, max_exclusive, sentinel, true};
    *out_of_range_count = count_invalid_values(static_cast<cudf::size_type>(element_count), transform, *exec);
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
