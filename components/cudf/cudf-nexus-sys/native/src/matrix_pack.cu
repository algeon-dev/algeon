#include "execution_context_internal.hpp"
#include "cudf_rust/reshape.h"
#include "shim_common.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>

namespace {
__global__ void matrix_pack_column_major(float* output,
                                         float const* input,
                                         std::uint64_t element_count,
                                         std::uint64_t rows,
                                         std::uint64_t columns)
{
  auto const index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) { return; }
  auto const column = index / rows;
  auto const row    = index % rows;
  output[index]     = input[row * columns + column];
}

cudf_status_t pack_borrowed_float32_matrix(const float* input,
                                            int32_t rows,
                                            int32_t columns,
                                            int32_t column_major,
                                            cudf_exec_options_t const* options,
                                            cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (rows <= 0 || columns <= 0) {
    set_error("flat Float32 matrix rows and columns must both be positive");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (column_major != 0 && column_major != 1) {
    set_error("column_major must be either 0 or 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY
  {
    auto const expected = static_cast<int64_t>(rows) * static_cast<int64_t>(columns);
    if (expected > static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max())) {
      set_error("flat Float32 input length must fit cudf size_type");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const element_count = static_cast<cudf::size_type>(expected);
    auto output = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT32},
                                                element_count,
                                                cudf::mask_state::UNALLOCATED,
                                                exec->stream,
                                                *exec->mr);
    auto* output_data        = output->mutable_view().data<float>();
    auto const payload_bytes = static_cast<std::size_t>(element_count) * sizeof(float);
    if (column_major == 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        output_data, input, payload_bytes, cudaMemcpyDeviceToDevice, exec->stream.value()));
    } else {
      constexpr int32_t threads      = 256;
      auto const element_count_u64 = static_cast<std::uint64_t>(element_count);
      auto const blocks            = static_cast<std::uint32_t>((element_count_u64 + threads - 1) / threads);
      matrix_pack_column_major<<<blocks, threads, 0, exec->stream.value()>>>(
        output_data,
        input,
        element_count_u64,
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(columns));
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    *out = cudf_release_column_handle(std::move(output), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // namespace

extern "C" {
cudf_status_t cudf_pack_flat_float32_matrix_ex(cudf_column_view_handle input,
                                                int32_t rows,
                                                int32_t columns,
                                                int32_t column_major,
                                                cudf_exec_options_t const* options,
                                                cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  if (rows <= 0 || columns <= 0) {
    set_error("flat Float32 matrix rows and columns must both be positive");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (column_major != 0 && column_major != 1) {
    set_error("column_major must be either 0 or 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY
  {
    auto const* view = reinterpret_cast<cudf::column_view const*>(input);
    if (view->type().id() != cudf::type_id::FLOAT32) {
      set_error("flat matrix packing requires FLOAT32 input");
      return CUDF_ERROR_TYPE;
    }
    if (view->null_count() != 0) {
      set_error("flat matrix packing requires non-null input");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto const expected = static_cast<int64_t>(rows) * static_cast<int64_t>(columns);
    if (expected > static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()) ||
        static_cast<int64_t>(view->size()) != expected) {
      set_error("flat Float32 input length must equal rows times columns");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    return pack_borrowed_float32_matrix(view->data<float>(), rows, columns, column_major, options, out);
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_pack_borrowed_float32_matrix_ex(const void* input,
                                                    int32_t rows,
                                                    int32_t columns,
                                                    int32_t column_major,
                                                    cudf_exec_options_t const* options,
                                                    cudf_column_handle* out)
{
  return pack_borrowed_float32_matrix(
    static_cast<float const*>(input), rows, columns, column_major, options, out);
}
}  // extern "C"
