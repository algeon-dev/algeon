#include "cudf_rust/copying.h"

#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>
#include <memory>
#include <vector>

extern "C" {
cudf_status_t cudf_gather_fixed_width_dont_check_preflight(
    int64_t output_rows, uint64_t output_data_bytes_upper_bound,
    cudf_gather_fixed_width_column_metadata_t const *columns,
    int32_t num_columns, int32_t device,
    cudf_gather_fixed_width_dont_check_preflight_result_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = {};
  if (num_columns < 0) {
    set_error("num_columns must be non-negative");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0 && columns == nullptr) {
    set_error("columns is null");
    return CUDF_ERROR_NULL;
  }

  CUDF_RUST_TRY {
    if (output_data_bytes_upper_bound >
        std::numeric_limits<std::size_t>::max()) {
      set_error("output_data_bytes_upper_bound exceeds size_t");
      return CUDF_ERROR_SIZE_OVERFLOW;
    }
    std::vector<cudf::gather_fixed_width_column_metadata> metadata;
    metadata.reserve(static_cast<std::size_t>(num_columns));
    for (int32_t i = 0; i < num_columns; ++i) {
      auto const &column = columns[i];
      if (column.nullable != 0 && column.nullable != 1) {
        set_error("column nullable must be 0 or 1");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      if (column.num_children < 0) {
        set_error("column num_children must be non-negative");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      auto const raw_type_id = static_cast<int32_t>(column.type_id);
      if (raw_type_id < static_cast<int32_t>(cudf::type_id::EMPTY) ||
          raw_type_id >= static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS)) {
        set_error("column type_id is invalid");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      metadata.push_back({cudf::data_type(to_cudf_type_id(column.type_id)),
                          column.nullable != 0, column.num_children});
    }

    auto const result = cudf::gather_fixed_width_dont_check_preflight(
        output_rows, metadata,
        static_cast<std::size_t>(output_data_bytes_upper_bound), device);
    out->gather_map_bytes = static_cast<uint64_t>(result.gather_map_bytes);
    out->output_data_bytes = static_cast<uint64_t>(result.output_data_bytes);
    out->output_null_mask_bytes =
        static_cast<uint64_t>(result.output_null_mask_bytes);
    out->target_mask_pointer_array_bytes =
        static_cast<uint64_t>(result.target_mask_pointer_array_bytes);
    out->source_table_device_view_bytes =
        static_cast<uint64_t>(result.source_table_device_view_bytes);
    out->valid_count_array_bytes =
        static_cast<uint64_t>(result.valid_count_array_bytes);
    out->native_temporary_workspace_bytes =
        static_cast<uint64_t>(result.native_temporary_workspace_bytes);
    out->active_phase_peak_bytes =
        static_cast<uint64_t>(result.active_phase_peak_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_gather_ex(cudf_table_view_handle input,
                             cudf_column_view_handle gather_map,
                             cudf_out_of_bounds_policy_t bounds_policy,
                             cudf_table_handle *out,
                             cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(gather_map, "gather_map is null");

  cudf::out_of_bounds_policy policy;
  switch (bounds_policy) {
  case CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY:
    policy = cudf::out_of_bounds_policy::NULLIFY;
    break;
  case CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK:
    policy = cudf::out_of_bounds_policy::DONT_CHECK;
    break;
  default:
    set_error("invalid bounds_policy");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY {
    auto *view = reinterpret_cast<cudf::table_view *>(input);
    auto *map_view = reinterpret_cast<cudf::column_view *>(gather_map);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto result =
          cudf::gather(*view, *map_view, policy, exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_gather_device_allocated_size_ex(
    cudf_table_view_handle input, cudf_column_view_handle gather_map,
    cudf_out_of_bounds_policy_t bounds_policy, size_t *out,
    cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(gather_map, "gather_map is null");

  cudf::out_of_bounds_policy policy;
  switch (bounds_policy) {
  case CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY:
    policy = cudf::out_of_bounds_policy::NULLIFY;
    break;
  case CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK:
    policy = cudf::out_of_bounds_policy::DONT_CHECK;
    break;
  default:
    set_error("invalid bounds_policy");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY {
    auto *view = reinterpret_cast<cudf::table_view *>(input);
    auto *map_view = reinterpret_cast<cudf::column_view *>(gather_map);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto result =
          cudf::gather(*view, *map_view, policy, exec->stream, *exec->mr);
      *out = result->alloc_size();
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scatter_ex(cudf_table_view_handle source,
                              cudf_column_view_handle scatter_map,
                              cudf_table_view_handle target,
                              cudf_table_handle *out,
                              cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(source, "source is null");
  CUDF_RUST_NULL_CHECK(scatter_map, "scatter_map is null");
  CUDF_RUST_NULL_CHECK(target, "target is null");

  CUDF_RUST_TRY {
    auto *source_view = reinterpret_cast<cudf::table_view *>(source);
    auto *map_view = reinterpret_cast<cudf::column_view *>(scatter_map);
    auto *target_view = reinterpret_cast<cudf::table_view *>(target);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto result = cudf::scatter(*source_view, *map_view, *target_view,
                                  exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(result),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_copy_if_else_vv_ex(cudf_column_view_handle lhs,
                                      cudf_column_view_handle rhs,
                                      cudf_column_view_handle boolean_mask,
                                      cudf_column_handle *out,
                                      cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(lhs, "lhs is null");
  CUDF_RUST_NULL_CHECK(rhs, "rhs is null");
  CUDF_RUST_NULL_CHECK(boolean_mask, "boolean_mask is null");
  CUDF_RUST_TRY {
    auto *lhs_view = reinterpret_cast<cudf::column_view *>(lhs);
    auto *rhs_view = reinterpret_cast<cudf::column_view *>(rhs);
    auto *mask_view = reinterpret_cast<cudf::column_view *>(boolean_mask);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto result = cudf::copy_if_else(*lhs_view, *rhs_view, *mask_view,
                                       exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_shift_ex(cudf_column_view_handle input, int32_t offset,
                            cudf_scalar_handle fill_value,
                            cudf_column_handle *out,
                            cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(input, "input is null");
  CUDF_RUST_NULL_CHECK(fill_value, "fill_value is null");

  CUDF_RUST_TRY {
    auto *input_view = reinterpret_cast<cudf::column_view *>(input);
    auto const &fill_scalar = cudf_scalar_from_handle_const(fill_value);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto result =
          cudf::shift(*input_view, static_cast<cudf::size_type>(offset),
                      fill_scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(result),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
