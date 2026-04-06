#include "column.h"
#include "execution_context_internal.hpp"
#include "shim_common.h"
#include "table.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

cudf::size_type require_non_negative_i32(int32_t value, char const* field) {
  if (value < 0) { throw cudf::logic_error(field); }
  return static_cast<cudf::size_type>(value);
}

cudf::size_type require_size_type_from_size_t(size_t value, char const* field) {
  if (value > static_cast<size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw cudf::logic_error(field);
  }
  return static_cast<cudf::size_type>(value);
}

cudf::data_type interop_data_type(cudf_interop_semantic_column_t const& column) {
  auto const id = static_cast<cudf::type_id>(column.type_id);
  switch (id) {
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return cudf::data_type{id, column.scale};
    default: return cudf::data_type{id};
  }
}

using column_index_t = std::unordered_map<uint32_t, cudf_interop_semantic_column_t const*>;
using binding_index_t = std::unordered_map<uint32_t, cudf_interop_buffer_binding_t const*>;

column_index_t build_column_index(
    cudf_interop_semantic_column_t const* columns,
    int32_t num_columns) {
  column_index_t index;
  index.reserve(static_cast<size_t>(num_columns));
  for (int32_t i = 0; i < num_columns; ++i) { index.emplace(columns[i].column_id, &columns[i]); }
  return index;
}

binding_index_t build_binding_index(
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings) {
  binding_index_t index;
  index.reserve(static_cast<size_t>(num_bindings));
  for (int32_t i = 0; i < num_bindings; ++i) { index.emplace(bindings[i].buffer_id, &bindings[i]); }
  return index;
}

cudf_interop_buffer_binding_t const* find_binding(
    binding_index_t const& index,
    uint32_t id) {
  auto const it = index.find(id);
  return it != index.end() ? it->second : nullptr;
}

void const* require_device_binding_ptr(
    binding_index_t const& index,
    uint32_t id,
    char const* missing_message,
    char const* placement_message) {
  auto const* binding = find_binding(index, id);
  if (binding == nullptr) { throw cudf::logic_error(missing_message); }
  if (binding->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return binding->data;
}

size_t require_device_binding_size(
    binding_index_t const& index,
    uint32_t id,
    char const* missing_message,
    char const* placement_message) {
  auto const* binding = find_binding(index, id);
  if (binding == nullptr) { throw cudf::logic_error(missing_message); }
  if (binding->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return binding->byte_len;
}

cudf::column_view build_prepared_column_view(
    uint32_t root_column_id,
    column_index_t const& col_index,
    binding_index_t const& bind_index) {
  auto const it = col_index.find(root_column_id);
  CUDF_EXPECTS(it != col_index.end(), "prepared import referenced a missing semantic column");
  auto const* column = it->second;

  auto const dtype      = interop_data_type(*column);
  auto const size       = require_non_negative_i32(column->len, "prepared import column length must be >= 0");
  auto const offset     = require_non_negative_i32(column->offset, "prepared import column offset must be >= 0");
  auto const null_count = column->null_count >= 0 ? static_cast<cudf::size_type>(column->null_count)
                                                  : static_cast<cudf::size_type>(-1);

  cudf::bitmask_type const* null_mask = nullptr;
  if (column->validity_kind == CUDF_INTEROP_VALIDITY_MASK) {
    null_mask = static_cast<cudf::bitmask_type const*>(require_device_binding_ptr(
        bind_index,
        column->validity_buffer_id,
        "prepared import referenced a missing validity binding",
        "prepared device import requires CUDA-device validity bindings"));
  } else {
    CUDF_EXPECTS(column->validity_kind == CUDF_INTEROP_VALIDITY_NONE, "prepared import validity kind is unsupported");
  }

  switch (column->layout_kind) {
    case CUDF_INTEROP_LAYOUT_FIXED_WIDTH: {
      auto const values = require_device_binding_ptr(
          bind_index,
          column->values_buffer_id,
          "prepared import referenced a missing fixed-width values binding",
          "prepared device import requires CUDA-device fixed-width value bindings");
      return cudf::column_view(dtype, size, values, null_mask, null_count, offset);
    }

    case CUDF_INTEROP_LAYOUT_BINARY_LIKE: {
      auto const offsets_ptr = require_device_binding_ptr(
          bind_index,
          column->offsets_buffer_id,
          "prepared import referenced a missing binary-like offsets binding",
          "prepared device import requires CUDA-device binary-like offsets bindings");
      auto const offsets_size_bytes = require_device_binding_size(
          bind_index,
          column->offsets_buffer_id,
          "prepared import referenced a missing binary-like offsets binding",
          "prepared device import requires CUDA-device binary-like offsets bindings");
      CUDF_EXPECTS(
          offsets_size_bytes % sizeof(int32_t) == 0,
          "prepared import binary-like offsets binding must be 32-bit aligned");

      auto const chars_ptr = require_device_binding_ptr(
          bind_index,
          column->data_buffer_id,
          "prepared import referenced a missing binary-like chars binding",
          "prepared device import requires CUDA-device binary-like chars bindings");
      auto const chars_size_bytes = require_device_binding_size(
          bind_index,
          column->data_buffer_id,
          "prepared import referenced a missing binary-like chars binding",
          "prepared device import requires CUDA-device binary-like chars bindings");

      std::vector<cudf::column_view> children;
      children.reserve(1);
      children.emplace_back(
          cudf::data_type{cudf::type_id::INT32},
          require_size_type_from_size_t(
              offsets_size_bytes / sizeof(int32_t),
              "prepared import binary-like offsets binding exceeds cudf::size_type"),
          offsets_ptr,
          nullptr,
          0,
          0);
      require_size_type_from_size_t(
          chars_size_bytes,
          "prepared import binary-like chars binding exceeds cudf::size_type");
      return cudf::column_view(dtype, size, chars_ptr, null_mask, null_count, offset, children);
    }

    case CUDF_INTEROP_LAYOUT_LIST: {
      auto const offsets_ptr = require_device_binding_ptr(
          bind_index,
          column->offsets_buffer_id,
          "prepared import referenced a missing list offsets binding",
          "prepared device import requires CUDA-device list offsets bindings");
      auto const offsets_size_bytes = require_device_binding_size(
          bind_index,
          column->offsets_buffer_id,
          "prepared import referenced a missing list offsets binding",
          "prepared device import requires CUDA-device list offsets bindings");
      CUDF_EXPECTS(
          offsets_size_bytes % sizeof(int32_t) == 0,
          "prepared import list offsets binding must be 32-bit aligned");

      std::vector<cudf::column_view> children;
      children.reserve(2);
      children.emplace_back(
          cudf::data_type{cudf::type_id::INT32},
          require_size_type_from_size_t(
              offsets_size_bytes / sizeof(int32_t),
              "prepared import list offsets binding exceeds cudf::size_type"),
          offsets_ptr,
          nullptr,
          0,
          0);
      children.emplace_back(build_prepared_column_view(
          column->child_column_id, col_index, bind_index));
      return cudf::column_view(dtype, size, nullptr, null_mask, null_count, offset, children);
    }

    case CUDF_INTEROP_LAYOUT_STRUCT: {
      CUDF_EXPECTS(column->num_child_column_ids > 0, "prepared import struct layout must reference at least one child");
      CUDF_EXPECTS(column->child_column_ids != nullptr, "prepared import struct child ids pointer is null");
      std::vector<cudf::column_view> children;
      children.reserve(static_cast<size_t>(column->num_child_column_ids));
      for (int32_t index = 0; index < column->num_child_column_ids; ++index) {
        children.emplace_back(build_prepared_column_view(
            column->child_column_ids[index], col_index, bind_index));
      }
      return cudf::column_view(dtype, size, nullptr, null_mask, null_count, offset, children);
    }

    case CUDF_INTEROP_LAYOUT_DICTIONARY: {
      auto const indices_ptr = require_device_binding_ptr(
          bind_index,
          column->indices_buffer_id,
          "prepared import referenced a missing dictionary indices binding",
          "prepared device import requires CUDA-device dictionary indices bindings");
      auto const indices_size_bytes = require_device_binding_size(
          bind_index,
          column->indices_buffer_id,
          "prepared import referenced a missing dictionary indices binding",
          "prepared device import requires CUDA-device dictionary indices bindings");
      CUDF_EXPECTS(
          indices_size_bytes % sizeof(int32_t) == 0,
          "prepared import dictionary indices binding must be 32-bit aligned");

      std::vector<cudf::column_view> children;
      children.reserve(2);
      children.emplace_back(
          cudf::data_type{cudf::type_id::INT32},
          require_size_type_from_size_t(
              indices_size_bytes / sizeof(int32_t),
              "prepared import dictionary indices binding exceeds cudf::size_type"),
          indices_ptr,
          nullptr,
          0,
          0);
      children.emplace_back(build_prepared_column_view(
          column->keys_column_id, col_index, bind_index));
      return cudf::column_view(dtype, size, nullptr, null_mask, null_count, offset, children);
    }
  }

  CUDF_FAIL("prepared import layout kind is unsupported");
}

}  // namespace

extern "C" {

cudf_status_t cudf_column_copy_from_prepared_device_export(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings,
    cudf_column_handle* out) {
  return cudf_column_copy_from_prepared_device_export_ex(
      root_column_id, semantic_columns, num_semantic_columns, bindings, num_bindings, nullptr, out);
}

cudf_status_t cudf_column_copy_from_prepared_device_export_ex(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings,
    cudf_exec_options_t const* options,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(semantic_columns, "semantic_columns is null");
  CUDF_RUST_NULL_CHECK(bindings, "bindings is null");
  if (num_semantic_columns <= 0) {
    set_error("num_semantic_columns must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_bindings < 0) {
    set_error("num_bindings must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const col_index = build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    auto const view = build_prepared_column_view(root_column_id, col_index, bind_index);
    auto copy = std::make_unique<cudf::column>(view, exec.stream, exec.mr);
    *out      = static_cast<cudf_column_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy_from_prepared_device_export(
    cudf_interop_semantic_table_t const* table,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings,
    cudf_table_handle* out) {
  return cudf_table_copy_from_prepared_device_export_ex(
      table, semantic_columns, num_semantic_columns, bindings, num_bindings, nullptr, out);
}

cudf_status_t cudf_table_copy_from_prepared_device_export_ex(
    cudf_interop_semantic_table_t const* table,
    cudf_interop_semantic_column_t const* semantic_columns,
    int32_t num_semantic_columns,
    cudf_interop_buffer_binding_t const* bindings,
    int32_t num_bindings,
    cudf_exec_options_t const* options,
    cudf_table_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(semantic_columns, "semantic_columns is null");
  CUDF_RUST_NULL_CHECK(bindings, "bindings is null");
  if (num_semantic_columns < 0) {
    set_error("num_semantic_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_bindings < 0) {
    set_error("num_bindings must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (table->num_rows < 0) {
    set_error("table num_rows must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (table->num_columns < 0) {
    set_error("table num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (table->num_columns > 0 && table->column_ids == nullptr) {
    set_error("table column_ids is null");
    return CUDF_ERROR_NULL;
  }
  CUDF_RUST_TRY {
    cudf_exec_context exec{cudf::get_default_stream(),
                           cudf::get_current_device_resource_ref(),
                           nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const col_index = build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    std::vector<cudf::column_view> columns;
    columns.reserve(static_cast<size_t>(table->num_columns));
    for (int32_t index = 0; index < table->num_columns; ++index) {
      columns.emplace_back(build_prepared_column_view(
          table->column_ids[index], col_index, bind_index));
    }

    auto copy = std::make_unique<cudf::table>(cudf::table_view(columns), exec.stream, exec.mr);
    *out      = static_cast<cudf_table_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
