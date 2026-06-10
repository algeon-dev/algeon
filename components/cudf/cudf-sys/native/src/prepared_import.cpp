#include "column.h"
#include "execution_context_internal.hpp"
#include "interop_common.hpp"
#include "owning_handle.hpp"
#include "shim_common.h"
#include "table.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cstring>
#include <memory>
#include <vector>

using cudf_interop::binding_index_t;
using cudf_interop::build_binding_index;
using cudf_interop::build_column_index;
using cudf_interop::column_index_t;
using cudf_interop::interop_data_type;
using cudf_interop::require_non_negative_i32;
using cudf_interop::require_size_type_from_size_t;

namespace {
cudf_interop_buffer_binding_t const *find_binding(binding_index_t const &index,
                                                  uint32_t id) {
  auto const it = index.find(id);
  return it != index.end() ? it->second : nullptr;
}

struct host_binding_view {
  void const *data;
  size_t byte_len;
};

host_binding_view require_host_binding(binding_index_t const &index,
                                       uint32_t id, char const *missing_message,
                                       char const *placement_message) {
  auto const *binding = find_binding(index, id);
  if (binding == nullptr) {
    throw cudf::logic_error(missing_message);
  }
  if (binding->placement_kind != CUDF_INTEROP_PLACEMENT_HOST) {
    throw cudf::logic_error(placement_message);
  }
  return host_binding_view{binding->data, binding->byte_len};
}

void const *require_host_binding_ptr(binding_index_t const &index, uint32_t id,
                                     char const *missing_message,
                                     char const *placement_message) {
  return require_host_binding(index, id, missing_message, placement_message)
      .data;
}

size_t require_host_binding_size(binding_index_t const &index, uint32_t id,
                                 char const *missing_message,
                                 char const *placement_message) {
  return require_host_binding(index, id, missing_message, placement_message)
      .byte_len;
}

void const *require_device_binding_ptr(binding_index_t const &index,
                                       uint32_t id, char const *missing_message,
                                       char const *placement_message) {
  auto const *binding = find_binding(index, id);
  if (binding == nullptr) {
    throw cudf::logic_error(missing_message);
  }
  if (binding->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return binding->data;
}

size_t require_device_binding_size(binding_index_t const &index, uint32_t id,
                                   char const *missing_message,
                                   char const *placement_message) {
  auto const *binding = find_binding(index, id);
  if (binding == nullptr) {
    throw cudf::logic_error(missing_message);
  }
  if (binding->placement_kind != CUDF_INTEROP_PLACEMENT_CUDA_DEVICE) {
    throw cudf::logic_error(placement_message);
  }
  return binding->byte_len;
}

cudf::column_view
build_prepared_column_view(uint32_t root_column_id,
                           column_index_t const &col_index,
                           binding_index_t const &bind_index) {
  auto const it = col_index.find(root_column_id);
  CUDF_EXPECTS(it != col_index.end(),
               "prepared import referenced a missing semantic column");
  auto const *column = it->second;

  auto const dtype = interop_data_type(*column);
  auto const size = require_non_negative_i32(
      column->len, "prepared import column length must be >= 0");
  auto const offset = require_non_negative_i32(
      column->offset, "prepared import column offset must be >= 0");
  auto const null_count = column->null_count >= 0
                              ? static_cast<cudf::size_type>(column->null_count)
                              : static_cast<cudf::size_type>(-1);

  cudf::bitmask_type const *null_mask = nullptr;
  if (column->validity_kind == CUDF_INTEROP_VALIDITY_MASK) {
    null_mask =
        static_cast<cudf::bitmask_type const *>(require_device_binding_ptr(
            bind_index, column->validity_buffer_id,
            "prepared import referenced a missing validity binding",
            "prepared device import requires CUDA-device validity bindings"));
  } else {
    CUDF_EXPECTS(column->validity_kind == CUDF_INTEROP_VALIDITY_NONE,
                 "prepared import validity kind is unsupported");
  }

  switch (column->layout_kind) {
  case CUDF_INTEROP_LAYOUT_FIXED_WIDTH: {
    auto const values = require_device_binding_ptr(
        bind_index, column->values_buffer_id,
        "prepared import referenced a missing fixed-width values binding",
        "prepared device import requires CUDA-device fixed-width value "
        "bindings");
    return cudf::column_view(dtype, size, values, null_mask, null_count,
                             offset);
  }

  case CUDF_INTEROP_LAYOUT_BINARY_LIKE: {
    auto const offsets_ptr = require_device_binding_ptr(
        bind_index, column->offsets_buffer_id,
        "prepared import referenced a missing binary-like offsets binding",
        "prepared device import requires CUDA-device binary-like offsets "
        "bindings");
    auto const offsets_size_bytes = require_device_binding_size(
        bind_index, column->offsets_buffer_id,
        "prepared import referenced a missing binary-like offsets binding",
        "prepared device import requires CUDA-device binary-like offsets "
        "bindings");
    CUDF_EXPECTS(
        offsets_size_bytes % sizeof(int32_t) == 0,
        "prepared import binary-like offsets binding must be 32-bit aligned");

    auto const chars_ptr = require_device_binding_ptr(
        bind_index, column->data_buffer_id,
        "prepared import referenced a missing binary-like chars binding",
        "prepared device import requires CUDA-device binary-like chars "
        "bindings");
    auto const chars_size_bytes = require_device_binding_size(
        bind_index, column->data_buffer_id,
        "prepared import referenced a missing binary-like chars binding",
        "prepared device import requires CUDA-device binary-like chars "
        "bindings");

    std::vector<cudf::column_view> children;
    children.reserve(1);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(offsets_size_bytes / sizeof(int32_t),
                                      "prepared import binary-like offsets "
                                      "binding exceeds cudf::size_type"),
        offsets_ptr, nullptr, 0, 0);
    require_size_type_from_size_t(
        chars_size_bytes,
        "prepared import binary-like chars binding exceeds cudf::size_type");
    return cudf::column_view(dtype, size, chars_ptr, null_mask, null_count,
                             offset, children);
  }

  case CUDF_INTEROP_LAYOUT_LIST: {
    auto const offsets_ptr = require_device_binding_ptr(
        bind_index, column->offsets_buffer_id,
        "prepared import referenced a missing list offsets binding",
        "prepared device import requires CUDA-device list offsets bindings");
    auto const offsets_size_bytes = require_device_binding_size(
        bind_index, column->offsets_buffer_id,
        "prepared import referenced a missing list offsets binding",
        "prepared device import requires CUDA-device list offsets bindings");
    CUDF_EXPECTS(offsets_size_bytes % sizeof(int32_t) == 0,
                 "prepared import list offsets binding must be 32-bit aligned");

    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(
            offsets_size_bytes / sizeof(int32_t),
            "prepared import list offsets binding exceeds cudf::size_type"),
        offsets_ptr, nullptr, 0, 0);
    children.emplace_back(build_prepared_column_view(column->child_column_id,
                                                     col_index, bind_index));
    return cudf::column_view(dtype, size, nullptr, null_mask, null_count,
                             offset, children);
  }

  case CUDF_INTEROP_LAYOUT_STRUCT: {
    CUDF_EXPECTS(
        column->num_child_column_ids > 0,
        "prepared import struct layout must reference at least one child");
    CUDF_EXPECTS(column->child_column_ids != nullptr,
                 "prepared import struct child ids pointer is null");
    std::vector<cudf::column_view> children;
    children.reserve(static_cast<size_t>(column->num_child_column_ids));
    for (int32_t index = 0; index < column->num_child_column_ids; ++index) {
      children.emplace_back(build_prepared_column_view(
          column->child_column_ids[index], col_index, bind_index));
    }
    return cudf::column_view(dtype, size, nullptr, null_mask, null_count,
                             offset, children);
  }

  case CUDF_INTEROP_LAYOUT_DICTIONARY: {
    auto const indices_ptr = require_device_binding_ptr(
        bind_index, column->indices_buffer_id,
        "prepared import referenced a missing dictionary indices binding",
        "prepared device import requires CUDA-device dictionary indices "
        "bindings");
    auto const indices_size_bytes = require_device_binding_size(
        bind_index, column->indices_buffer_id,
        "prepared import referenced a missing dictionary indices binding",
        "prepared device import requires CUDA-device dictionary indices "
        "bindings");
    CUDF_EXPECTS(
        indices_size_bytes % sizeof(int32_t) == 0,
        "prepared import dictionary indices binding must be 32-bit aligned");

    std::vector<cudf::column_view> children;
    children.reserve(2);
    children.emplace_back(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(indices_size_bytes / sizeof(int32_t),
                                      "prepared import dictionary indices "
                                      "binding exceeds cudf::size_type"),
        indices_ptr, nullptr, 0, 0);
    children.emplace_back(build_prepared_column_view(column->keys_column_id,
                                                     col_index, bind_index));
    return cudf::column_view(dtype, size, nullptr, null_mask, null_count,
                             offset, children);
  }
  }

  CUDF_FAIL("prepared import layout kind is unsupported");
}

rmm::device_buffer copy_host_binding_to_device(
    binding_index_t const &bind_index, uint32_t id, char const *missing_message,
    char const *placement_message, cudf_exec_context const &exec) {
  auto const binding =
      require_host_binding(bind_index, id, missing_message, placement_message);
  if (binding.byte_len == 0) {
    return rmm::device_buffer{};
  }
  return rmm::device_buffer(binding.data, binding.byte_len, exec.stream,
                            *exec.mr);
}

cudf::size_type
required_host_null_count(cudf_interop_semantic_column_t const &column) {
  switch (column.validity_kind) {
  case CUDF_INTEROP_VALIDITY_NONE:
    return 0;
  case CUDF_INTEROP_VALIDITY_MASK:
    if (column.null_count < 0) {
      throw cudf::logic_error("prepared host import requires an explicit null "
                              "count when a validity binding is "
                              "present");
    }
    return require_non_negative_i32(
        column.null_count, "prepared host import null_count must be >= 0");
  }
  throw cudf::logic_error("prepared host import validity kind is unsupported");
}

rmm::device_buffer
build_prepared_host_null_mask(cudf_interop_semantic_column_t const &column,
                              binding_index_t const &bind_index,
                              cudf::size_type size,
                              cudf_exec_context const &exec) {
  if (column.validity_kind == CUDF_INTEROP_VALIDITY_NONE) {
    return rmm::device_buffer{};
  }
  if (column.validity_kind != CUDF_INTEROP_VALIDITY_MASK) {
    throw cudf::logic_error(
        "prepared host import validity kind is unsupported");
  }

  auto const compact_bytes = static_cast<size_t>((size + 7) / 8);
  auto const alloc_bytes = cudf::bitmask_allocation_size_bytes(size);
  // Prepared host exports may carry either the exact packed bit count or the
  // cudf allocation size if the producer already padded the buffer for device
  // consumption. No other size is accepted.
  auto const actual_bytes = require_host_binding_size(
      bind_index, column.validity_buffer_id,
      "prepared host import referenced a missing validity binding",
      "prepared host import requires host-visible validity bindings");
  CUDF_EXPECTS(actual_bytes == compact_bytes || actual_bytes == alloc_bytes,
               "prepared host import validity binding size mismatch");
  if (alloc_bytes == 0) {
    return rmm::device_buffer{};
  }

  auto const *host_mask = static_cast<uint8_t const *>(require_host_binding_ptr(
      bind_index, column.validity_buffer_id,
      "prepared host import referenced a missing validity binding",
      "prepared host import requires host-visible validity bindings"));
  if (actual_bytes == alloc_bytes) {
    return rmm::device_buffer(host_mask, alloc_bytes, exec.stream, *exec.mr);
  }

  std::vector<uint8_t> padded_mask(alloc_bytes, 0);
  if (compact_bytes > 0) {
    std::memcpy(padded_mask.data(), host_mask, compact_bytes);
  }
  return rmm::device_buffer(padded_mask.data(), alloc_bytes, exec.stream,
                            *exec.mr);
}

std::unique_ptr<cudf::column> build_prepared_host_column(
    uint32_t root_column_id, column_index_t const &col_index,
    binding_index_t const &bind_index, cudf_exec_context const &exec) {
  auto const it = col_index.find(root_column_id);
  CUDF_EXPECTS(it != col_index.end(),
               "prepared host import referenced a missing semantic column");
  auto const *column = it->second;

  auto const dtype = interop_data_type(*column);
  auto const size = require_non_negative_i32(
      column->len, "prepared host import column length must be >= 0");
  auto const offset = require_non_negative_i32(
      column->offset, "prepared host import column offset must be >= 0");
  CUDF_EXPECTS(
      offset == 0,
      "prepared host import requires materialized semantic offsets to be 0");

  auto const null_count = required_host_null_count(*column);
  auto null_mask =
      build_prepared_host_null_mask(*column, bind_index, size, exec);

  switch (column->layout_kind) {
  case CUDF_INTEROP_LAYOUT_FIXED_WIDTH: {
    CUDF_EXPECTS(
        cudf::is_fixed_width(dtype),
        "prepared host import fixed-width layout requires a fixed-width type");
    auto const expected_bytes =
        static_cast<size_t>(size) * static_cast<size_t>(cudf::size_of(dtype));
    auto const actual_bytes = require_host_binding_size(
        bind_index, column->values_buffer_id,
        "prepared host import referenced a missing fixed-width values binding",
        "prepared host import requires host-visible fixed-width value "
        "bindings");
    CUDF_EXPECTS(actual_bytes == expected_bytes,
                 "prepared host import fixed-width binding size mismatch");
    auto data = copy_host_binding_to_device(
        bind_index, column->values_buffer_id,
        "prepared host import referenced a missing fixed-width values binding",
        "prepared host import requires host-visible fixed-width value bindings",
        exec);
    return std::make_unique<cudf::column>(dtype, size, std::move(data),
                                          std::move(null_mask), null_count);
  }

  case CUDF_INTEROP_LAYOUT_BINARY_LIKE: {
    CUDF_EXPECTS(
        dtype.id() == cudf::type_id::STRING,
        "prepared host import binary-like layout requires STRING type");
    auto const offsets_size_bytes = require_host_binding_size(
        bind_index, column->offsets_buffer_id,
        "prepared host import referenced a missing binary-like offsets binding",
        "prepared host import requires host-visible binary-like offsets "
        "bindings");
    CUDF_EXPECTS(offsets_size_bytes % sizeof(int32_t) == 0,
                 "prepared host import binary-like offsets binding must be "
                 "32-bit aligned");
    auto const expected_offsets_size =
        static_cast<size_t>(size + 1) * sizeof(int32_t);
    CUDF_EXPECTS(
        offsets_size_bytes == expected_offsets_size,
        "prepared host import binary-like offsets binding size mismatch");

    auto offsets_data = copy_host_binding_to_device(
        bind_index, column->offsets_buffer_id,
        "prepared host import referenced a missing binary-like offsets binding",
        "prepared host import requires host-visible binary-like offsets "
        "bindings",
        exec);
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(
            offsets_size_bytes / sizeof(int32_t),
            "prepared host import binary-like offsets binding exceeds "
            "cudf::size_type"),
        std::move(offsets_data), rmm::device_buffer{}, 0);

    auto chars = copy_host_binding_to_device(
        bind_index, column->data_buffer_id,
        "prepared host import referenced a missing binary-like chars binding",
        "prepared host import requires host-visible binary-like chars bindings",
        exec);
    return cudf::make_strings_column(size, std::move(offsets_col),
                                     std::move(chars), null_count,
                                     std::move(null_mask));
  }

  case CUDF_INTEROP_LAYOUT_LIST: {
    auto const offsets_size_bytes = require_host_binding_size(
        bind_index, column->offsets_buffer_id,
        "prepared host import referenced a missing list offsets binding",
        "prepared host import requires host-visible list offsets bindings");
    CUDF_EXPECTS(
        offsets_size_bytes % sizeof(int32_t) == 0,
        "prepared host import list offsets binding must be 32-bit aligned");
    auto const expected_offsets_size =
        static_cast<size_t>(size + 1) * sizeof(int32_t);
    CUDF_EXPECTS(offsets_size_bytes == expected_offsets_size,
                 "prepared host import list offsets binding size mismatch");

    auto offsets_data = copy_host_binding_to_device(
        bind_index, column->offsets_buffer_id,
        "prepared host import referenced a missing list offsets binding",
        "prepared host import requires host-visible list offsets bindings",
        exec);
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(offsets_size_bytes / sizeof(int32_t),
                                      "prepared host import list offsets "
                                      "binding exceeds cudf::size_type"),
        std::move(offsets_data), rmm::device_buffer{}, 0);
    auto child = build_prepared_host_column(column->child_column_id, col_index,
                                            bind_index, exec);
    return cudf::make_lists_column(size, std::move(offsets_col),
                                   std::move(child), null_count,
                                   std::move(null_mask));
  }

  case CUDF_INTEROP_LAYOUT_STRUCT: {
    CUDF_EXPECTS(
        column->num_child_column_ids > 0,
        "prepared host import struct layout must reference at least one child");
    CUDF_EXPECTS(column->child_column_ids != nullptr,
                 "prepared host import struct child ids pointer is null");
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(static_cast<size_t>(column->num_child_column_ids));
    for (int32_t index = 0; index < column->num_child_column_ids; ++index) {
      children.emplace_back(build_prepared_host_column(
          column->child_column_ids[index], col_index, bind_index, exec));
    }
    return cudf::make_structs_column(size, std::move(children), null_count,
                                     std::move(null_mask), exec.stream,
                                     *exec.mr);
  }

  case CUDF_INTEROP_LAYOUT_DICTIONARY: {
    auto const indices_size_bytes = require_host_binding_size(
        bind_index, column->indices_buffer_id,
        "prepared host import referenced a missing dictionary indices binding",
        "prepared host import requires host-visible dictionary indices "
        "bindings");
    CUDF_EXPECTS(indices_size_bytes % sizeof(int32_t) == 0,
                 "prepared host import dictionary indices binding must be "
                 "32-bit aligned");
    auto const expected_indices_size =
        static_cast<size_t>(size) * sizeof(int32_t);
    CUDF_EXPECTS(
        indices_size_bytes == expected_indices_size,
        "prepared host import dictionary indices binding size mismatch");

    auto indices_data = copy_host_binding_to_device(
        bind_index, column->indices_buffer_id,
        "prepared host import referenced a missing dictionary indices binding",
        "prepared host import requires host-visible dictionary indices "
        "bindings",
        exec);
    auto indices = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        require_size_type_from_size_t(indices_size_bytes / sizeof(int32_t),
                                      "prepared host import dictionary indices "
                                      "binding exceeds cudf::size_type"),
        std::move(indices_data), rmm::device_buffer{}, 0);
    auto keys = build_prepared_host_column(column->keys_column_id, col_index,
                                           bind_index, exec);
    return cudf::make_dictionary_column(std::move(keys), std::move(indices),
                                        std::move(null_mask), null_count);
  }
  }

  CUDF_FAIL("prepared host import layout kind is unsupported");
}
} // namespace

extern "C" {
cudf_status_t cudf_column_copy_from_prepared_device_export_ex(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_column_handle *out) {
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
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto const col_index =
        build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    auto const view =
        build_prepared_column_view(root_column_id, col_index, bind_index);
    {
      auto copy = std::make_unique<cudf::column>(view, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(copy),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy_from_prepared_device_export_ex(
    cudf_interop_semantic_table_t const *table,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_table_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(semantic_columns, "semantic_columns is null");
  CUDF_RUST_NULL_CHECK(bindings, "bindings is null");
  // Tables may be empty at the top level, so a zero-column / zero-semantic
  // graph is valid here even though column import requires one semantic root.
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
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto const col_index =
        build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    std::vector<cudf::column_view> columns;
    columns.reserve(static_cast<size_t>(table->num_columns));
    for (int32_t index = 0; index < table->num_columns; ++index) {
      columns.emplace_back(build_prepared_column_view(table->column_ids[index],
                                                      col_index, bind_index));
    }

    {
      auto copy = std::make_unique<cudf::table>(cudf::table_view(columns),
                                                exec->stream, *exec->mr);
      *out = cudf_release_table_handle(std::move(copy),
                                       cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_from_prepared_host_export_ex(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_column_handle *out) {
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
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto const col_index =
        build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    auto column = build_prepared_host_column(root_column_id, col_index,
                                             bind_index, *exec);
    *out = cudf_release_column_handle(std::move(column),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_table_copy_from_prepared_host_export_ex(
    cudf_interop_semantic_table_t const *table,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_exec_options_t const *options,
    cudf_table_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(semantic_columns, "semantic_columns is null");
  CUDF_RUST_NULL_CHECK(bindings, "bindings is null");
  // Tables may be empty at the top level, so a zero-column / zero-semantic
  // graph is valid here even though column import requires one semantic root.
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
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto const col_index =
        build_column_index(semantic_columns, num_semantic_columns);
    auto const bind_index = build_binding_index(bindings, num_bindings);
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(static_cast<size_t>(table->num_columns));
    for (int32_t index = 0; index < table->num_columns; ++index) {
      columns.emplace_back(build_prepared_host_column(
          table->column_ids[index], col_index, bind_index, *exec));
    }

    auto owned = std::make_unique<cudf::table>(std::move(columns));
    *out = cudf_release_table_handle(std::move(owned),
                                     cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
