#include "cudf_rust/column.h"

#include "execution_context_internal.hpp"
#include "owning_handle.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"
#include "test_support_internal.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
bool is_decimal_type_id(cudf_type_id_t type_id)
{
  return type_id == CUDF_TYPE_DECIMAL32 || type_id == CUDF_TYPE_DECIMAL64 ||
         type_id == CUDF_TYPE_DECIMAL128;
}

cudf::data_type host_fixed_width_dtype(cudf_type_id_t type_id, int32_t scale)
{
  auto const id = to_cudf_type_id(type_id);
  return is_decimal_type_id(type_id) ? cudf::data_type{id, scale} : cudf::data_type{id};
}

cudf_status_t maybe_fail_column_view_metadata_read()
{
  if (!cudf::rust::test_support::take_column_view_metadata_read_failure()) { return CUDF_SUCCESS; }
  set_error("test metadata read failure");
  return CUDF_ERROR_UNKNOWN;
}

void maybe_inject_import_fault(bool injected, char const* message)
{
  if (injected) { throw std::runtime_error(message); }
}

cudf_status_t resolve_column_view_null_count(cudf::column_view const& view,
                                             rmm::cuda_stream_view stream,
                                             int32_t* out)
{
  auto const stored = view.null_count();
  if (stored >= 0) {
    *out = stored;
    return CUDF_SUCCESS;
  }
  if (!view.nullable() || view.size() == 0) {
    *out = 0;
    return CUDF_SUCCESS;
  }

  *out = view.null_count(0, view.size(), stream);
  return CUDF_SUCCESS;
}

// Mirrors cudf::empty_like for nested columns: a zero-row LIST holds an empty
// INT32 offsets child before its element child, and no level owns a buffer.
std::unique_ptr<cudf::column> make_empty_column_from_type_tree(cudf_column_type_node_t const* nodes,
                                                               int32_t num_nodes,
                                                               int32_t& cursor)
{
  CUDF_EXPECTS(cursor < num_nodes,
               "type tree ends before every declared child is described",
               std::invalid_argument);
  auto const& node = nodes[cursor++];
  CUDF_EXPECTS(node.num_children >= 0, "type tree num_children must be >= 0", std::invalid_argument);
  CUDF_EXPECTS(node.type_id > CUDF_TYPE_EMPTY && node.type_id <= CUDF_TYPE_STRUCT &&
                 node.type_id != CUDF_TYPE_DICTIONARY32,
               "type tree node has no empty column construction",
               cudf::data_type_error);

  std::vector<std::unique_ptr<cudf::column>> children;
  if (node.type_id == CUDF_TYPE_LIST) {
    CUDF_EXPECTS(
      node.num_children == 1, "type tree LIST node requires exactly one child", std::invalid_argument);
    children.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
    children.push_back(make_empty_column_from_type_tree(nodes, num_nodes, cursor));
  } else if (node.type_id == CUDF_TYPE_STRUCT) {
    CUDF_EXPECTS(
      node.num_children > 0, "type tree STRUCT node requires at least one child", std::invalid_argument);
    children.reserve(static_cast<size_t>(node.num_children));
    for (int32_t child = 0; child < node.num_children; ++child) {
      children.push_back(make_empty_column_from_type_tree(nodes, num_nodes, cursor));
    }
  } else {
    CUDF_EXPECTS(node.num_children == 0, "type tree leaf node must have no children", std::invalid_argument);
    return cudf::make_empty_column(host_fixed_width_dtype(node.type_id, node.scale));
  }
  return std::make_unique<cudf::column>(cudf::data_type{to_cudf_type_id(node.type_id)},
                                        0,
                                        rmm::device_buffer{},
                                        rmm::device_buffer{},
                                        0,
                                        std::move(children));
}

}  // namespace

static cudf_status_t create_column_from_device_data_impl(cudf_type_id_t type_id,
                                                         void const* device_data,
                                                         int32_t size,
                                                         cudf_exec_options_t const* options,
                                                         bool synchronize_before_return,
                                                         cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (size > 0) { CUDF_RUST_NULL_CHECK(device_data, "device_data is null"); }
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    if (size == 0) {
      auto col = cudf::make_empty_column(dtype);
      *out     = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    auto elem_size    = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    rmm::device_buffer dev_data(data_bytes, exec->stream, *exec->mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      dev_data.data(), device_data, data_bytes, cudaMemcpyDeviceToDevice, exec->stream.value()));
    if (synchronize_before_return) {
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
    }

    auto col = std::make_unique<cudf::column>(
      dtype, static_cast<cudf::size_type>(size), std::move(dev_data), rmm::device_buffer{}, 0);

    *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

extern "C" {
cudf_status_t cudf_column_create_empty_ex(cudf_type_id_t type_id,
                                          cudf_exec_options_t const* options,
                                          cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      auto col = cudf::make_empty_column(dtype);
      *out     = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_empty_from_type_tree_ex(cudf_column_type_node_t const* nodes,
                                                         int32_t num_nodes,
                                                         cudf_exec_options_t const* options,
                                                         cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(nodes, "nodes is null");
  if (num_nodes <= 0) {
    set_error(CUDF_ERROR_INVALID_ARGUMENT, "num_nodes must be > 0");
    return CUDF_ERROR_INVALID_ARGUMENT;
  }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    int32_t cursor = 0;
    auto col       = make_empty_column_from_type_tree(nodes, num_nodes, cursor);
    CUDF_EXPECTS(cursor == num_nodes,
                 "type tree has nodes after the root subtree",
                 std::invalid_argument);
    *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_numeric_ex(cudf_type_id_t type_id,
                                            void const* data,
                                            int32_t size,
                                            uint8_t const* null_mask,
                                            int32_t null_count,
                                            cudf_exec_options_t const* options,
                                            cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (null_count < 0 || null_count > size) {
    set_error("null_count must be in [0, size]");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (size > 0) { CUDF_RUST_NULL_CHECK(data, "data is null"); }
  if (null_count > 0) { CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0"); }
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto elem_size    = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    std::optional<rmm::device_buffer> dev_data;
    std::optional<rmm::device_buffer> dev_null_mask;
    cudf_host_source_copy_sync_guard host_copy_guard{exec->stream};
    if (data_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
    dev_data.emplace(data, data_bytes, exec->stream, *exec->mr);

    if (null_count > 0) {
      // cudf bitmask is stored as uint32_t words.
      size_t mask_bytes = cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(size));
      // Rust passes packed bits zero-padded to the cudf mask allocation size.
      // The device buffer takes the full allocation size.
      if (mask_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
      dev_null_mask.emplace(null_mask, mask_bytes, exec->stream, *exec->mr);
    }

    host_copy_guard.synchronize_and_release();

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) { null_mask_buffer = std::move(*dev_null_mask); }
    auto col = std::make_unique<cudf::column>(dtype,
                                              static_cast<cudf::size_type>(size),
                                              std::move(*dev_data),
                                              std::move(null_mask_buffer),
                                              static_cast<cudf::size_type>(null_count));

    *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_fixed_width_from_host_ex(cudf_type_id_t type_id,
                                                          int32_t scale,
                                                          void const* data,
                                                          int32_t size,
                                                          uint8_t const* null_mask,
                                                          int32_t null_count,
                                                          cudf_exec_options_t const* options,
                                                          cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (null_count < 0 || null_count > size) {
    set_error("null_count must be in [0, size]");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (!is_decimal_type_id(type_id) && scale != 0) {
    set_error("scale must be 0 for non-decimal fixed-width host columns");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (size > 0) { CUDF_RUST_NULL_CHECK(data, "data is null"); }
  if (null_count > 0) { CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0"); }
  CUDF_RUST_TRY
  {
    auto const dtype = host_fixed_width_dtype(type_id, scale);
    if (!cudf::is_fixed_width(dtype) || type_id == CUDF_TYPE_DICTIONARY32) {
      set_error(
        "fixed-width host column construction requires a "
        "non-dictionary fixed-width type");
      return CUDF_ERROR_TYPE;
    }

    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto const elem_size  = cudf::size_of(dtype);
    auto const data_bytes = static_cast<size_t>(size) * elem_size;
    std::vector<uint8_t> padded_mask;
    std::optional<rmm::device_buffer> dev_data;
    std::optional<rmm::device_buffer> dev_null_mask;
    cudf_host_source_copy_sync_guard host_copy_guard{exec->stream};
    if (data_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
    dev_data.emplace(data, data_bytes, exec->stream, *exec->mr);
    maybe_inject_import_fault(
      cudf::rust::test_support::take_fixed_width_host_import_after_data_copy(),
      "test fixed-width host import failure after data copy");

    if (null_mask != nullptr && size > 0) {
      auto const compact_mask_bytes = static_cast<size_t>((size + 7) / 8);
      auto const alloc_mask_bytes =
        cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(size));
      padded_mask.assign(alloc_mask_bytes, 0);
      std::memcpy(padded_mask.data(), null_mask, compact_mask_bytes);
      if (alloc_mask_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
      dev_null_mask.emplace(padded_mask.data(), alloc_mask_bytes, exec->stream, *exec->mr);
      maybe_inject_import_fault(
        cudf::rust::test_support::take_fixed_width_host_import_after_null_mask_copy(),
        "test fixed-width host import failure after null mask copy");
    }

    host_copy_guard.synchronize_and_release();

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) { null_mask_buffer = std::move(*dev_null_mask); }
    auto col = std::make_unique<cudf::column>(dtype,
                                              static_cast<cudf::size_type>(size),
                                              std::move(*dev_data),
                                              std::move(null_mask_buffer),
                                              static_cast<cudf::size_type>(null_count));

    *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_sequence_ex(cudf_scalar_handle init,
                                             cudf_scalar_handle step,
                                             int32_t size,
                                             cudf_exec_options_t const* options,
                                             cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(init, "init is null");
  CUDF_RUST_NULL_CHECK(step, "step is null");
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto const& init_scalar         = cudf_scalar_from_handle_const(init);
    auto const& step_scalar         = cudf_scalar_from_handle_const(step);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      auto col = cudf::sequence(
        static_cast<cudf::size_type>(size), init_scalar, step_scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_ex(cudf_column_handle src,
                                  cudf_exec_options_t const* options,
                                  cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      auto copy = cudf_column_state_from_handle(src).deep_copy(exec->stream, *exec->mr);
      *out      = cudf_release_column_handle(std::move(copy), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_from_view_ex(cudf_column_view_handle src,
                                            cudf_exec_options_t const* options,
                                            cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY
  {
    auto const& view                = *reinterpret_cast<cudf::column_view const*>(src);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    {
      auto copy = std::make_unique<cudf::column>(view, exec->stream, *exec->mr);
      *out      = cudf_release_column_handle(std::move(copy), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_payload_size_ex(cudf_column_handle handle,
                                          cudf_exec_options_t const* options,
                                          size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    *out = cudf_column_state_from_handle(handle).payload_size(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_device_allocated_size(cudf_column_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    *out = cudf_column_state_from_handle(handle).device_allocated_size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_backing_allocation_pointers(cudf_column_handle handle,
                                                       void** out,
                                                       size_t capacity,
                                                       size_t* count)
{
  CUDF_RUST_NULL_CHECK(count, "count is null");
  *count = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  if (capacity != 0) { CUDF_RUST_NULL_CHECK(out, "out is null with non-zero capacity"); }
  CUDF_RUST_TRY
  {
    auto pointers = cudf_column_state_from_handle(handle).backing_allocation_pointers();
    *count        = pointers.size();
    if (out == nullptr && capacity == 0) { return CUDF_SUCCESS; }
    if (capacity < pointers.size()) {
      set_error("backing allocation pointer capacity is too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    std::copy(pointers.begin(), pointers.end(), out);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_destroy(cudf_column_handle handle)
{
  // Ownership is consumed on every return path. A failure means cleanup did
  // not complete and the allocation was deliberately quarantined.
  return cudf_destroy_column_handle_under_provenance(handle);
}

cudf_status_t cudf_column_view(cudf_column_handle handle, cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = new cudf::column_view(cudf_column_state_from_handle(handle).view());
    *out       = reinterpret_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_destroy(cudf_column_view_handle handle)
{
  try {
    delete reinterpret_cast<cudf::column_view*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_status_t cudf_column_view_type(cudf_column_view_handle handle, cudf_type_id_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = CUDF_TYPE_EMPTY;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = static_cast<cudf_type_id_t>(view->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_data_type(cudf_column_view_handle handle,
                                         cudf_type_id_t* out_type_id,
                                         int32_t* out_scale)
{
  CUDF_RUST_NULL_CHECK(out_type_id, "out_type_id is null");
  CUDF_RUST_NULL_CHECK(out_scale, "out_scale is null");
  *out_type_id = CUDF_TYPE_EMPTY;
  *out_scale   = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view   = reinterpret_cast<cudf::column_view*>(handle);
    auto dtype   = view->type();
    *out_type_id = static_cast<cudf_type_id_t>(dtype.id());
    *out_scale   = dtype.scale();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_size(cudf_column_view_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  cudf_status_t const fail_status = maybe_fail_column_view_metadata_read();
  if (fail_status != CUDF_SUCCESS) { return fail_status; }
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = view->size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  cudf_status_t const fail_status = maybe_fail_column_view_metadata_read();
  if (fail_status != CUDF_SUCCESS) { return fail_status; }
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = view->null_count();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_count_ex(cudf_column_view_handle handle,
                                             cudf_exec_options_t const* options,
                                             int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  cudf_status_t const fail_status = maybe_fail_column_view_metadata_read();
  if (fail_status != CUDF_SUCCESS) { return fail_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    return resolve_column_view_null_count(*view, exec->stream, out);
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = view->nullable() ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = view->num_children();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_child(cudf_column_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view  = reinterpret_cast<cudf::column_view*>(handle);
    auto* child = new cudf::column_view(view->child(index));
    *out        = reinterpret_cast<cudf_column_view_handle>(child);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_list_sliced_child_ex(cudf_column_view_handle handle,
                                                    cudf_exec_options_t const* options,
                                                    cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    if (view->type().id() != cudf::type_id::LIST) {
      set_error("column is not LIST type");
      return CUDF_ERROR_TYPE;
    }

    cudf::lists_column_view const lists{*view};
    auto* child = new cudf::column_view(lists.get_sliced_child(exec->stream));
    *out        = reinterpret_cast<cudf_column_view_handle>(child);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_data_ptr(cudf_column_view_handle handle, void const** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    // data<uint8_t>() returns the base device pointer. For STRING columns
    // this is nullptr; callers should use child views instead.
    *out = view->head<uint8_t>();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_string_chars_ptr_and_size_ex(cudf_column_view_handle handle,
                                                            cudf_exec_options_t const* options,
                                                            void const** out_ptr,
                                                            size_t* out_size)
{
  CUDF_RUST_NULL_CHECK(out_ptr, "out_ptr is null");
  CUDF_RUST_NULL_CHECK(out_size, "out_size is null");
  *out_ptr  = nullptr;
  *out_size = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    if (view->type().id() != cudf::type_id::STRING) {
      set_error("column type is not STRING");
      return CUDF_ERROR_TYPE;
    }

    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    cudf::strings_column_view scv(*view);
    cudf::size_type first_offset = 0;
    cudf::size_type last_offset  = 0;
    auto const offsets = cudf::strings::detail::get_first_and_last_offset(scv, exec->stream);
    first_offset       = static_cast<cudf::size_type>(offsets.first);
    last_offset        = static_cast<cudf::size_type>(offsets.second);
    if (first_offset < 0 || last_offset < first_offset) {
      set_error("string chars span is invalid");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    // The returned pointer is the BASE of the chars buffer, and the slice's
    // offsets index from that base, so the reported span must end at
    // last_offset; not span last_offset - first_offset bytes. Otherwise a
    // sliced view with first_offset > 0 under-reports the bound and rows near
    // the end of the slice reference bytes past the declared span.
    auto const span_bytes = static_cast<uint64_t>(last_offset);
    if (span_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      set_error("string chars span exceeds size_t range");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    *out_ptr  = scv.chars_begin(exec->stream);
    *out_size = static_cast<size_t>(span_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_mask_ptr(cudf_column_view_handle handle, void const** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    if (!view->nullable()) {
      set_error("Column has no null mask");
      return CUDF_ERROR_NULL;
    }
    *out = view->null_mask();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_offset(cudf_column_view_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    *out       = view->offset();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_slice_ex(cudf_column_view_handle handle,
                                        int32_t offset,
                                        int32_t length,
                                        cudf_exec_options_t const* options,
                                        cudf_column_view_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  if (offset < 0) {
    set_error("offset must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (length < 0) {
    set_error("length must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }

  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view       = reinterpret_cast<cudf::column_view*>(handle);
    auto const start = static_cast<cudf::size_type>(offset);
    auto const len   = static_cast<cudf::size_type>(length);
    if (start > view->size() || len > view->size() - start) {
      set_error("slice range is out of bounds");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    constexpr auto max_size_type =
      static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max());
    auto const active_begin64 = static_cast<int64_t>(view->offset()) + static_cast<int64_t>(start);
    auto const active_end64   = active_begin64 + static_cast<int64_t>(len);
    if (active_begin64 > max_size_type || active_end64 > max_size_type) {
      set_error("slice range exceeds active column offset range");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto const active_begin = static_cast<cudf::size_type>(active_begin64);

    std::vector<cudf::column_view> children{};
    children.reserve(view->num_children());
    for (cudf::size_type index = 0; index < view->num_children(); ++index) {
      children.emplace_back(view->child(index));
    }

    auto null_count = cudf::size_type{0};
    if (view->nullable()) {
      auto const parent_null_count = view->null_count();
      if (len == 0 || parent_null_count == 0) {
        null_count = 0;
      } else if (parent_null_count == view->size()) {
        null_count = len;
      } else {
        null_count = view->null_count(start, start + len, exec->stream);
      }
    }

    auto* slice = new cudf::column_view(
      view->type(), len, view->head(), view->null_mask(), null_count, active_begin, children);
    *out = reinterpret_cast<cudf_column_view_handle>(slice);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_to_host_ex(cudf_column_view_handle handle,
                                          void* host_buf,
                                          size_t buf_size,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    auto dtype = view->type();
    if (!cudf::is_fixed_width(dtype)) {
      set_error("column_view_to_host only supports fixed-width types");
      return CUDF_ERROR_TYPE;
    }
    auto elem_size = cudf::size_of(dtype);
    size_t needed  = static_cast<size_t>(view->size()) * elem_size;
    if (buf_size < needed) {
      set_error("Buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    // Copy from device to host. data<uint8_t>() would advance by
    // offset() * sizeof(uint8_t) bytes regardless of the element width, so
    // compute the sliced start from head() with the real element size.
    auto const* src = view->head<uint8_t>() + static_cast<size_t>(view->offset()) * elem_size;
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(host_buf, src, needed, cudaMemcpyDeviceToHost, exec->stream.value()));
    // The caller's host buffer is populated on return.
    exec->stream.synchronize();
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_mask_to_host_ex(cudf_column_view_handle handle,
                                                    uint8_t* host_buf,
                                                    size_t buf_size,
                                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY
  {
    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    if (!view->nullable()) {
      set_error("Column has no null mask");
      return CUDF_ERROR_NULL;
    }
    size_t mask_bytes = cudf::bitmask_allocation_size_bytes(view->size());
    if (buf_size < mask_bytes) {
      set_error("Buffer too small for null mask");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    if (mask_bytes == 0) { return CUDF_SUCCESS; }

    // Zero the full aligned allocation so bits beyond the logical slice
    // (mask_bytes may be up to 63 bytes larger than compact_mask_bytes due
    // to cuDF's 64-byte bitmask alignment) are deterministically zero.
    std::memset(host_buf, 0, mask_bytes);
    size_t compact_mask_bytes = (static_cast<size_t>(view->size()) + 7) / 8;
    std::optional<rmm::device_buffer> normalized_mask;
    {
      normalized_mask.emplace(cudf::copy_bitmask(*view, exec->stream, *exec->mr));
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_buf,
                                  normalized_mask->data(),
                                  compact_mask_bytes,
                                  cudaMemcpyDeviceToHost,
                                  exec->stream.value()));
    // The caller's host buffer is populated on return.
    exec->stream.synchronize();
    if ((view->size() % 8) != 0) {
      auto const last_byte_mask = static_cast<uint8_t>((1u << (view->size() % 8)) - 1u);
      host_buf[compact_mask_bytes - 1] &= last_byte_mask;
    }
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_from_device_data(cudf_type_id_t type_id,
                                                  void const* device_data,
                                                  int32_t size,
                                                  cudf_column_handle* out)
{
  return create_column_from_device_data_impl(type_id, device_data, size, nullptr, true, out);
}

cudf_status_t cudf_column_create_from_device_data_ex(cudf_type_id_t type_id,
                                                     void const* device_data,
                                                     int32_t size,
                                                     cudf_exec_options_t const* options,
                                                     cudf_column_handle* out)
{
  return create_column_from_device_data_impl(type_id, device_data, size, options, false, out);
}

cudf_status_t cudf_column_create_strings_ex(char const** strings,
                                            int32_t const* lengths,
                                            int32_t num_strings,
                                            uint8_t const* null_mask,
                                            int32_t null_count,
                                            cudf_exec_options_t const* options,
                                            cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_strings < 0) {
    set_error("num_strings must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_strings == std::numeric_limits<int32_t>::max()) {
    set_error("num_strings is too large to build string offsets");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (null_count < 0 || null_count > num_strings) {
    set_error("null_count must be in [0, num_strings]");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_strings > 0) {
    CUDF_RUST_NULL_CHECK(strings, "strings is null");
    CUDF_RUST_NULL_CHECK(lengths, "lengths is null");
  }
  if (null_count > 0) { CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0"); }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    if (num_strings == 0) {
      auto col = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
      *out     = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    auto const num_strings_size = static_cast<size_t>(num_strings);
    std::vector<int32_t> host_offsets(num_strings_size + 1);
    host_offsets[0] = 0;
    for (int32_t i = 0; i < num_strings; ++i) {
      if (lengths[i] < 0) {
        set_error("string lengths must be >= 0");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      if (lengths[i] > 0 && strings[i] == nullptr) {
        set_error("string entry is null");
        return CUDF_ERROR_NULL;
      }
      auto const next_offset =
        static_cast<int64_t>(host_offsets[i]) + static_cast<int64_t>(lengths[i]);
      if (next_offset > std::numeric_limits<int32_t>::max()) {
        set_error("total string bytes exceed INT32_MAX");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      host_offsets[i + 1] = static_cast<int32_t>(next_offset);
    }
    int32_t total_chars = host_offsets[num_strings];

    std::vector<char> host_chars(total_chars);
    for (int32_t i = 0; i < num_strings; ++i) {
      if (lengths[i] > 0) {
        std::memcpy(host_chars.data() + host_offsets[i], strings[i], lengths[i]);
      }
    }

    std::vector<uint8_t> padded_mask;
    std::optional<rmm::device_buffer> dev_offsets;
    std::optional<rmm::device_buffer> dev_chars;
    std::optional<rmm::device_buffer> dev_null_mask;
    cudf_host_source_copy_sync_guard host_copy_guard{exec->stream};

    auto const offsets_count = num_strings_size + 1;
    auto const offsets_bytes = offsets_count * sizeof(int32_t);
    if (offsets_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
    dev_offsets.emplace(host_offsets.data(), offsets_bytes, exec->stream, *exec->mr);
    maybe_inject_import_fault(
      cudf::rust::test_support::take_strings_host_import_after_offsets_copy(),
      "test strings host import failure after offsets copy");

    auto const chars_bytes = static_cast<size_t>(total_chars);
    if (chars_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
    dev_chars.emplace(host_chars.data(), chars_bytes, exec->stream, *exec->mr);
    maybe_inject_import_fault(cudf::rust::test_support::take_strings_host_import_after_chars_copy(),
                              "test strings host import failure after chars copy");

    if (null_count > 0) {
      auto const compact_mask_bytes = (num_strings_size + 7) / 8;
      auto const alloc_mask_bytes =
        cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(num_strings));
      padded_mask.assign(alloc_mask_bytes, 0);
      std::memcpy(padded_mask.data(), null_mask, compact_mask_bytes);
      if (alloc_mask_bytes > 0) { host_copy_guard.mark_copy_may_be_pending(); }
      dev_null_mask.emplace(padded_mask.data(), alloc_mask_bytes, exec->stream, *exec->mr);
      maybe_inject_import_fault(
        cudf::rust::test_support::take_strings_host_import_after_null_mask_copy(),
        "test strings host import failure after null mask copy");
    }

    maybe_inject_import_fault(
      cudf::rust::test_support::take_strings_host_import_before_make_strings_column(),
      "test strings host import failure before make_strings_column");
    host_copy_guard.synchronize_and_release();

    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                      static_cast<cudf::size_type>(offsets_count),
                                                      std::move(*dev_offsets),
                                                      rmm::device_buffer{},
                                                      0);

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) { null_mask_buffer = std::move(*dev_null_mask); }
    std::unique_ptr<cudf::column> col;
    {
      col = cudf::make_strings_column(static_cast<cudf::size_type>(num_strings),
                                      std::move(offsets_col),
                                      std::move(*dev_chars),
                                      static_cast<cudf::size_type>(null_count),
                                      std::move(null_mask_buffer));
    }
    maybe_inject_import_fault(
      cudf::rust::test_support::take_strings_host_import_after_make_strings_column(),
      "test strings host import failure after make_strings_column");

    *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_strings_to_host_ex(cudf_column_view_handle handle,
                                                  int32_t* offsets_buf,
                                                  size_t offsets_buf_size,
                                                  char* chars_buf,
                                                  size_t chars_buf_size,
                                                  int32_t* total_chars_bytes,
                                                  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(total_chars_bytes, "total_chars_bytes is null");
  *total_chars_bytes = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = reinterpret_cast<cudf::column_view*>(handle);
    if (view->type().id() != cudf::type_id::STRING) {
      set_error("column is not STRING type");
      return CUDF_ERROR_TYPE;
    }

    auto n = view->size();
    if (n == 0) {
      CUDF_RUST_NULL_CHECK(offsets_buf, "offsets_buf is null");
      if (offsets_buf_size < sizeof(int32_t)) {
        set_error("offsets buffer too small");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      offsets_buf[0]     = 0;
      *total_chars_bytes = 0;
      return CUDF_SUCCESS;
    }

    // String columns: child 0 = offsets, child 1 = chars
    // For the *sliced* view, n+1 offsets are needed. The offsets child can hold
    // more entries than n+1 when the view is a slice of a larger column.
    if (view->num_children() < 1) {
      set_error("string column view must expose an offsets child");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto offsets_view             = view->child(0);
    auto offsets_count            = n + 1;
    auto const offsets_count_size = static_cast<size_t>(offsets_count);

    size_t needed_offsets_bytes = offsets_count_size * sizeof(int32_t);
    CUDF_RUST_NULL_CHECK(offsets_buf, "offsets_buf is null");
    if (offsets_buf_size < needed_offsets_bytes) {
      set_error("offsets buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    std::vector<int64_t> host_offsets(offsets_count_size);
    auto const offsets_type = offsets_view.type().id();
    if (offsets_type == cudf::type_id::INT32) {
      std::vector<int32_t> raw_offsets(offsets_count_size);
      auto offsets_data = offsets_view.data<int32_t>() + view->offset();
      CUDF_CUDA_TRY(cudaMemcpyAsync(raw_offsets.data(),
                                    offsets_data,
                                    offsets_count_size * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost,
                                    exec->stream.value()));
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
      for (size_t i = 0; i < offsets_count_size; ++i) {
        host_offsets[i] = raw_offsets[i];
      }
    } else if (offsets_type == cudf::type_id::INT64) {
      auto offsets_data = offsets_view.data<int64_t>() + view->offset();
      CUDF_CUDA_TRY(cudaMemcpyAsync(host_offsets.data(),
                                    offsets_data,
                                    offsets_count_size * sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,
                                    exec->stream.value()));
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
    } else {
      set_error("string offsets child must be INT32 or INT64");
      return CUDF_ERROR_TYPE;
    }

    int64_t char_begin = host_offsets[0];
    int64_t char_end   = host_offsets[offsets_count_size - 1];
    if (char_begin < 0 || char_end < char_begin) {
      set_error("string offsets are invalid");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    int64_t total = char_end - char_begin;
    if (total > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      set_error("string chars span exceeds INT32_MAX");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    *total_chars_bytes = static_cast<int32_t>(total);

    if (total > 0) {
      auto const total_chars = static_cast<size_t>(total);
      if (chars_buf != nullptr) {
        if (chars_buf_size < total_chars) {
          set_error("chars buffer too small");
          return CUDF_ERROR_INVALID_OPERATION;
        }

        // The chars child of the STRING column is read through strings_column_view.
        cudf::strings_column_view scv(*view);
        char const* chars_ptr = scv.chars_begin(exec->stream);

        // chars_begin does NOT account for view offset, but the offsets
        // already record where chars start (host_offsets[0] = char_begin).
        CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buf,
                                      chars_ptr + char_begin,
                                      total_chars,
                                      cudaMemcpyDeviceToHost,
                                      exec->stream.value()));
        // The caller's chars buffer is populated on return.
        exec->stream.synchronize();
        cudf_record_synchronized_stream(exec->stream);
      }
    }

    // Adjust offsets to be zero-based for the caller
    for (size_t i = 0; i < offsets_count_size; ++i) {
      auto const normalized = host_offsets[i] - char_begin;
      if (normalized < 0 ||
          normalized > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        set_error("normalized string offset exceeds INT32 range");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      offsets_buf[i] = static_cast<int32_t>(normalized);
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_from_scalar_ex(cudf_scalar_handle scalar,
                                         int32_t size,
                                         cudf_column_handle* out,
                                         cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(scalar, "scalar is null");
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto const& sc                  = cudf_scalar_from_handle_const(scalar);
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    {
      // libcudf's LIST scalar broadcast path still allocates nested
      // temporaries from RMM's current resource despite receiving an explicit
      // memory_resource_ref. Keep current aligned with the explicit MR for
      // this allocation boundary.
      auto col = cudf::make_column_from_scalar(
        sc, static_cast<cudf::size_type>(size), exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(col), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
