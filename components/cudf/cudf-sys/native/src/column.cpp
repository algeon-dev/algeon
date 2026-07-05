#include "column.h"

#include "execution_context_internal.hpp"
#include "owning_handle.hpp"
#include "scalar_handle.hpp"
#include "shim_common.h"

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

#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {
thread_local bool g_fail_next_column_view_metadata_read = false;

enum class fixed_width_host_import_fault {
  none,
  after_data_copy,
  after_null_mask_copy,
};

thread_local fixed_width_host_import_fault
    g_next_fixed_width_host_import_fault = fixed_width_host_import_fault::none;

enum class strings_host_import_fault {
  none,
  after_offsets_copy,
  after_chars_copy,
  after_null_mask_copy,
  before_make_strings_column,
  after_make_strings_column,
};

thread_local strings_host_import_fault g_next_strings_host_import_fault =
    strings_host_import_fault::none;
// Synthetic fixed-width test views only need a stable non-null backing pointer.
// The shim never writes through this storage.
std::array<int64_t, 8> g_test_column_view_dummy_storage = {};

bool should_fail_next_column_view_metadata_read() {
  if (!g_fail_next_column_view_metadata_read) {
    return false;
  }
  g_fail_next_column_view_metadata_read = false;
  return true;
}

bool is_decimal_type_id(cudf_type_id_t type_id) {
  return type_id == CUDF_TYPE_DECIMAL32 || type_id == CUDF_TYPE_DECIMAL64 ||
         type_id == CUDF_TYPE_DECIMAL128;
}

cudf::data_type host_fixed_width_dtype(cudf_type_id_t type_id, int32_t scale) {
  auto const id = to_cudf_type_id(type_id);
  return is_decimal_type_id(type_id) ? cudf::data_type{id, scale}
                                     : cudf::data_type{id};
}

cudf_status_t maybe_fail_column_view_metadata_read() {
  if (!should_fail_next_column_view_metadata_read()) {
    return CUDF_SUCCESS;
  }
  set_error("test metadata read failure");
  return CUDF_ERROR_UNKNOWN;
}

bool consume_fixed_width_host_import_fault(
    fixed_width_host_import_fault fault) {
  if (g_next_fixed_width_host_import_fault != fault) {
    return false;
  }
  g_next_fixed_width_host_import_fault = fixed_width_host_import_fault::none;
  return true;
}

void maybe_fail_fixed_width_host_import(fixed_width_host_import_fault fault) {
  if (!consume_fixed_width_host_import_fault(fault)) {
    return;
  }
  switch (fault) {
  case fixed_width_host_import_fault::after_data_copy:
    throw std::runtime_error(
        "test fixed-width host import failure after data copy");
  case fixed_width_host_import_fault::after_null_mask_copy:
    throw std::runtime_error(
        "test fixed-width host import failure after null mask copy");
  case fixed_width_host_import_fault::none:
    break;
  }
}

bool consume_strings_host_import_fault(strings_host_import_fault fault) {
  if (g_next_strings_host_import_fault != fault) {
    return false;
  }
  g_next_strings_host_import_fault = strings_host_import_fault::none;
  return true;
}

void maybe_fail_strings_host_import(strings_host_import_fault fault) {
  if (!consume_strings_host_import_fault(fault)) {
    return;
  }
  switch (fault) {
  case strings_host_import_fault::after_offsets_copy:
    throw std::runtime_error(
        "test strings host import failure after offsets copy");
  case strings_host_import_fault::after_chars_copy:
    throw std::runtime_error(
        "test strings host import failure after chars copy");
  case strings_host_import_fault::after_null_mask_copy:
    throw std::runtime_error(
        "test strings host import failure after null mask copy");
  case strings_host_import_fault::before_make_strings_column:
    throw std::runtime_error(
        "test strings host import failure before make_strings_column");
  case strings_host_import_fault::after_make_strings_column:
    throw std::runtime_error(
        "test strings host import failure after make_strings_column");
  case strings_host_import_fault::none:
    break;
  }
}

class host_source_copy_sync_guard {
public:
  explicit host_source_copy_sync_guard(rmm::cuda_stream_view stream)
      : stream_{stream} {}

  host_source_copy_sync_guard(host_source_copy_sync_guard const &) = delete;

  host_source_copy_sync_guard &
  operator=(host_source_copy_sync_guard const &) = delete;

  ~host_source_copy_sync_guard() noexcept {
    if (!copy_may_be_pending_) {
      return;
    }
    try {
      stream_.synchronize();
      cudf_record_synchronized_stream(stream_);
    } catch (...) {
      // Destructors cannot report synchronization failures without masking the
      // original error; explicit synchronize calls still propagate failures.
    }
  }

  void mark_copy_may_be_pending() noexcept { copy_may_be_pending_ = true; }

  void synchronize_and_release() {
    stream_.synchronize();
    cudf_record_synchronized_stream(stream_);
    copy_may_be_pending_ = false;
  }

private:
  rmm::cuda_stream_view stream_;
  bool copy_may_be_pending_ = false;
};

} // namespace

static cudf_status_t create_column_from_device_data_impl(
    cudf_type_id_t type_id, const void *device_data, int32_t size,
    cudf_exec_options_t const *options, bool synchronize_before_return,
    cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (size > 0) {
    CUDF_RUST_NULL_CHECK(device_data, "device_data is null");
  }
  CUDF_RUST_TRY {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    if (size == 0) {
      auto col = cudf::make_empty_column(dtype);
      *out = cudf_release_column_handle(std::move(col),
                                        cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    auto elem_size = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    rmm::device_buffer dev_data(data_bytes, exec->stream, *exec->mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dev_data.data(), device_data, data_bytes,
                                  cudaMemcpyDeviceToDevice,
                                  exec->stream.value()));
    if (synchronize_before_return) {
      exec->stream.synchronize();
      cudf_record_synchronized_stream(exec->stream);
    }

    auto col = std::make_unique<cudf::column>(
        dtype, static_cast<cudf::size_type>(size), std::move(dev_data),
        rmm::device_buffer{}, 0);

    *out = cudf_release_column_handle(std::move(col),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

extern "C" {
void cudf_test_fail_next_column_view_metadata_read() {
  g_fail_next_column_view_metadata_read = true;
}

void cudf_test_fail_next_fixed_width_host_import_after_data_copy() {
  g_next_fixed_width_host_import_fault =
      fixed_width_host_import_fault::after_data_copy;
}

void cudf_test_fail_next_fixed_width_host_import_after_null_mask_copy() {
  g_next_fixed_width_host_import_fault =
      fixed_width_host_import_fault::after_null_mask_copy;
}

void cudf_test_fail_next_strings_host_import_after_offsets_copy() {
  g_next_strings_host_import_fault =
      strings_host_import_fault::after_offsets_copy;
}

void cudf_test_fail_next_strings_host_import_after_chars_copy() {
  g_next_strings_host_import_fault =
      strings_host_import_fault::after_chars_copy;
}

void cudf_test_fail_next_strings_host_import_after_null_mask_copy() {
  g_next_strings_host_import_fault =
      strings_host_import_fault::after_null_mask_copy;
}

void cudf_test_fail_next_strings_host_import_before_make_strings_column() {
  g_next_strings_host_import_fault =
      strings_host_import_fault::before_make_strings_column;
}

void cudf_test_fail_next_strings_host_import_after_make_strings_column() {
  g_next_strings_host_import_fault =
      strings_host_import_fault::after_make_strings_column;
}

cudf_status_t cudf_test_column_view_create_synthetic_fixed_width(
    cudf_type_id_t type_id, int32_t size, int32_t offset,
    cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (offset < 0) {
    set_error("offset must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("synthetic test view only supports fixed-width types");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto *view = new cudf::column_view(
        dtype, static_cast<cudf::size_type>(size),
        static_cast<void const *>(g_test_column_view_dummy_storage.data()),
        nullptr, 0, static_cast<cudf::size_type>(offset));
    *out = static_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_test_column_view_create_synthetic_string_without_children(
    int32_t size, cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size <= 0) {
    set_error("size must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto *view = new cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                       static_cast<cudf::size_type>(size),
                                       nullptr, nullptr, 0, 0, {});
    *out = static_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_empty(cudf_type_id_t type_id,
                                       cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto col = cudf::make_empty_column(dtype);
    *out = cudf_release_column_handle(std::move(col));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_numeric_ex(cudf_type_id_t type_id,
                                            const void *data, int32_t size,
                                            const uint8_t *null_mask,
                                            int32_t null_count,
                                            cudf_exec_options_t const *options,
                                            cudf_column_handle *out) {
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
  if (size > 0) {
    CUDF_RUST_NULL_CHECK(data, "data is null");
  }
  if (null_count > 0) {
    CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0");
  }
  CUDF_RUST_TRY {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    // Compute element size for the given fixed-width type.
    auto elem_size = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    // Copy host data to device.
    std::optional<rmm::device_buffer> dev_data;
    std::optional<rmm::device_buffer> dev_null_mask;
    host_source_copy_sync_guard host_copy_guard{exec->stream};
    if (data_bytes > 0) {
      host_copy_guard.mark_copy_may_be_pending();
    }
    dev_data.emplace(data, data_bytes, exec->stream, *exec->mr);

    // Copy null mask to device if provided.
    if (null_mask != nullptr && null_count > 0) {
      // cudf bitmask is stored as uint32_t words.
      size_t mask_bytes = cudf::bitmask_allocation_size_bytes(
          static_cast<cudf::size_type>(size));
      // We receive packed bytes from Rust — copy them to a device buffer
      // of the right alignment size.
      if (mask_bytes > 0) {
        host_copy_guard.mark_copy_may_be_pending();
      }
      dev_null_mask.emplace(null_mask, mask_bytes, exec->stream, *exec->mr);
    }

    host_copy_guard.synchronize_and_release();

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) {
      null_mask_buffer = std::move(*dev_null_mask);
    }
    auto col = std::make_unique<cudf::column>(
        dtype, static_cast<cudf::size_type>(size), std::move(*dev_data),
        std::move(null_mask_buffer), static_cast<cudf::size_type>(null_count));

    *out = cudf_release_column_handle(std::move(col),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_fixed_width_from_host_ex(
    cudf_type_id_t type_id, int32_t scale, const void *data, int32_t size,
    const uint8_t *null_mask, int32_t null_count,
    cudf_exec_options_t const *options, cudf_column_handle *out) {
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
  if (size > 0) {
    CUDF_RUST_NULL_CHECK(data, "data is null");
  }
  if (null_count > 0) {
    CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0");
  }
  CUDF_RUST_TRY {
    auto const dtype = host_fixed_width_dtype(type_id, scale);
    if (!cudf::is_fixed_width(dtype) || type_id == CUDF_TYPE_DICTIONARY32) {
      set_error("fixed-width host column construction requires a "
                "non-dictionary fixed-width type");
      return CUDF_ERROR_TYPE;
    }

    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto const elem_size = cudf::size_of(dtype);
    auto const data_bytes = static_cast<size_t>(size) * elem_size;
    std::vector<uint8_t> padded_mask;
    std::optional<rmm::device_buffer> dev_data;
    std::optional<rmm::device_buffer> dev_null_mask;
    host_source_copy_sync_guard host_copy_guard{exec->stream};
    if (data_bytes > 0) {
      host_copy_guard.mark_copy_may_be_pending();
    }
    dev_data.emplace(data, data_bytes, exec->stream, *exec->mr);
    maybe_fail_fixed_width_host_import(
        fixed_width_host_import_fault::after_data_copy);

    if (null_mask != nullptr && size > 0) {
      auto const compact_mask_bytes = static_cast<size_t>((size + 7) / 8);
      auto const alloc_mask_bytes = cudf::bitmask_allocation_size_bytes(
          static_cast<cudf::size_type>(size));
      padded_mask.assign(alloc_mask_bytes, 0);
      std::memcpy(padded_mask.data(), null_mask, compact_mask_bytes);
      if (alloc_mask_bytes > 0) {
        host_copy_guard.mark_copy_may_be_pending();
      }
      dev_null_mask.emplace(padded_mask.data(), alloc_mask_bytes, exec->stream,
                            *exec->mr);
      maybe_fail_fixed_width_host_import(
          fixed_width_host_import_fault::after_null_mask_copy);
    }

    host_copy_guard.synchronize_and_release();

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) {
      null_mask_buffer = std::move(*dev_null_mask);
    }
    auto col = std::make_unique<cudf::column>(
        dtype, static_cast<cudf::size_type>(size), std::move(*dev_data),
        std::move(null_mask_buffer), static_cast<cudf::size_type>(null_count));

    *out = cudf_release_column_handle(std::move(col),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_sequence_ex(cudf_scalar_handle init,
                                             cudf_scalar_handle step,
                                             int32_t size,
                                             cudf_exec_options_t const *options,
                                             cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(init, "init is null");
  CUDF_RUST_NULL_CHECK(step, "step is null");
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto const &init_scalar = cudf_scalar_from_handle_const(init);
    auto const &step_scalar = cudf_scalar_from_handle_const(step);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto col = cudf::sequence(static_cast<cudf::size_type>(size), init_scalar,
                                step_scalar, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(col),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_ex(cudf_column_handle src,
                                  cudf_exec_options_t const *options,
                                  cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto copy =
          cudf_column_state_from_handle(src).deep_copy(exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(copy),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_from_view_ex(cudf_column_view_handle src,
                                            cudf_exec_options_t const *options,
                                            cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY {
    auto const &view = *static_cast<cudf::column_view const *>(src);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    {
      auto copy = std::make_unique<cudf::column>(view, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(copy),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_from_borrowed_view_ex(
    cudf_type_id_t type_id, int32_t scale, int32_t size, int32_t offset,
    void const *data, void const *null_mask, int32_t null_count,
    cudf_exec_options_t const *options, cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (offset < 0) {
    set_error("offset must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (null_count < 0 || null_count > size) {
    set_error("null_count must be in [0, size]");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (size > 0 && data == nullptr) {
    set_error("data is null");
    return CUDF_ERROR_NULL;
  }
  if (null_count > 0 && null_mask == nullptr) {
    set_error("null_mask is null when null_count > 0");
    return CUDF_ERROR_NULL;
  }
  CUDF_RUST_TRY {
    auto const id = static_cast<cudf::type_id>(type_id);
    cudf::data_type const dtype = id == cudf::type_id::DECIMAL32 ||
                                          id == cudf::type_id::DECIMAL64 ||
                                          id == cudf::type_id::DECIMAL128
                                      ? cudf::data_type{id, scale}
                                      : cudf::data_type{id};
    if (!cudf::is_fixed_width(dtype)) {
      set_error("borrowed descriptor import only supports fixed-width types");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    cudf::column_view const view{
        dtype,
        static_cast<cudf::size_type>(size),
        data,
        static_cast<cudf::bitmask_type const *>(null_mask),
        static_cast<cudf::size_type>(null_count),
        static_cast<cudf::size_type>(offset)};
    {
      auto copy = std::make_unique<cudf::column>(view, exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(copy),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_from_owned_device_transfer(
    uint32_t root_column_id,
    cudf_interop_semantic_column_t const *semantic_columns,
    int32_t num_semantic_columns, cudf_interop_buffer_binding_t const *bindings,
    int32_t num_bindings, cudf_column_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto state = cudf_make_foreign_column_handle_state(
        root_column_id, semantic_columns, num_semantic_columns, bindings,
        num_bindings);
    *out = cudf_release_column_handle(std::move(state));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_payload_size_ex(cudf_column_handle handle,
                                          cudf_exec_options_t const *options,
                                          size_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    *out = cudf_column_state_from_handle(handle).payload_size(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_device_allocated_size(cudf_column_handle handle,
                                                size_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    *out = cudf_column_state_from_handle(handle).device_allocated_size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_column_destroy(cudf_column_handle handle) {
  if (handle == nullptr) {
    return;
  }
  auto *state = static_cast<cudf_column_handle_state *>(handle);
  if (auto provenance = state->destroy_provenance()) {
    auto const cleanup =
        cudf_destroy_under_provenance(*provenance, [state]() { delete state; });
    if (cleanup != cudf_provenance_cleanup_result::completed) {
      std::terminate();
    }
  } else {
    delete state;
  }
}

cudf_status_t cudf_column_view(cudf_column_handle handle,
                               cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view =
        new cudf::column_view(cudf_column_state_from_handle(handle).view());
    *out = static_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_column_view_destroy(cudf_column_view_handle handle) {
  delete static_cast<cudf::column_view *>(handle);
}

cudf_status_t cudf_column_view_type(cudf_column_view_handle handle,
                                    cudf_type_id_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = CUDF_TYPE_EMPTY;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = static_cast<cudf_type_id_t>(view->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_data_type(cudf_column_view_handle handle,
                                         cudf_type_id_t *out_type_id,
                                         int32_t *out_scale) {
  CUDF_RUST_NULL_CHECK(out_type_id, "out_type_id is null");
  CUDF_RUST_NULL_CHECK(out_scale, "out_scale is null");
  *out_type_id = CUDF_TYPE_EMPTY;
  *out_scale = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    auto dtype = view->type();
    *out_type_id = static_cast<cudf_type_id_t>(dtype.id());
    *out_scale = dtype.scale();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_size(cudf_column_view_handle handle,
                                    int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  cudf_status_t const fail_status = maybe_fail_column_view_metadata_read();
  if (fail_status != CUDF_SUCCESS) {
    return fail_status;
  }
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = view->size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle,
                                          int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  cudf_status_t const fail_status = maybe_fail_column_view_metadata_read();
  if (fail_status != CUDF_SUCCESS) {
    return fail_status;
  }
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = view->null_count();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle,
                                        int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = view->nullable() ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle,
                                            int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = view->num_children();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_child(cudf_column_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    auto *child = new cudf::column_view(view->child(index));
    *out = static_cast<cudf_column_view_handle>(child);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_column_view_list_sliced_child_ex(cudf_column_view_handle handle,
                                      cudf_exec_options_t const *options,
                                      cudf_column_view_handle *out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = static_cast<cudf::column_view *>(handle);
    if (view->type().id() != cudf::type_id::LIST) {
      set_error("column is not LIST type");
      return CUDF_ERROR_TYPE;
    }

    cudf::lists_column_view const lists{*view};
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto *child = new cudf::column_view(lists.get_sliced_child(exec->stream));
      *out = static_cast<cudf_column_view_handle>(child);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_data_ptr(cudf_column_view_handle handle,
                                        const void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    // data<uint8_t>() returns the base device pointer. For STRING columns
    // this is nullptr — callers should use child views instead.
    *out = view->head<uint8_t>();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_string_chars_ptr_and_size_ex(
    cudf_column_view_handle handle, cudf_exec_options_t const *options,
    const void **out_ptr, size_t *out_size) {
  CUDF_RUST_NULL_CHECK(out_ptr, "out_ptr is null");
  CUDF_RUST_NULL_CHECK(out_size, "out_size is null");
  *out_ptr = nullptr;
  *out_size = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    if (view->type().id() != cudf::type_id::STRING) {
      set_error("column type is not STRING");
      return CUDF_ERROR_TYPE;
    }

    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    cudf::strings_column_view scv(*view);
    cudf::size_type first_offset = 0;
    cudf::size_type last_offset = 0;
    {
      auto const offsets =
          cudf::strings::detail::get_first_and_last_offset(scv, exec->stream);
      first_offset = static_cast<cudf::size_type>(offsets.first);
      last_offset = static_cast<cudf::size_type>(offsets.second);
    }
    if (first_offset < 0 || last_offset < first_offset) {
      set_error("string chars span is invalid");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto const active_bytes = static_cast<uint64_t>(last_offset - first_offset);
    if (active_bytes >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      set_error("string chars span exceeds size_t range");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    *out_ptr = scv.chars_begin(exec->stream);
    *out_size = static_cast<size_t>(active_bytes);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_mask_ptr(cudf_column_view_handle handle,
                                             const void **out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    if (!view->nullable()) {
      set_error("Column has no null mask");
      return CUDF_ERROR_NULL;
    }
    *out = view->null_mask();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_offset(cudf_column_view_handle handle,
                                      int32_t *out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    *out = view->offset();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_slice_ex(cudf_column_view_handle handle,
                                        int32_t offset, int32_t length,
                                        cudf_exec_options_t const *options,
                                        cudf_column_view_handle *out) {
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

  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = static_cast<cudf::column_view *>(handle);
    auto const start = static_cast<cudf::size_type>(offset);
    auto const len = static_cast<cudf::size_type>(length);
    if (start > view->size() || len > view->size() - start) {
      set_error("slice range is out of bounds");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    constexpr auto max_size_type =
        static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max());
    auto const active_begin64 =
        static_cast<int64_t>(view->offset()) + static_cast<int64_t>(start);
    auto const active_end64 = active_begin64 + static_cast<int64_t>(len);
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

    auto const null_count =
        view->nullable() ? view->null_count(start, start + len, exec->stream)
                         : 0;

    auto *slice = new cudf::column_view(view->type(), len, view->head(),
                                        view->null_mask(), null_count,
                                        active_begin, children);
    *out = static_cast<cudf_column_view_handle>(slice);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_to_host_ex(cudf_column_view_handle handle,
                                          void *host_buf, size_t buf_size,
                                          cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = static_cast<cudf::column_view *>(handle);
    auto dtype = view->type();
    if (!cudf::is_fixed_width(dtype)) {
      set_error("column_view_to_host only supports fixed-width types");
      return CUDF_ERROR_TYPE;
    }
    auto elem_size = cudf::size_of(dtype);
    size_t needed = static_cast<size_t>(view->size()) * elem_size;
    if (buf_size < needed) {
      set_error("Buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    // Copy from device to host.
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_buf, view->data<uint8_t>(), needed,
                                  cudaMemcpyDeviceToHost,
                                  exec->stream.value()));
    // This API promises the caller's host buffer is populated on return.
    exec->stream.synchronize();
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t
cudf_column_view_null_mask_to_host_ex(cudf_column_view_handle handle,
                                      uint8_t *host_buf, size_t buf_size,
                                      cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY {
    auto *view = static_cast<cudf::column_view *>(handle);
    if (!view->nullable()) {
      set_error("Column has no null mask");
      return CUDF_ERROR_NULL;
    }
    size_t mask_bytes = cudf::bitmask_allocation_size_bytes(view->size());
    if (buf_size < mask_bytes) {
      set_error("Buffer too small for null mask");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    if (mask_bytes == 0) {
      return CUDF_SUCCESS;
    }

    // Zero the full aligned allocation so bits beyond the logical slice
    // (mask_bytes may be up to 63 bytes larger than compact_mask_bytes due
    // to cuDF's 64-byte bitmask alignment) are deterministically zero.
    std::memset(host_buf, 0, mask_bytes);
    size_t compact_mask_bytes = (static_cast<size_t>(view->size()) + 7) / 8;
    std::optional<rmm::device_buffer> normalized_mask;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      normalized_mask.emplace(
          cudf::copy_bitmask(*view, exec->stream, *exec->mr));
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_buf, normalized_mask->data(),
                                  compact_mask_bytes, cudaMemcpyDeviceToHost,
                                  exec->stream.value()));
    // This API promises the caller's host buffer is populated on return.
    exec->stream.synchronize();
    if ((view->size() % 8) != 0) {
      auto const last_byte_mask =
          static_cast<uint8_t>((1u << (view->size() % 8)) - 1u);
      host_buf[compact_mask_bytes - 1] &= last_byte_mask;
    }
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_from_device_data(cudf_type_id_t type_id,
                                                  const void *device_data,
                                                  int32_t size,
                                                  cudf_column_handle *out) {
  return create_column_from_device_data_impl(type_id, device_data, size,
                                             nullptr, true, out);
}

cudf_status_t cudf_column_create_from_device_data_ex(
    cudf_type_id_t type_id, const void *device_data, int32_t size,
    cudf_exec_options_t const *options, cudf_column_handle *out) {
  return create_column_from_device_data_impl(type_id, device_data, size,
                                             options, false, out);
}

cudf_status_t cudf_column_create_strings_ex(
    const char **strings, const int32_t *lengths, int32_t num_strings,
    const uint8_t *null_mask, int32_t null_count,
    cudf_exec_options_t const *options, cudf_column_handle *out) {
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
  if (null_count > 0) {
    CUDF_RUST_NULL_CHECK(null_mask, "null_mask is null when null_count > 0");
  }
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    if (num_strings == 0) {
      auto col =
          cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
      *out = cudf_release_column_handle(std::move(col),
                                        cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    // Compute offsets: offsets[0] = 0, offsets[i+1] = offsets[i] + lengths[i]
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
      auto const next_offset = static_cast<int64_t>(host_offsets[i]) +
                               static_cast<int64_t>(lengths[i]);
      if (next_offset > std::numeric_limits<int32_t>::max()) {
        set_error("total string bytes exceed INT32_MAX");
        return CUDF_ERROR_INVALID_OPERATION;
      }
      host_offsets[i + 1] = static_cast<int32_t>(next_offset);
    }
    int32_t total_chars = host_offsets[num_strings];

    // Build a single concatenated chars buffer on host
    std::vector<char> host_chars(total_chars);
    for (int32_t i = 0; i < num_strings; ++i) {
      if (lengths[i] > 0 && strings[i] != nullptr) {
        std::memcpy(host_chars.data() + host_offsets[i], strings[i],
                    lengths[i]);
      }
    }

    std::vector<uint8_t> padded_mask;
    std::optional<rmm::device_buffer> dev_offsets;
    std::optional<rmm::device_buffer> dev_chars;
    std::optional<rmm::device_buffer> dev_null_mask;
    host_source_copy_sync_guard host_copy_guard{exec->stream};

    auto const offsets_count = num_strings_size + 1;
    auto const offsets_bytes = offsets_count * sizeof(int32_t);
    if (offsets_bytes > 0) {
      host_copy_guard.mark_copy_may_be_pending();
    }
    dev_offsets.emplace(host_offsets.data(), offsets_bytes, exec->stream,
                        *exec->mr);
    maybe_fail_strings_host_import(
        strings_host_import_fault::after_offsets_copy);

    auto const chars_bytes = static_cast<size_t>(total_chars);
    if (chars_bytes > 0) {
      host_copy_guard.mark_copy_may_be_pending();
    }
    dev_chars.emplace(host_chars.data(), chars_bytes, exec->stream, *exec->mr);
    maybe_fail_strings_host_import(strings_host_import_fault::after_chars_copy);

    if (null_mask != nullptr && null_count > 0) {
      auto const compact_mask_bytes = (num_strings_size + 7) / 8;
      auto const alloc_mask_bytes = cudf::bitmask_allocation_size_bytes(
          static_cast<cudf::size_type>(num_strings));
      padded_mask.assign(alloc_mask_bytes, 0);
      std::memcpy(padded_mask.data(), null_mask, compact_mask_bytes);
      if (alloc_mask_bytes > 0) {
        host_copy_guard.mark_copy_may_be_pending();
      }
      dev_null_mask.emplace(padded_mask.data(), alloc_mask_bytes, exec->stream,
                            *exec->mr);
      maybe_fail_strings_host_import(
          strings_host_import_fault::after_null_mask_copy);
    }

    maybe_fail_strings_host_import(
        strings_host_import_fault::before_make_strings_column);
    host_copy_guard.synchronize_and_release();

    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::INT32),
        static_cast<cudf::size_type>(offsets_count), std::move(*dev_offsets),
        rmm::device_buffer{}, 0);

    rmm::device_buffer null_mask_buffer{};
    if (dev_null_mask.has_value()) {
      null_mask_buffer = std::move(*dev_null_mask);
    }
    auto col = cudf::make_strings_column(
        static_cast<cudf::size_type>(num_strings), std::move(offsets_col),
        std::move(*dev_chars), static_cast<cudf::size_type>(null_count),
        std::move(null_mask_buffer));
    maybe_fail_strings_host_import(
        strings_host_import_fault::after_make_strings_column);

    *out = cudf_release_column_handle(std::move(col),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_strings_to_host_ex(
    cudf_column_view_handle handle, int32_t *offsets_buf,
    size_t offsets_buf_size, char *chars_buf, size_t chars_buf_size,
    int32_t *total_chars_bytes, cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(total_chars_bytes, "total_chars_bytes is null");
  *total_chars_bytes = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }

    auto *view = static_cast<cudf::column_view *>(handle);
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
      offsets_buf[0] = 0;
      *total_chars_bytes = 0;
      return CUDF_SUCCESS;
    }

    // String columns: child 0 = offsets, child 1 = chars
    // For the *sliced* view, we need n+1 offsets (not the unsliced child size,
    // which may be larger when the view is a slice of a larger column).
    if (view->num_children() < 1) {
      set_error("string column view must expose an offsets child");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto offsets_view = view->child(0);
    auto offsets_count = n + 1;

    // Copy offsets to host
    size_t needed_offsets_bytes =
        static_cast<size_t>(offsets_count) * sizeof(int32_t);
    CUDF_RUST_NULL_CHECK(offsets_buf, "offsets_buf is null");
    if (offsets_buf_size < needed_offsets_bytes) {
      set_error("offsets buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    // Account for view offset in the offsets child
    auto offsets_data = offsets_view.data<int32_t>();
    CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_buf, offsets_data,
                                  needed_offsets_bytes, cudaMemcpyDeviceToHost,
                                  exec->stream.value()));
    // We need the copied offsets on host before we can size the chars copy.
    exec->stream.synchronize();
    cudf_record_synchronized_stream(exec->stream);

    // Compute total chars bytes from offsets
    int32_t char_begin = offsets_buf[0];
    int32_t char_end = offsets_buf[offsets_count - 1];
    int32_t total = char_end - char_begin;
    *total_chars_bytes = total;

    if (total > 0) {
      auto const total_chars = static_cast<size_t>(total);
      if (chars_buf != nullptr) {
        if (chars_buf_size < total_chars) {
          set_error("chars buffer too small");
          return CUDF_ERROR_INVALID_OPERATION;
        }

        // In cudf 25.x the chars are embedded in the offsets column's parent
        // We need to get chars from the strings_column_view
        cudf::strings_column_view scv(*view);
        auto chars_ptr = scv.chars_begin(exec->stream);

        // chars_begin does NOT account for view offset, but the offsets
        // already tell us where chars start (offsets_buf[0] = char_begin).
        CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buf, chars_ptr + char_begin,
                                      total_chars, cudaMemcpyDeviceToHost,
                                      exec->stream.value()));
        // This API promises the caller's chars buffer is populated on return.
        exec->stream.synchronize();
        cudf_record_synchronized_stream(exec->stream);
      }
      // If chars_buf is null, we just return total_chars_bytes without copying.
    }

    // Adjust offsets to be zero-based for the caller
    for (int i = 0; i < offsets_count; ++i) {
      offsets_buf[i] -= char_begin;
    }

    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_from_scalar_ex(cudf_scalar_handle scalar,
                                         int32_t size, cudf_column_handle *out,
                                         cudf_exec_options_t const *options) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(scalar, "scalar is null");
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto const &sc = cudf_scalar_from_handle_const(scalar);
    auto exec = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status =
        cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) {
      return exec_status;
    }
    {
      // libcudf's LIST scalar broadcast path still allocates nested
      // temporaries from RMM's current resource despite receiving an explicit
      // memory_resource_ref. Keep current aligned with the explicit MR for
      // this allocation boundary.
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto col = cudf::make_column_from_scalar(
          sc, static_cast<cudf::size_type>(size), exec->stream, *exec->mr);
      *out = cudf_release_column_handle(std::move(col),
                                        cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
} // extern "C"
