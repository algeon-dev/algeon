#include "column.h"
#include "execution_context_internal.hpp"
#include "scalar.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>

#include <cstring>
#include <limits>
#include <memory>

static cudf::type_id to_cudf_type_id(cudf_type_id_t id) {
  return static_cast<cudf::type_id>(id);
}

extern "C" {

cudf_status_t cudf_column_create_empty(cudf_type_id_t type_id,
                                       cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto dtype = cudf::data_type(to_cudf_type_id(type_id));
    auto col = cudf::make_empty_column(dtype);
    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_numeric(cudf_type_id_t type_id,
                                         const void* data, int32_t size,
                                         const uint8_t* null_mask,
                                         int32_t null_count,
                                         cudf_column_handle* out) {
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
    auto dtype = cudf::data_type(to_cudf_type_id(type_id));
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    // Compute element size for the given fixed-width type.
    auto elem_size = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    // Copy host data to device.
    rmm::device_buffer dev_data(data, data_bytes, stream, mr);

    // Copy null mask to device if provided.
    rmm::device_buffer dev_null_mask{};
    if (null_mask != nullptr && null_count > 0) {
      // cudf bitmask is stored as uint32_t words.
      size_t mask_bytes =
          cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(size));
      // We receive packed bytes from Rust — copy them to a device buffer
      // of the right alignment size.
      dev_null_mask = rmm::device_buffer(null_mask, mask_bytes, stream, mr);
    }

    auto col = std::make_unique<cudf::column>(
        dtype, static_cast<cudf::size_type>(size), std::move(dev_data),
        std::move(dev_null_mask), static_cast<cudf::size_type>(null_count));

    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy(cudf_column_handle src,
                               cudf_column_handle* out) {
  return cudf_column_copy_ex(src, nullptr, out);
}

cudf_status_t cudf_column_copy_ex(cudf_column_handle src,
                                  cudf_exec_options_t const* options,
                                  cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY {
    auto exec = cudf_exec_context{cudf::get_default_stream(),
                                  cudf::get_current_device_resource_ref(),
                                  nullptr};
    auto const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* col = static_cast<cudf::column*>(src);
    auto copy = std::make_unique<cudf::column>(*col, exec.stream, exec.mr);
    *out = static_cast<cudf_column_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_copy_from_view(cudf_column_view_handle src,
                                         cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY {
    auto const& view = *static_cast<cudf::column_view const*>(src);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto copy = std::make_unique<cudf::column>(view, stream, mr);
    *out = static_cast<cudf_column_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_column_destroy(cudf_column_handle handle) {
  delete static_cast<cudf::column*>(handle);
}

cudf_status_t cudf_column_view(cudf_column_handle handle,
                               cudf_column_view_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* col = static_cast<cudf::column*>(handle);
    auto* view = new cudf::column_view(col->view());
    *out = static_cast<cudf_column_view_handle>(view);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_column_view_destroy(cudf_column_view_handle handle) {
  delete static_cast<cudf::column_view*>(handle);
}

cudf_status_t cudf_column_view_type(cudf_column_view_handle handle,
                                    cudf_type_id_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = CUDF_TYPE_EMPTY;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = static_cast<cudf_type_id_t>(view->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_size(cudf_column_view_handle handle,
                                    int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle,
                                          int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->null_count();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle,
                                        int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->nullable() ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle,
                                            int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->num_children();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_child(cudf_column_view_handle handle,
                                     int32_t index,
                                     cudf_column_view_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    auto* child = new cudf::column_view(view->child(index));
    *out = static_cast<cudf_column_view_handle>(child);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_data_ptr(cudf_column_view_handle handle,
                                        const void** out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    // data<uint8_t>() returns the base device pointer. For STRING columns
    // this is nullptr — callers should use child views instead.
    *out = view->head<uint8_t>();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_offset(cudf_column_view_handle handle,
                                      int32_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->offset();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_to_host(cudf_column_view_handle handle,
                                       void* host_buf, size_t buf_size) {
  return cudf_column_view_to_host_ex(handle, host_buf, buf_size, nullptr);
}

cudf_status_t cudf_column_view_to_host_ex(cudf_column_view_handle handle,
                                          void* host_buf,
                                          size_t buf_size,
                                          cudf_exec_options_t const* options) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY {
    auto exec = cudf_exec_context{cudf::get_default_stream(),
                                  cudf::get_current_device_resource_ref(),
                                  nullptr};
    auto const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::column_view*>(handle);
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
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_buf, view->data<uint8_t>(),
                                  needed, cudaMemcpyDeviceToHost,
                                  exec.stream.value()));
    // This API promises the caller's host buffer is populated on return.
    exec.stream.synchronize();
    cudf_record_synchronized_stream(exec.stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_mask_to_host(
    cudf_column_view_handle handle, uint8_t* host_buf, size_t buf_size) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    if (!view->nullable()) {
      set_error("Column has no null mask");
      return CUDF_ERROR_NULL;
    }
    if (view->offset() != 0) {
      set_error("null mask export for sliced views (offset != 0) is not supported");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    size_t mask_bytes =
        cudf::bitmask_allocation_size_bytes(view->size());
    if (buf_size < mask_bytes) {
      set_error("Buffer too small for null mask");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto stream = cudf::get_default_stream();
    CUDF_CUDA_TRY(cudaMemcpyAsync(host_buf, view->null_mask(), mask_bytes,
                                  cudaMemcpyDeviceToHost, stream.value()));
    // This API promises the caller's host buffer is populated on return.
    stream.synchronize();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_from_device_data(
    cudf_type_id_t type_id,
    const void* device_data,
    int32_t size,
    cudf_column_handle* out) {
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
    auto dtype = cudf::data_type(to_cudf_type_id(type_id));
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    if (size == 0) {
      auto col = cudf::make_empty_column(dtype);
      *out = static_cast<cudf_column_handle>(col.release());
      return CUDF_SUCCESS;
    }

    // Compute data size in bytes.
    auto elem_size = cudf::size_of(dtype);
    size_t data_bytes = static_cast<size_t>(size) * elem_size;

    // Device-to-device copy into a new rmm::device_buffer.
    rmm::device_buffer dev_data(data_bytes, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(dev_data.data(), device_data, data_bytes,
                                  cudaMemcpyDeviceToDevice, stream.value()));
    // The input pointer is borrowed from the caller with no stream handoff.
    // Complete the copy before returning an owning column handle.
    stream.synchronize();

    // Construct an owning column from the device buffer (no null mask).
    auto col = std::make_unique<cudf::column>(
        dtype, static_cast<cudf::size_type>(size), std::move(dev_data),
        rmm::device_buffer{}, 0);

    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_create_strings(
    const char** strings,
    const int32_t* lengths,
    int32_t num_strings,
    const uint8_t* null_mask,
    int32_t null_count,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (num_strings < 0) {
    set_error("num_strings must be >= 0");
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
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    if (num_strings == 0) {
      auto col = cudf::make_empty_column(cudf::data_type(cudf::type_id::STRING));
      *out = static_cast<cudf_column_handle>(col.release());
      return CUDF_SUCCESS;
    }

    // Compute offsets: offsets[0] = 0, offsets[i+1] = offsets[i] + lengths[i]
    std::vector<int32_t> host_offsets(num_strings + 1);
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

    // Build a single concatenated chars buffer on host
    std::vector<char> host_chars(total_chars);
    for (int32_t i = 0; i < num_strings; ++i) {
      if (lengths[i] > 0 && strings[i] != nullptr) {
        std::memcpy(host_chars.data() + host_offsets[i], strings[i], lengths[i]);
      }
    }

    // Upload offsets to device
    size_t offsets_bytes = static_cast<size_t>(num_strings + 1) * sizeof(int32_t);
    rmm::device_buffer dev_offsets(host_offsets.data(), offsets_bytes, stream, mr);
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type(cudf::type_id::INT32),
        static_cast<cudf::size_type>(num_strings + 1),
        std::move(dev_offsets),
        rmm::device_buffer{}, 0);

    // Upload chars to device
    rmm::device_buffer dev_chars(host_chars.data(),
                                 static_cast<size_t>(total_chars), stream, mr);

    // Build null mask on device if needed
    rmm::device_buffer dev_null_mask{};
    if (null_mask != nullptr && null_count > 0) {
      size_t mask_bytes =
          cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(num_strings));
      dev_null_mask = rmm::device_buffer(null_mask, mask_bytes, stream, mr);
    }

    auto col = cudf::make_strings_column(
        static_cast<cudf::size_type>(num_strings),
        std::move(offsets_col),
        std::move(dev_chars),
        static_cast<cudf::size_type>(null_count),
        std::move(dev_null_mask));

    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_strings_to_host(
    cudf_column_view_handle handle,
    int32_t* offsets_buf,
    size_t offsets_buf_size,
    char* chars_buf,
    size_t chars_buf_size,
    int32_t* total_chars_bytes) {
  CUDF_RUST_NULL_CHECK(total_chars_bytes, "total_chars_bytes is null");
  *total_chars_bytes = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    if (view->type().id() != cudf::type_id::STRING) {
      set_error("column is not STRING type");
      return CUDF_ERROR_TYPE;
    }

    auto n = view->size();
    auto stream = cudf::get_default_stream();

    if (n == 0) {
      *total_chars_bytes = 0;
      return CUDF_SUCCESS;
    }

    // String columns: child 0 = offsets, child 1 = chars
    // The offsets child has n+1 elements.
    auto offsets_view = view->child(0);
    auto offsets_count = offsets_view.size();  // should be n+1

    // Copy offsets to host
    size_t needed_offsets_bytes = static_cast<size_t>(offsets_count) * sizeof(int32_t);
    CUDF_RUST_NULL_CHECK(offsets_buf, "offsets_buf is null");
    if (offsets_buf_size < needed_offsets_bytes) {
      set_error("offsets buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    // Account for view offset in the offsets child
    auto offsets_data = offsets_view.data<int32_t>();
    CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_buf, offsets_data,
                                  needed_offsets_bytes, cudaMemcpyDeviceToHost, stream.value()));
    // We need the copied offsets on host before we can size the chars copy.
    stream.synchronize();

    // Compute total chars bytes from offsets
    int32_t char_begin = offsets_buf[0];
    int32_t char_end = offsets_buf[offsets_count - 1];
    int32_t total = char_end - char_begin;
    *total_chars_bytes = total;

    if (total > 0) {
      if (chars_buf != nullptr) {
        if (chars_buf_size < static_cast<size_t>(total)) {
          set_error("chars buffer too small");
          return CUDF_ERROR_INVALID_OPERATION;
        }

        // In cudf 25.x the chars are embedded in the offsets column's parent
        // We need to get chars from the strings_column_view
        cudf::strings_column_view scv(*view);
        auto chars_ptr = scv.chars_begin(stream);

        // chars_begin does NOT account for view offset, but the offsets
        // already tell us where chars start (offsets_buf[0] = char_begin).
        CUDF_CUDA_TRY(cudaMemcpyAsync(chars_buf, chars_ptr + char_begin,
                                      static_cast<size_t>(total),
                                      cudaMemcpyDeviceToHost, stream.value()));
        // This API promises the caller's chars buffer is populated on return.
        stream.synchronize();
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

cudf_status_t cudf_column_from_scalar(cudf_scalar_handle scalar,
                                      int32_t size,
                                      cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(scalar, "scalar is null");
  if (size < 0) {
    set_error("size must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto const& sc = *static_cast<cudf::scalar const*>(scalar);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto col = cudf::make_column_from_scalar(
        sc, static_cast<cudf::size_type>(size), stream, mr);
    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
