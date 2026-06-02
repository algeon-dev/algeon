#include "device_buffer.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

static cudf::type_id to_cudf_type_id(cudf_type_id_t id) {
  return static_cast<cudf::type_id>(id);
}

extern "C" {

cudf_status_t cudf_device_buffer_create(size_t size,
                                        cudf_device_buffer_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto* buf = new rmm::device_buffer(size, stream, mr);
    *out = static_cast<cudf_device_buffer_handle>(buf);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_device_buffer_destroy(cudf_device_buffer_handle handle) {
  delete static_cast<rmm::device_buffer*>(handle);
}

cudf_status_t cudf_device_buffer_data(cudf_device_buffer_handle handle,
                                      void** out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  auto* buf = static_cast<rmm::device_buffer*>(handle);
  *out = buf->data();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_device_buffer_size(cudf_device_buffer_handle handle,
                                      size_t* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  auto* buf = static_cast<rmm::device_buffer*>(handle);
  *out = buf->size();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_device_buffer_copy_from_host(
    cudf_device_buffer_handle handle,
    const void* host_data,
    size_t size) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_data, "host_data is null");
  auto* buf = static_cast<rmm::device_buffer*>(handle);
  if (size > buf->size()) {
    set_error("copy size exceeds buffer size");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY {
    auto stream = cudf::get_default_stream();
    CUDF_CUDA_TRY(cudaMemcpyAsync(buf->data(), host_data, size,
                                  cudaMemcpyHostToDevice, stream.value()));
    stream.synchronize();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_adopt_device_buffer(
    cudf_type_id_t type_id,
    cudf_device_buffer_handle buffer,
    int32_t row_count,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (row_count < 0) {
    set_error("row_count must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(buffer, "buffer is null");
  CUDF_RUST_TRY {
    auto dtype = cudf::data_type(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("only fixed-width types are supported for buffer adoption");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* buf = static_cast<rmm::device_buffer*>(buffer);
    auto elem_size = cudf::size_of(dtype);
    size_t expected = static_cast<size_t>(row_count) * elem_size;
    if (buf->size() != expected) {
      set_error("buffer size does not match row_count * element_size");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    // Move the rmm::device_buffer into the column — zero copy.
    auto col = std::make_unique<cudf::column>(
        dtype,
        static_cast<cudf::size_type>(row_count),
        std::move(*buf),           // ownership transfer
        rmm::device_buffer{},      // no null mask
        0);                        // null_count = 0

    // The original rmm::device_buffer object is now moved-from.
    // Delete the heap wrapper so the Rust side doesn't double-free.
    delete buf;

    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_adopt_device_buffer_nullable(
    cudf_type_id_t type_id,
    cudf_device_buffer_handle data_buffer,
    cudf_device_buffer_handle null_mask_buffer,
    int32_t null_count,
    int32_t row_count,
    cudf_column_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (row_count < 0) {
    set_error("row_count must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (null_count < 0 || null_count > row_count) {
    set_error("null_count must be in [0, row_count]");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(data_buffer, "data_buffer is null");
  CUDF_RUST_NULL_CHECK(null_mask_buffer, "null_mask_buffer is null");
  CUDF_RUST_TRY {
    auto dtype = cudf::data_type(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("only fixed-width types are supported for buffer adoption");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* data_buf = static_cast<rmm::device_buffer*>(data_buffer);
    auto* mask_buf = static_cast<rmm::device_buffer*>(null_mask_buffer);

    auto elem_size = cudf::size_of(dtype);
    size_t expected_data = static_cast<size_t>(row_count) * elem_size;
    if (data_buf->size() != expected_data) {
      set_error("data buffer size does not match row_count * element_size");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    size_t min_mask = cudf::bitmask_allocation_size_bytes(
        static_cast<cudf::size_type>(row_count));
    if (mask_buf->size() < min_mask) {
      set_error("null mask buffer is too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    // Move both rmm::device_buffers into the column — zero copy.
    auto col = std::make_unique<cudf::column>(
        dtype,
        static_cast<cudf::size_type>(row_count),
        std::move(*data_buf),
        std::move(*mask_buf),
        static_cast<cudf::size_type>(null_count));

    // Both original rmm::device_buffer objects are now moved-from.
    // Delete the heap wrappers.
    delete data_buf;
    delete mask_buf;

    *out = static_cast<cudf_column_handle>(col.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
