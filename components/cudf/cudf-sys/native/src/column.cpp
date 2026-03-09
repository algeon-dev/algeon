#include "column.h"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>

#include <cstring>
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
  *out = nullptr;
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
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto* col = static_cast<cudf::column*>(src);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    auto copy = std::make_unique<cudf::column>(*col, stream, mr);
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
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
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
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = static_cast<cudf_type_id_t>(view->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_size(cudf_column_view_handle handle,
                                    int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->size();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_null_count(cudf_column_view_handle handle,
                                          int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->null_count();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_nullable(cudf_column_view_handle handle,
                                        int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    *out = view->nullable() ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_num_children(cudf_column_view_handle handle,
                                            int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
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
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    auto* child = new cudf::column_view(view->child(index));
    *out = static_cast<cudf_column_view_handle>(child);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_view_to_host(cudf_column_view_handle handle,
                                       void* host_buf, size_t buf_size) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_buf, "host_buf is null");
  CUDF_RUST_TRY {
    auto* view = static_cast<cudf::column_view*>(handle);
    auto dtype = view->type();
    auto elem_size = cudf::size_of(dtype);
    size_t needed = static_cast<size_t>(view->size()) * elem_size;
    if (buf_size < needed) {
      set_error("Buffer too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    auto stream = cudf::get_default_stream();
    // Copy from device to host.
    cudaMemcpyAsync(host_buf, view->data<uint8_t>(),
                    needed, cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();
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
    cudaMemcpyAsync(host_buf, view->null_mask(), mask_bytes,
                    cudaMemcpyDeviceToHost, stream.value());
    stream.synchronize();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
