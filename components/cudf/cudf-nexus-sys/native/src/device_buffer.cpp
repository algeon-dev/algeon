#include "cudf_rust/device_buffer.h"

#include "cudf_rust/device.h"
#include "execution_context_internal.hpp"
#include "handle_provenance.hpp"
#include "owning_handle.hpp"
#include "shim_common.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <exception>
#include <memory>
#include <optional>
#include <utility>

namespace {
struct cudf_device_buffer_owner {
  cudf_handle_provenance provenance;
  rmm::device_buffer buffer;
};

cudf_device_buffer_handle release_device_buffer_handle(rmm::device_buffer buffer,
                                                       cudf_handle_provenance provenance)
{
  auto owner        = std::make_unique<cudf_device_buffer_owner>();
  owner->buffer     = std::move(buffer);
  owner->provenance = std::move(provenance);
  return reinterpret_cast<cudf_device_buffer_handle>(owner.release());
}

cudf_device_buffer_owner& device_buffer_owner_from_handle(cudf_device_buffer_handle handle)
{
  return *reinterpret_cast<cudf_device_buffer_owner*>(handle);
}

rmm::device_buffer& device_buffer_from_handle(cudf_device_buffer_handle handle)
{
  return device_buffer_owner_from_handle(handle).buffer;
}

class adopted_device_buffer_column_handle_state final : public cudf_column_handle_state {
 public:
  adopted_device_buffer_column_handle_state(cudf::data_type dtype,
                                            cudf::size_type size,
                                            cudf::size_type null_count,
                                            cudf_device_buffer_owner* data_owner,
                                            cudf_device_buffer_owner* null_mask_owner,
                                            cudf_handle_provenance provenance)
    : _provenance{std::move(provenance)},
      _dtype{dtype},
      _size{size},
      _null_count{null_count},
      _pending_data_owner{data_owner},
      _pending_null_mask_owner{null_mask_owner}
  {
  }

  void commit_published_handles() noexcept
  {
    _data_owner.reset(_pending_data_owner);
    _pending_data_owner = nullptr;

    _null_mask_owner.reset(_pending_null_mask_owner);
    _pending_null_mask_owner = nullptr;
  }

  [[nodiscard]] bool is_native() const noexcept override { return true; }

  [[nodiscard]] cudf::size_type size() const override { return _size; }

  [[nodiscard]] cudf::column_view view() const override
  {
    auto const& data_buffer = data_owner().buffer;
    auto const* null_mask =
      null_mask_owner() == nullptr
        ? nullptr
        : static_cast<cudf::bitmask_type const*>(null_mask_owner()->buffer.data());
    return cudf::column_view(_dtype, _size, data_buffer.data(), null_mask, _null_count);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> deep_copy(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return std::make_unique<cudf::column>(view(), stream, mr);
  }

  [[nodiscard]] std::size_t payload_size(rmm::cuda_stream_view) const override
  {
    auto total = static_cast<std::size_t>(_size) * cudf::size_of(_dtype);
    if (null_mask_owner() != nullptr) {
      total +=
        static_cast<std::size_t>(cudf::num_bitmask_words(_size)) * sizeof(cudf::bitmask_type);
    }
    return total;
  }

  [[nodiscard]] std::size_t device_allocated_size() const override
  {
    auto total = data_owner().buffer.size();
    if (null_mask_owner() != nullptr) { total += null_mask_owner()->buffer.size(); }
    return total;
  }

  [[nodiscard]] std::optional<cudf_handle_provenance> destroy_provenance() const noexcept override
  {
    return _provenance;
  }

  std::unique_ptr<cudf::column> release_native_column() override
  {
    auto column = std::make_unique<cudf::column>(
      _dtype,
      _size,
      std::move(data_owner().buffer),
      null_mask_owner() == nullptr ? rmm::device_buffer{} : std::move(null_mask_owner()->buffer),
      _null_count);
    _null_mask_owner.reset();
    _data_owner.reset();
    return column;
  }

 private:
  [[nodiscard]] cudf_device_buffer_owner& data_owner() const
  {
    return _data_owner == nullptr ? *_pending_data_owner : *_data_owner;
  }

  [[nodiscard]] cudf_device_buffer_owner* null_mask_owner() const
  {
    return _null_mask_owner == nullptr ? _pending_null_mask_owner : _null_mask_owner.get();
  }

  cudf_handle_provenance _provenance;
  cudf::data_type _dtype;
  cudf::size_type _size;
  cudf::size_type _null_count;
  cudf_device_buffer_owner* _pending_data_owner;
  cudf_device_buffer_owner* _pending_null_mask_owner;
  std::unique_ptr<cudf_device_buffer_owner> _data_owner;
  std::unique_ptr<cudf_device_buffer_owner> _null_mask_owner;
};

cudf_column_handle release_adopted_column_handle(
  std::unique_ptr<adopted_device_buffer_column_handle_state> state)
{
  auto* state_ptr                                           = state.get();
  std::unique_ptr<cudf_column_handle_state> published_state = std::move(state);
  auto handle = cudf_release_column_handle(std::move(published_state));
  state_ptr->commit_published_handles();
  return handle;
}
}  // namespace

extern "C" {
cudf_status_t cudf_device_buffer_create_ex(size_t size,
                                           cudf_exec_options_t const* options,
                                           cudf_device_buffer_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    rmm::device_buffer buffer(size, exec->stream, *exec->mr);
    *out = release_device_buffer_handle(std::move(buffer), cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_device_buffer_destroy(cudf_device_buffer_handle handle)
{
  if (handle == nullptr) { return CUDF_SUCCESS; }
  try {
    auto* owner           = reinterpret_cast<cudf_device_buffer_owner*>(handle);
    auto const provenance = owner->provenance;
    // The handle is consumed even when provenance entry or cleanup fails.
    return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

cudf_status_t cudf_device_buffer_data(cudf_device_buffer_handle handle, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  auto& buf = device_buffer_from_handle(handle);
  *out      = buf.data();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_device_buffer_size(cudf_device_buffer_handle handle, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  auto const& buf = device_buffer_from_handle(handle);
  *out            = buf.size();
  return CUDF_SUCCESS;
}

cudf_status_t cudf_device_buffer_copy_from_host_ex(cudf_device_buffer_handle handle,
                                                   void const* host_data,
                                                   size_t size,
                                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_data, "host_data is null");
  auto& owner = device_buffer_owner_from_handle(handle);
  auto& buf   = owner.buffer;
  if (size > buf.size()) {
    set_error("copy size exceeds buffer size");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    if (exec->device != owner.provenance.device) {
      set_error("exec options device must match device buffer provenance");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    CUDF_CUDA_TRY(
      cudaMemcpyAsync(buf.data(), host_data, size, cudaMemcpyHostToDevice, exec->stream.value()));
    exec->stream.synchronize();
    cudf_record_synchronized_stream(exec->stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_device_buffer_copy_from_host_range_async_ex(cudf_device_buffer_handle handle,
                                                               size_t dst_offset,
                                                               void const* host_data,
                                                               size_t size,
                                                               cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(host_data, "host_data is null");
  auto& owner = device_buffer_owner_from_handle(handle);
  auto& buf   = owner.buffer;
  if (dst_offset > buf.size() || size > buf.size() - dst_offset) {
    set_error("copy range exceeds buffer size");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }
    if (exec->device != owner.provenance.device) {
      set_error("exec options device must match device buffer provenance");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (size == 0) { return CUDF_SUCCESS; }

    CUDF_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(buf.data()) + dst_offset,
                                  host_data,
                                  size,
                                  cudaMemcpyHostToDevice,
                                  exec->stream.value()));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_pinned_host_alloc(size_t size, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (size == 0) {
    set_error("pinned host allocation size must be greater than zero");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    void* ptr = nullptr;
    CUDF_CUDA_TRY(cudaHostAlloc(&ptr, size, cudaHostAllocPortable));
    *out = ptr;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_pinned_host_free(void* ptr)
{
  if (ptr == nullptr) { return CUDF_SUCCESS; }
  CUDF_RUST_TRY
  {
    CUDF_CUDA_TRY(cudaFreeHost(ptr));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_adopt_device_buffer(cudf_type_id_t type_id,
                                              cudf_device_buffer_handle buffer,
                                              int32_t row_count,
                                              cudf_column_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (row_count < 0) {
    set_error("row_count must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(buffer, "buffer is null");
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("only fixed-width types are supported for buffer adoption");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* owner     = reinterpret_cast<cudf_device_buffer_owner*>(buffer);
    auto& buf       = owner->buffer;
    auto elem_size  = cudf::size_of(dtype);
    size_t expected = static_cast<size_t>(row_count) * elem_size;
    if (buf.size() != expected) {
      set_error("buffer size does not match row_count * element_size");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto state = std::make_unique<adopted_device_buffer_column_handle_state>(
      dtype, static_cast<cudf::size_type>(row_count), 0, owner, nullptr, owner->provenance);
    *out = release_adopted_column_handle(std::move(state));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_column_adopt_device_buffer_nullable(cudf_type_id_t type_id,
                                                       cudf_device_buffer_handle data_buffer,
                                                       cudf_device_buffer_handle null_mask_buffer,
                                                       int32_t null_count,
                                                       int32_t row_count,
                                                       cudf_column_handle* out)
{
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
  if (data_buffer == null_mask_buffer) {
    set_error("data_buffer and null_mask_buffer must be distinct handles");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    cudf::data_type dtype(to_cudf_type_id(type_id));
    if (!cudf::is_fixed_width(dtype)) {
      set_error("only fixed-width types are supported for buffer adoption");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto* data_owner = reinterpret_cast<cudf_device_buffer_owner*>(data_buffer);
    auto* mask_owner = reinterpret_cast<cudf_device_buffer_owner*>(null_mask_buffer);
    auto& data_buf   = data_owner->buffer;
    auto& mask_buf   = mask_owner->buffer;

    auto elem_size       = cudf::size_of(dtype);
    size_t expected_data = static_cast<size_t>(row_count) * elem_size;
    if (data_buf.size() != expected_data) {
      set_error("data buffer size does not match row_count * element_size");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    size_t min_mask = cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(row_count));
    if (mask_buf.size() < min_mask) {
      set_error("null mask buffer is too small");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    if (expected_data > 0 && min_mask > 0) {
      int32_t data_device = -1;
      cudf_status_t const data_device_status =
        cudf_cuda_pointer_get_device(data_buf.data(), &data_device);
      if (data_device_status != CUDF_SUCCESS) { return data_device_status; }

      int32_t mask_device = -1;
      cudf_status_t const mask_device_status =
        cudf_cuda_pointer_get_device(mask_buf.data(), &mask_device);
      if (mask_device_status != CUDF_SUCCESS) { return mask_device_status; }

      if (data_device != mask_device) {
        set_error(
          "data buffer and null mask buffer must belong to the same "
          "CUDA device");
        return CUDF_ERROR_INVALID_OPERATION;
      }
    }

    auto state = std::make_unique<adopted_device_buffer_column_handle_state>(
      dtype,
      static_cast<cudf::size_type>(row_count),
      static_cast<cudf::size_type>(null_count),
      data_owner,
      mask_owner,
      data_owner->provenance);
    *out = release_adopted_column_handle(std::move(state));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
