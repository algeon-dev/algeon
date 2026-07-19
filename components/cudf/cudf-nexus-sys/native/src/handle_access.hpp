#ifndef CUDF_RUST_HANDLE_ACCESS_HPP
#define CUDF_RUST_HANDLE_ACCESS_HPP

#include "memory_resource.h"
#include "memory_resource_compat.hpp"
#include "stream.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle);

int32_t cudf_stream_device_from_handle(cudf_stream_handle handle);

std::shared_ptr<void>
cudf_stream_lifetime_from_handle(cudf_stream_handle handle);

rmm::mr::device_memory_resource *
cudf_memory_resource_raw_from_handle(cudf_memory_resource_handle handle);

std::shared_ptr<void>
cudf_memory_resource_lifetime_from_handle(cudf_memory_resource_handle handle);

std::shared_ptr<void>
cudf_memory_resource_lifetime_from_raw(rmm::mr::device_memory_resource *raw_mr);

#endif // CUDF_RUST_HANDLE_ACCESS_HPP
