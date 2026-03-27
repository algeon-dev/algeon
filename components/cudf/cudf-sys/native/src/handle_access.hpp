#ifndef CUDF_RUST_HANDLE_ACCESS_HPP
#define CUDF_RUST_HANDLE_ACCESS_HPP

#include "memory_resource.h"
#include "stream.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace rmm::mr {
class device_memory_resource;
}

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle);
rmm::device_async_resource_ref cudf_memory_resource_ref_from_handle(
    cudf_memory_resource_handle handle);
rmm::mr::device_memory_resource* cudf_memory_resource_raw_from_handle(
    cudf_memory_resource_handle handle);

#endif  // CUDF_RUST_HANDLE_ACCESS_HPP
