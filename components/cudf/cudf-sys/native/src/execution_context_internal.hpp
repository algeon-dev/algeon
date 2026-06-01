#ifndef CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
#define CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP

#include "execution_context.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace rmm::mr {
class device_memory_resource;
}

struct cudf_exec_context {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;
  rmm::mr::device_memory_resource* mr_raw;
};

cudf_status_t cudf_resolve_exec_options(cudf_exec_options_t const* options,
                                        cudf_exec_context* out);
void cudf_record_synchronized_stream(rmm::cuda_stream_view stream);

#endif  // CUDF_RUST_EXECUTION_CONTEXT_INTERNAL_HPP
