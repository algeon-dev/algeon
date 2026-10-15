#include "edge_count.h"
#include "memory_resource_compat.hpp"

#include <cudf_rust/error_boundary.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/algorithm>
#include <cuda/std/tuple>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>

namespace {

using DeviceResource = cuda::mr::any_resource<cuda::mr::device_accessible>;

cugraph_error_code_t record_error(cugraph_rs_edge_count_error_t* error,
                                   cugraph_error_code_t code,
                                   char const* message)
{
  auto const length = std::min(std::strlen(message), sizeof(error->message) - 1);
  std::memcpy(error->message, message, length);
  error->message[length] = '\0';
  return code;
}

struct logical_count_summary {
  size_t count;
  size_t invalid_group;
  size_t multiplicity;
};

template <typename vertex_t>
cugraph_error_code_t count_logical_edges(
  rmm::cuda_stream_view stream,
  DeviceResource mr,
  vertex_t const* source_data,
  vertex_t const* destination_data,
  size_t size,
  cugraph_graph_properties_t properties,
  cugraph_rs_edge_count_result_t* result,
  cugraph_rs_edge_count_error_t* error)
{
  if (size == 0 || (properties.is_multigraph && !properties.is_symmetric)) {
    result->count = size;
    return CUGRAPH_SUCCESS;
  }
  rmm::device_uvector<vertex_t> sources(size, stream, mr);
  rmm::device_uvector<vertex_t> destinations(size, stream, mr);
  auto const normalize = properties.is_symmetric;
  auto edges = thrust::make_zip_iterator(cuda::std::make_tuple(sources.begin(), destinations.begin()));
  thrust::transform(
    rmm::exec_policy(stream, mr), thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(size), edges,
    [source_data, destination_data, normalize] __device__(size_t i) -> cuda::std::tuple<vertex_t, vertex_t> {
      auto source = source_data[i];
      auto destination = destination_data[i];
      return normalize && destination < source ? cuda::std::make_tuple(destination, source)
                                                : cuda::std::make_tuple(source, destination);
    });
  thrust::sort(rmm::exec_policy(stream, mr), edges, edges + size);
  auto const src = sources.data();
  auto const dst = destinations.data();
  auto group_start = [src, dst] __device__(size_t i) -> bool {
    return i == 0 || src[i] != src[i - 1] || dst[i] != dst[i - 1];
  };
  if (!properties.is_multigraph) {
    result->count = thrust::count_if(
      rmm::exec_policy(stream, mr), thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(size), group_start);
    return CUGRAPH_SUCCESS;
  }
  rmm::device_uvector<size_t> starts(size, stream, mr);
  auto const end = thrust::copy_if(
    rmm::exec_policy(stream, mr), thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(size), starts.begin(), group_start);
  auto const group_count = static_cast<size_t>(end - starts.begin());
  auto const start_data = starts.data();
  auto const valid = std::numeric_limits<size_t>::max();
  auto summary = thrust::transform_reduce(
    rmm::exec_policy(stream, mr), thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(group_count),
    [src, dst, start_data, group_count, size, valid] __device__(size_t group) -> logical_count_summary {
      auto const start = start_data[group];
      auto const count = (group + 1 < group_count ? start_data[group + 1] : size) - start;
      auto const self_loop = src[start] == dst[start];
      auto const invalid = !self_loop && count % 2 != 0;
      return logical_count_summary{self_loop ? count : count / 2,
                                   invalid ? start : valid, invalid ? count : size_t{0}};
    },
    logical_count_summary{0, valid, 0},
    [] __device__(logical_count_summary lhs, logical_count_summary rhs) -> logical_count_summary {
      return logical_count_summary{lhs.count + rhs.count,
                                   cuda::std::min(lhs.invalid_group, rhs.invalid_group),
                                   lhs.invalid_group < rhs.invalid_group ? lhs.multiplicity : rhs.multiplicity};
    });
  if (summary.invalid_group != valid) {
    vertex_t source;
    vertex_t destination;
    RMM_CUDA_TRY(cudaMemcpyAsync(&source, sources.data() + summary.invalid_group, sizeof(vertex_t),
                                  cudaMemcpyDeviceToHost, stream.value()));
    RMM_CUDA_TRY(cudaMemcpyAsync(&destination, destinations.data() + summary.invalid_group, sizeof(vertex_t),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    result->source = source;
    result->destination = destination;
    result->multiplicity = summary.multiplicity;
    return record_error(error, CUGRAPH_INVALID_STATE,
                        "undirected multigraph decompressed edgelist contained an odd count for a non-self-loop edge");
  }
  result->count = summary.count;
  return CUGRAPH_SUCCESS;
}


}  // namespace

extern "C" cugraph_error_code_t cugraph_rs_count_logical_edges(
  void* cuda_stream,
  void* device_memory_resource,
  cugraph_type_erased_device_array_view_t const* sources,
  cugraph_type_erased_device_array_view_t const* destinations,
  cugraph_graph_properties_t properties,
  cugraph_rs_edge_count_result_t* result,
  cugraph_rs_edge_count_error_t* error)
{
  if (error == nullptr) { return CUGRAPH_INVALID_INPUT; }
  *error = {};
  error->allocation_source = CUGRAPH_ALLOCATION_SOURCE_UNKNOWN;
  error->failure_source = CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN;
  error->cuda_error = -1;
  try {
    if (device_memory_resource == nullptr || sources == nullptr || destinations == nullptr || result == nullptr) {
      return record_error(error, CUGRAPH_INVALID_INPUT, "edge counting requires explicit storage and memory resource");
    }
    *result = {};
    auto const size = cugraph_type_erased_device_array_view_size(sources);
    if (size != cugraph_type_erased_device_array_view_size(destinations)) {
      return record_error(error, CUGRAPH_INVALID_STATE, "decompressed edgelist returned mismatched source/destination lengths");
    }
    auto const dtype = cugraph_type_erased_device_array_view_type(sources);
    if (dtype != cugraph_type_erased_device_array_view_type(destinations)) {
      return record_error(error, CUGRAPH_INVALID_STATE, "decompressed edgelist source/destination vertex dtypes differ");
    }
    auto const stream = rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(cuda_stream)};
    auto mr = rmm::mr::any_device_resource_from_raw(
      reinterpret_cast<rmm::mr::device_memory_resource*>(device_memory_resource));
    auto const* source_data = cugraph_type_erased_device_array_view_pointer(sources);
    auto const* destination_data = cugraph_type_erased_device_array_view_pointer(destinations);
    if (dtype == INT32) {
      return count_logical_edges(stream, mr, static_cast<int32_t const*>(source_data),
                                 static_cast<int32_t const*>(destination_data), size, properties, result, error);
    }
    if (dtype == INT64) {
      return count_logical_edges(stream, mr, static_cast<int64_t const*>(source_data),
                                 static_cast<int64_t const*>(destination_data), size, properties, result, error);
    }
    return record_error(error, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                        "decompressed edgelist endpoints must have an integer vertex dtype");
  } catch (cudf::rust::memory::attributed_out_of_memory const& failure) {
    auto const identity = failure.identity();
    error->allocation_source = CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY;
    switch (failure.source()) {
      case CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL: error->failure_source = CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL; break;
      case CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT: error->failure_source = CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT; break;
      default: break;
    }
    error->domain_id = identity.primary_id;
    error->member_id = identity.secondary_id;
    error->generation = identity.generation;
    error->requested_bytes = failure.requested_bytes();
    error->alignment = failure.alignment();
    error->cuda_error = failure.cuda_error();
    return record_error(error, CUGRAPH_ALLOC_ERROR, failure.what());
  } catch (rmm::out_of_memory const& failure) {
    error->allocation_source = CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY;
    return record_error(error, CUGRAPH_ALLOC_ERROR, failure.what());
  } catch (rmm::bad_alloc const& failure) {
    error->allocation_source = CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC;
    return record_error(error, CUGRAPH_ALLOC_ERROR, failure.what());
  } catch (std::bad_alloc const& failure) {
    error->allocation_source = CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC;
    return record_error(error, CUGRAPH_ALLOC_ERROR, failure.what());
  } catch (std::exception const& failure) {
    return record_error(error, CUGRAPH_UNKNOWN_ERROR, failure.what());
  } catch (...) {
    return record_error(error, CUGRAPH_UNKNOWN_ERROR, "unknown logical edge count failure");
  }
}
