#include "sorting_internal.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/execution>
#include <cuda/std/limits>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace cudf_rust::sorting {
namespace {

template <typename T>
using rank_key_t = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;

// Unsigned key whose integer order is the float order: non-negative values get
// the sign bit set, negative values have every bit inverted. Every NaN maps to
// the largest key, so NaNs tie with each other above +inf, as in cuDF's sort.
// No finite or infinite value reaches the largest or the smallest key (both
// are NaN bit patterns), so a null row mapped to the key the selection reaches
// last cannot displace a non-null row.
template <typename T>
struct float_rank_key {
  using key_type = rank_key_t<T>;
  static constexpr key_type sign = key_type{1} << (sizeof(key_type) * 8 - 1);

  cudf::column_device_view input;
  key_type null_key;

  __device__ key_type operator()(cudf::size_type row) const {
    if (input.is_null(row)) {
      return null_key;
    }
    T const value = input.element<T>(row);
    if (cuda::std::isnan(value)) {
      return cuda::std::numeric_limits<key_type>::max();
    }
    auto const bits = cuda::std::bit_cast<key_type>(value);
    return (bits & sign) != 0 ? key_type(~bits) : key_type(bits | sign);
  }
};

template <typename T>
T float_from_rank_key(rank_key_t<T> key) {
  using key_type = rank_key_t<T>;
  if (key == cuda::std::numeric_limits<key_type>::max()) {
    return cuda::std::numeric_limits<T>::quiet_NaN();
  }
  auto const sign = float_rank_key<T>::sign;
  auto const bits = (key & sign) != 0 ? key_type(key ^ sign) : key_type(~key);
  return cuda::std::bit_cast<T>(bits);
}

template <typename T>
std::unique_ptr<cudf::scalar>
boundary(cudf::column_view const &input, cudf::size_type k, cudf::order order,
         cudf_exec_context const &exec) {
  using key_type = rank_key_t<T>;
  auto const stream = exec.stream;
  bool const descending = order == cudf::order::DESCENDING;
  auto const device_input =
      cudf::column_device_view::create_with_mr(input, stream, *exec.mr);
  auto const keys = cuda::make_transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      float_rank_key<T>{*device_input,
                        descending ? cuda::std::numeric_limits<key_type>::min()
                                   : cuda::std::numeric_limits<key_type>::max()});
  auto selected = rmm::device_uvector<key_type>(k, stream, *exec.mr);

  // The k selected keys come back unordered; nothing needs them sorted.
  auto const requirements =
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::output_ordering::unsorted);
  auto const env = cuda::std::execution::env{cuda::stream_ref{stream.value()},
                                             requirements};
  auto const select = [&](void *storage, std::size_t &bytes) {
    return descending ? cub::DeviceTopK::MaxKeys(storage, bytes, keys,
                                                 selected.begin(), input.size(),
                                                 k, env)
                      : cub::DeviceTopK::MinKeys(storage, bytes, keys,
                                                 selected.begin(), input.size(),
                                                 k, env);
  };
  std::size_t select_bytes = 0;
  CUDF_CUDA_TRY(select(nullptr, select_bytes));
  {
    auto select_storage = rmm::device_buffer(select_bytes, stream, *exec.mr);
    CUDF_CUDA_TRY(select(select_storage.data(), select_bytes));
  }

  // The boundary is the worst of the k best keys.
  auto worst = rmm::device_scalar<key_type>(stream, *exec.mr);
  auto const reduce = [&](void *storage, std::size_t &bytes) {
    return descending
               ? cub::DeviceReduce::Min(storage, bytes, selected.begin(),
                                        worst.data(), k, stream.value())
               : cub::DeviceReduce::Max(storage, bytes, selected.begin(),
                                        worst.data(), k, stream.value());
  };
  std::size_t reduce_bytes = 0;
  CUDF_CUDA_TRY(reduce(nullptr, reduce_bytes));
  {
    auto reduce_storage = rmm::device_buffer(reduce_bytes, stream, *exec.mr);
    CUDF_CUDA_TRY(reduce(reduce_storage.data(), reduce_bytes));
  }
  return std::make_unique<cudf::numeric_scalar<T>>(
      float_from_rank_key<T>(worst.value(stream)), true, stream, *exec.mr);
}

} // namespace

std::unique_ptr<cudf::scalar>
float_top_k_boundary(cudf::column_view const &input, cudf::size_type k,
                     cudf::order order, cudf_exec_context const &exec) {
  CUDF_EXPECTS(k > 0 && k <= input.size() - input.null_count(),
               "k must select among the non-null rows", std::invalid_argument);
  switch (input.type().id()) {
  case cudf::type_id::FLOAT32:
    return boundary<float>(input, k, order, exec);
  case cudf::type_id::FLOAT64:
    return boundary<double>(input, k, order, exec);
  default:
    CUDF_FAIL("float_top_k_boundary requires a FLOAT32 or FLOAT64 column",
              cudf::data_type_error);
  }
}

} // namespace cudf_rust::sorting
