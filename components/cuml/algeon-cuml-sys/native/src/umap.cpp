#include "cuml_rust/umap.h"
#include "cuml_rust/error.h"
#include "cuml_rust/types.h"
#include "shim_common.hpp"

#include <cuml/manifold/umap.hpp>
#include <raft/core/host_coo_matrix.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>

namespace raft {
class handle_t;
}

raft::handle_t* cuml_internal_raft_handle(cuml_handle_t handle) noexcept;
cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;

namespace {

void validate_shape(int32_t rows, int32_t columns, int32_t n_components)
{
  if (rows < 1 || columns < 1 || n_components < 1) {
    throw std::invalid_argument("UMAP dimensions must be positive");
  }
}

ML::UMAPParams make_params(cuml_umap_params_t const& source)
{
  if (source.n_neighbors < 1 || source.n_components < 1 || source.n_epochs < 0 ||
      source.negative_sample_rate < 1 || source.init < 0 || source.init > 1 ||
      !std::isfinite(source.learning_rate) || source.learning_rate <= 0.0f ||
      !std::isfinite(source.min_dist) || source.min_dist < 0.0f ||
      !std::isfinite(source.spread) || source.spread <= 0.0f ||
      !std::isfinite(source.set_op_mix_ratio) || source.set_op_mix_ratio < 0.0f ||
      source.set_op_mix_ratio > 1.0f || !std::isfinite(source.local_connectivity) ||
      source.local_connectivity < 0.0f || !std::isfinite(source.repulsion_strength) ||
      source.repulsion_strength < 0.0f || !std::isfinite(source.transform_queue_size) ||
      source.transform_queue_size <= 0.0f) {
    throw std::invalid_argument("invalid UMAP parameters");
  }
  ML::UMAPParams params;
  params.n_neighbors = source.n_neighbors;
  params.n_components = source.n_components;
  params.n_epochs = source.n_epochs;
  params.learning_rate = source.learning_rate;
  params.initial_alpha = source.learning_rate;
  params.min_dist = source.min_dist;
  params.spread = source.spread;
  params.set_op_mix_ratio = source.set_op_mix_ratio;
  params.local_connectivity = source.local_connectivity;
  params.repulsion_strength = source.repulsion_strength;
  params.negative_sample_rate = source.negative_sample_rate;
  params.transform_queue_size = source.transform_queue_size;
  params.init = source.init;
  params.random_state = source.random_state;
  params.deterministic = true;
  params.metric = ML::distance::DistanceType::L2SqrtExpanded;
  params.build_algo = ML::UMAPParams::BRUTE_FORCE_KNN;
  params.a = -1.0f;
  params.b = -1.0f;
  params.callback = nullptr;
  return params;
}

std::size_t matrix_bytes(int32_t rows, int32_t columns)
{
  auto const elements = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("UMAP matrix size overflows size_t");
  }
  return elements * sizeof(float);
}

}  // namespace

extern "C" cuml_status_t cuml_umap_fit_params(cuml_handle_t handle,
                                              const float* input,
                                              int32_t rows,
                                              int32_t columns,
                                              const cuml_umap_params_t* source,
                                              float* embedding)
{
  cuml_clear_error();
  if (handle == nullptr || input == nullptr || source == nullptr || embedding == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "UMAP fit received a null argument");
  }
  CUML_RUST_TRY {
    validate_shape(rows, columns, source->n_components);
    if (rows < 2 || source->n_neighbors >= rows) {
      throw std::invalid_argument("UMAP requires at least two rows and n_neighbors less than rows");
    }
    auto params = make_params(*source);
    std::unique_ptr<rmm::device_buffer> native_embedding;
    auto* raft_handle = cuml_internal_raft_handle(handle);
    if (raft_handle == nullptr) throw std::logic_error("cuML handle has no RAFT handle");
    ML::UMAP::find_ab(*raft_handle, &params);
    raft::host_coo_matrix<float, int, int, uint64_t> graph{*raft_handle, rows, rows, 0};
    ML::UMAP::fit(*raft_handle,
                  const_cast<float*>(input),
                  nullptr,
                  rows,
                  columns,
                  nullptr,
                  nullptr,
                  &params,
                  native_embedding,
                  graph);
    auto const bytes = matrix_bytes(rows, source->n_components);
    auto const status = cudaMemcpyAsync(embedding,
                                        native_embedding->data(),
                                        bytes,
                                        cudaMemcpyDeviceToDevice,
                                        raft_handle->get_stream());
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    raft_handle->sync_stream();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_umap_transform_params(cuml_handle_t handle,
                                                     const float* input,
                                                     int32_t rows,
                                                     int32_t columns,
                                                     const float* training_input,
                                                     int32_t training_rows,
                                                     const float* embedding,
                                                     int32_t embedding_rows,
                                                     const cuml_umap_params_t* source,
                                                     float* output)
{
  cuml_clear_error();
  if (handle == nullptr || input == nullptr || training_input == nullptr || embedding == nullptr ||
      source == nullptr || output == nullptr) {
    return cuml_record_error(CUML_ERROR_NULL, "UMAP transform received a null argument");
  }
  CUML_RUST_TRY {
    validate_shape(rows, columns, source->n_components);
    if (training_rows < 2 || embedding_rows != training_rows || source->n_neighbors >= training_rows) {
      throw std::invalid_argument("invalid fitted UMAP state or n_neighbors");
    }
    auto params = make_params(*source);
    auto* raft_handle = cuml_internal_raft_handle(handle);
    if (raft_handle == nullptr) throw std::logic_error("cuML handle has no RAFT handle");
    ML::UMAP::find_ab(*raft_handle, &params);
    auto const zero_status = cudaMemsetAsync(output, 0, matrix_bytes(rows, source->n_components), raft_handle->get_stream());
    if (zero_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(zero_status));
    ML::UMAP::transform(*raft_handle,
                        const_cast<float*>(input),
                        rows,
                        columns,
                        const_cast<float*>(training_input),
                        training_rows,
                        const_cast<float*>(embedding),
                        embedding_rows,
                        &params,
                        output);
    raft_handle->sync_stream();
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
