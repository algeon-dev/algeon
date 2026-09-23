#include "cuml_rust/pca.h"
#include "shim_common.hpp"

#include <cuml/decomposition/pca.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;
raft::handle_t& cuml_handle_raft(cuml_handle_t handle);

namespace {
void validate(cuml_handle_t handle, const void* const* pointers, size_t count,
              int64_t rows, int64_t columns, int64_t components)
{
  if (handle == nullptr) throw std::invalid_argument("handle is null");
  for (size_t i = 0; i < count; ++i) {
    if (pointers[i] == nullptr) throw std::invalid_argument("required buffer is null");
  }
  if (rows < 2 || columns < 2 || components < 1 || components > columns) {
    throw std::invalid_argument("PCA dimensions are out of range");
  }
  if (static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(columns) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(components) > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("PCA dimensions overflow size_t");
  }
}
ML::paramsPCA params(int64_t rows, int64_t columns, int64_t components,
                     int32_t algorithm, float tolerance, uint32_t iterations,
                     int32_t copy, int32_t whiten)
{
  if (algorithm < 0 || algorithm > 1 || !std::isfinite(tolerance) || tolerance < 0 || iterations == 0) {
    throw std::invalid_argument("PCA solver parameters are invalid");
  }
  ML::paramsPCA p;
  p.n_rows = static_cast<size_t>(rows);
  p.n_cols = static_cast<size_t>(columns);
  p.n_components = static_cast<size_t>(components);
  p.algorithm = static_cast<ML::solver>(algorithm);
  p.tol = tolerance;
  p.n_iterations = iterations;
  p.copy = copy != 0;
  p.whiten = whiten != 0;
  return p;
}
}

extern "C" cuml_status_t cuml_pca_fit(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
    int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
    int32_t copy, int32_t whiten, float* output_components, float* explained_variance,
    float* explained_variance_ratio, float* singular_values, float* mean, float* noise_variance)
{
  const void* buffers[]{input, output_components, explained_variance, explained_variance_ratio,
                        singular_values, mean, noise_variance};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 7, rows, columns, components);
    auto p = params(rows, columns, components, algorithm, tolerance, iterations, copy, whiten);
    ML::pcaFit(cuml_handle_raft(handle), input, output_components, explained_variance,
               explained_variance_ratio, singular_values, mean, noise_variance, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_pca_fit_transform(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
    int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
    int32_t copy, int32_t whiten, float* transformed, float* output_components,
    float* explained_variance, float* explained_variance_ratio, float* singular_values,
    float* mean, float* noise_variance)
{
  const void* buffers[]{input, transformed, output_components, explained_variance,
                        explained_variance_ratio, singular_values, mean, noise_variance};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 8, rows, columns, components);
    auto p = params(rows, columns, components, algorithm, tolerance, iterations, copy, whiten);
    ML::pcaFitTransform(cuml_handle_raft(handle), input, transformed, output_components,
                        explained_variance, explained_variance_ratio, singular_values, mean,
                        noise_variance, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_pca_transform(cuml_handle_t handle, const float* input, int64_t rows,
    int64_t columns, int64_t components, int32_t whiten, const float* model_components,
    const float* singular_values, const float* mean, float* transformed)
{
  const void* buffers[]{input, model_components, singular_values, mean, transformed};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 5, rows, columns, components);
    ML::paramsPCA p; p.n_rows = rows; p.n_cols = columns; p.n_components = components; p.whiten = whiten != 0;
    auto& raft_handle = cuml_handle_raft(handle);
    auto const row_count = static_cast<size_t>(rows);
    auto const column_count = static_cast<size_t>(columns);
    if (row_count > std::numeric_limits<size_t>::max() / column_count / sizeof(float)) {
      throw std::overflow_error("PCA input size overflows size_t");
    }
    auto const elements = row_count * column_count;
    rmm::device_uvector<float> scratch(elements, raft_handle.get_stream());
    auto const status = cudaMemcpyAsync(scratch.data(), input, elements * sizeof(float),
                                        cudaMemcpyDeviceToDevice, raft_handle.get_stream());
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    // Pinned RAFT PCA centers its transform input in place and restores it with float arithmetic.
    ML::pcaTransform(raft_handle, scratch.data(),
                     const_cast<float*>(model_components), transformed,
                     const_cast<float*>(singular_values), const_cast<float*>(mean), p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_pca_inverse_transform(cuml_handle_t handle, const float* transformed,
    int64_t rows, int64_t columns, int64_t components, int32_t whiten,
    const float* model_components, const float* singular_values, const float* mean, float* output)
{
  const void* buffers[]{transformed, model_components, singular_values, mean, output};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "PCA input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 5, rows, columns, components);
    ML::paramsPCA p; p.n_rows = rows; p.n_cols = columns; p.n_components = components; p.whiten = whiten != 0;
    ML::pcaInverseTransform(cuml_handle_raft(handle), const_cast<float*>(transformed),
                            const_cast<float*>(model_components), const_cast<float*>(singular_values),
                            const_cast<float*>(mean), output, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
