#include "cuml_rust/tsvd.h"
#include "shim_common.hpp"

#include <cuml/decomposition/tsvd.hpp>
#include <raft/core/handle.hpp>

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
    throw std::invalid_argument("TruncatedSVD dimensions are out of range");
  }
  if (static_cast<uint64_t>(rows) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(columns) > std::numeric_limits<size_t>::max() ||
      static_cast<uint64_t>(components) > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error("TruncatedSVD dimensions overflow size_t");
  }
}
ML::paramsTSVD params(int64_t rows, int64_t columns, int64_t components,
                      int32_t algorithm, float tolerance, uint32_t iterations)
{
  if (algorithm < 0 || algorithm > 1 || !std::isfinite(tolerance) || tolerance < 0 || iterations == 0) {
    throw std::invalid_argument("TruncatedSVD solver parameters are invalid");
  }
  ML::paramsTSVD p;
  p.n_rows = static_cast<size_t>(rows);
  p.n_cols = static_cast<size_t>(columns);
  p.n_components = static_cast<size_t>(components);
  p.algorithm = static_cast<ML::solver>(algorithm);
  p.tol = tolerance;
  p.n_iterations = iterations;
  return p;
}
}

extern "C" cuml_status_t cuml_tsvd_fit(cuml_handle_t handle, float* input, int64_t rows,
    int64_t columns, int64_t components, int32_t algorithm, float tolerance,
    uint32_t iterations, float* output_components, float* singular_values)
{
  const void* buffers[]{input, output_components, singular_values};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 3, rows, columns, components);
    auto p = params(rows, columns, components, algorithm, tolerance, iterations);
    ML::tsvdFit(cuml_handle_raft(handle), input, output_components, singular_values, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_tsvd_fit_transform(cuml_handle_t handle, float* input, int64_t rows,
    int64_t columns, int64_t components, int32_t algorithm, float tolerance,
    uint32_t iterations, float* transformed, float* output_components,
    float* explained_variance, float* explained_variance_ratio, float* singular_values)
{
  const void* buffers[]{input, transformed, output_components, explained_variance,
                        explained_variance_ratio, singular_values};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 6, rows, columns, components);
    auto p = params(rows, columns, components, algorithm, tolerance, iterations);
    ML::tsvdFitTransform(cuml_handle_raft(handle), input, transformed, output_components,
                         explained_variance, explained_variance_ratio, singular_values, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_tsvd_transform(cuml_handle_t handle, const float* input,
    int64_t rows, int64_t columns, int64_t components, const float* model_components,
    float* transformed)
{
  const void* buffers[]{input, model_components, transformed};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 3, rows, columns, components);
    ML::paramsTSVD p; p.n_rows = rows; p.n_cols = columns; p.n_components = components;
    ML::tsvdTransform(cuml_handle_raft(handle), const_cast<float*>(input),
                      const_cast<float*>(model_components), transformed, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

extern "C" cuml_status_t cuml_tsvd_inverse_transform(cuml_handle_t handle,
    const float* transformed, int64_t rows, int64_t columns, int64_t components,
    const float* model_components, float* output)
{
  const void* buffers[]{transformed, model_components, output};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "TruncatedSVD input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 3, rows, columns, components);
    ML::paramsTSVD p; p.n_rows = rows; p.n_cols = columns; p.n_components = components;
    ML::tsvdInverseTransform(cuml_handle_raft(handle), const_cast<float*>(transformed),
                             const_cast<float*>(model_components), output, p);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
