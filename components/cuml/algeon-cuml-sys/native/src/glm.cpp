#include "cuml_rust/glm.h"
#include "shim_common.hpp"

#include <cuml/linear_model/glm.hpp>
#include <cuml/linear_model/qn.h>
#include <cuda_runtime_api.h>
#include <raft/core/handle.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

cuml_status_t cuml_record_error(cuml_status_t status, char const* message) noexcept;
raft::handle_t& cuml_handle_raft(cuml_handle_t handle);

namespace {
ML::GLM::qn_params make_params(int32_t loss, double penalty_l1, double penalty_l2,
                               int32_t max_iter, int32_t linesearch_max_iter,
                               int32_t lbfgs_memory, int32_t fit_intercept,
                               int32_t penalty_normalized, double grad_tol,
                               double change_tol)
{
  if ((loss != ML::GLM::QN_LOSS_LOGISTIC && loss != ML::GLM::QN_LOSS_SOFTMAX) ||
      !std::isfinite(penalty_l1) || penalty_l1 < 0 ||
      !std::isfinite(penalty_l2) || penalty_l2 < 0 ||
      max_iter <= 0 || linesearch_max_iter <= 0 || lbfgs_memory <= 0 ||
      !std::isfinite(grad_tol) || grad_tol < 0 ||
      !std::isfinite(change_tol) || change_tol < 0) {
    throw std::invalid_argument("QN logistic regression parameters are invalid");
  }
  ML::GLM::qn_params p;
  p.loss = static_cast<ML::GLM::qn_loss_type>(loss);
  p.penalty_l1 = penalty_l1;
  p.penalty_l2 = penalty_l2;
  p.max_iter = max_iter;
  p.linesearch_max_iter = linesearch_max_iter;
  p.lbfgs_memory = lbfgs_memory;
  p.fit_intercept = fit_intercept != 0;
  p.penalty_normalized = penalty_normalized != 0;
  p.grad_tol = grad_tol;
  p.change_tol = change_tol;
  p.verbose = static_cast<int>(cuml_native_log_level);
  return p;
}
void validate(cuml_handle_t handle, const void* const* pointers, size_t count,
              int32_t min_rows, int32_t rows, int32_t columns, int32_t classes)
{
  if (handle == nullptr) throw std::invalid_argument("handle is null");
  for (size_t i = 0; i < count; ++i) {
    if (pointers[i] == nullptr) throw std::invalid_argument("required buffer is null");
  }
  if (rows < min_rows || columns < 1 || classes < 1) {
    throw std::invalid_argument("QN dimensions are out of range");
  }
}
}

extern "C" cuml_status_t cuml_glm_qn_fit(
    cuml_handle_t handle, float* input, int32_t rows, int32_t columns,
    int32_t classes, int32_t column_major, float* labels, float* coefficients,
    int32_t loss, double penalty_l1, double penalty_l2, int32_t max_iter,
    int32_t linesearch_max_iter, int32_t lbfgs_memory, int32_t fit_intercept,
    int32_t penalty_normalized, double grad_tol, double change_tol,
    double* objective, int32_t* iterations)
{
  const void* buffers[]{input, labels, coefficients, objective, iterations};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "QN handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "QN input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 5, 2, rows, columns, classes);
    if (column_major != 0 && column_major != 1) throw std::invalid_argument("matrix layout is invalid");
    auto p = make_params(loss, penalty_l1, penalty_l2, max_iter,
                         linesearch_max_iter, lbfgs_memory, fit_intercept,
                         penalty_normalized, grad_tol, change_tol);
    auto& raft_handle = cuml_handle_raft(handle);
    auto const width = static_cast<size_t>(columns) + (p.fit_intercept ? 1U : 0U);
    auto const targets = loss == ML::GLM::QN_LOSS_LOGISTIC ? 1U : static_cast<size_t>(classes);
    auto const count = width * targets;
    auto const memset_status = cudaMemsetAsync(coefficients, 0, count * sizeof(float), raft_handle.get_stream());
    if (memset_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(memset_status));
    float f = 0;
    int num_iters = 0;
    ML::GLM::qnFit<float, int>(raft_handle, p, input, column_major != 0, labels,
                               rows, columns, classes, coefficients, &f, &num_iters);
    *objective = f;
    *iterations = num_iters;
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}

// Writes the linear scores in sample-major order: `rows` entries for binary
// logistic loss (one coefficient vector), `rows * classes` for softmax.
extern "C" cuml_status_t cuml_glm_qn_decision_function(
    cuml_handle_t handle, const float* input, int32_t rows, int32_t columns,
    int32_t classes, int32_t column_major, const float* coefficients,
    int32_t loss, int32_t fit_intercept, float* scores)
{
  const void* buffers[]{input, coefficients, scores};
  cuml_clear_error();
  if (handle == nullptr) return cuml_record_error(CUML_ERROR_NULL, "QN handle is null");
  for (auto buffer : buffers) if (buffer == nullptr) return cuml_record_error(CUML_ERROR_NULL, "QN input or output buffer is null");
  CUML_RUST_TRY {
    validate(handle, buffers, 3, 1, rows, columns, classes);
    if (column_major != 0 && column_major != 1) throw std::invalid_argument("matrix layout is invalid");
    auto p = make_params(loss, 0, 0, 1, 1, 1, fit_intercept, 0, 0, 0);
    ML::GLM::qnDecisionFunction<float, int>(cuml_handle_raft(handle), p,
                                            const_cast<float*>(input), column_major != 0,
                                            rows, columns, classes,
                                            const_cast<float*>(coefficients), scores);
    return CUML_SUCCESS;
  } CUML_RUST_CATCH
}
