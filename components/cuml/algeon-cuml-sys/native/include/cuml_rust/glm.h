#ifndef CUML_RUST_GLM_H
#define CUML_RUST_GLM_H

#include "types.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

cuml_status_t cuml_glm_qn_fit(cuml_handle_t handle, float* input, int32_t rows, int32_t columns,
                              int32_t classes, int32_t column_major, float* labels,
                              float* coefficients, int32_t loss, double penalty_l1,
                              double penalty_l2, int32_t max_iter, int32_t linesearch_max_iter,
                              int32_t lbfgs_memory, int32_t fit_intercept,
                              int32_t penalty_normalized, double grad_tol,
                              double change_tol, double* objective, int32_t* iterations);
cuml_status_t cuml_glm_qn_decision_function(cuml_handle_t handle, const float* input,
                                            int32_t rows, int32_t columns, int32_t classes,
                                            int32_t column_major, const float* coefficients,
                                            int32_t loss, int32_t fit_intercept, float* scores);

#ifdef __cplusplus
}
#endif
#endif
