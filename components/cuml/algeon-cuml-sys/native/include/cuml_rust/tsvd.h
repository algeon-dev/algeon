#ifndef CUML_RUST_TSVD_H
#define CUML_RUST_TSVD_H

#include "types.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

cuml_status_t cuml_tsvd_fit(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
                            int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
                            float* output_components, float* singular_values);
cuml_status_t cuml_tsvd_fit_transform(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
                                      int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
                                      float* transformed, float* output_components,
                                      float* explained_variance, float* explained_variance_ratio,
                                      float* singular_values);
cuml_status_t cuml_tsvd_transform(cuml_handle_t handle, const float* input, int64_t rows, int64_t columns,
                                  int64_t components, const float* model_components, float* transformed);
cuml_status_t cuml_tsvd_inverse_transform(cuml_handle_t handle, const float* transformed, int64_t rows,
                                          int64_t columns, int64_t components,
                                          const float* model_components, float* output);

#ifdef __cplusplus
}
#endif
#endif
