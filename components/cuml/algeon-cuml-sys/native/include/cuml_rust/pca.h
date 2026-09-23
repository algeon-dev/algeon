#ifndef CUML_RUST_PCA_H
#define CUML_RUST_PCA_H

#include "types.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

cuml_status_t cuml_pca_fit(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
                           int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
                           int32_t copy, int32_t whiten, float* output_components,
                           float* explained_variance, float* explained_variance_ratio,
                           float* singular_values, float* mean, float* noise_variance);
cuml_status_t cuml_pca_fit_transform(cuml_handle_t handle, float* input, int64_t rows, int64_t columns,
                                     int64_t components, int32_t algorithm, float tolerance, uint32_t iterations,
                                     int32_t copy, int32_t whiten, float* transformed,
                                     float* output_components, float* explained_variance,
                                     float* explained_variance_ratio, float* singular_values,
                                     float* mean, float* noise_variance);
/* PCA transform mutates its input temporarily; the shim copies it before calling cuML.
 * Inverse transform reads its input and both entry points only read model buffers. */
cuml_status_t cuml_pca_transform(cuml_handle_t handle, const float* input, int64_t rows, int64_t columns,
                                 int64_t components, int32_t whiten, const float* model_components,
                                 const float* singular_values, const float* mean, float* transformed);
cuml_status_t cuml_pca_inverse_transform(cuml_handle_t handle, const float* transformed, int64_t rows,
                                         int64_t columns, int64_t components, int32_t whiten,
                                         const float* model_components, const float* singular_values,
                                         const float* mean, float* output);

#ifdef __cplusplus
}
#endif
#endif
