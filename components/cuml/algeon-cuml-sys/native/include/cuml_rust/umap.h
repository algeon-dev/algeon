#ifndef CUML_RUST_UMAP_H
#define CUML_RUST_UMAP_H

#include "cuml_rust/handle.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuml_umap_params_t {
  int32_t n_neighbors;
  int32_t n_components;
  int32_t n_epochs;
  float learning_rate;
  float min_dist;
  float spread;
  float set_op_mix_ratio;
  float local_connectivity;
  float repulsion_strength;
  int32_t negative_sample_rate;
  float transform_queue_size;
  int32_t init;
  uint64_t random_state;
} cuml_umap_params_t;

cuml_status_t cuml_umap_fit_params(cuml_handle_t handle,
                                   const float* input,
                                   int32_t rows,
                                   int32_t columns,
                                   const cuml_umap_params_t* params,
                                   float* embedding);

cuml_status_t cuml_umap_transform_params(cuml_handle_t handle,
                                          const float* input,
                                          int32_t rows,
                                          int32_t columns,
                                          const float* training_input,
                                          int32_t training_rows,
                                          const float* embedding,
                                          int32_t embedding_rows,
                                          const cuml_umap_params_t* params,
                                          float* output);

#ifdef __cplusplus
}
#endif
#endif
