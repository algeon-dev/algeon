#ifndef CUML_RUST_RANDOM_FOREST_H
#define CUML_RUST_RANDOM_FOREST_H

#include "cuml_rust/handle.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuml_rf_training_streams cuml_rf_training_streams;
typedef cuml_rf_training_streams* cuml_rf_training_streams_t;
typedef struct cuml_rf_forest cuml_rf_forest;
typedef cuml_rf_forest* cuml_rf_forest_t;

cuml_status_t cuml_rf_training_streams_create(cuml_handle_t handle,
                                               int32_t count,
                                               cuml_rf_training_streams_t* out_streams);
cuml_status_t cuml_rf_training_streams_raw(cuml_rf_training_streams_t streams,
                                            int32_t index,
                                            void** out_stream);
cuml_status_t cuml_rf_training_streams_destroy(cuml_rf_training_streams_t streams);

cuml_status_t cuml_rf_classifier_fit(cuml_handle_t handle,
                                     cuml_rf_training_streams_t streams,
                                     const float* input,
                                     int32_t rows,
                                     int32_t columns,
                                     const int32_t* labels,
                                     int32_t n_classes,
                                     int32_t n_trees,
                                     int32_t bootstrap,
                                     float max_samples,
                                     uint64_t seed,
                                     int32_t max_depth,
                                     int32_t max_leaves,
                                     float max_features,
                                     int32_t max_n_bins,
                                     int32_t min_samples_leaf,
                                     int32_t min_samples_split,
                                     float min_impurity_decrease,
                                     int32_t max_batch_size,
                                     int32_t split_criterion,
                                     int32_t input_row_major,
                                     cuml_rf_forest_t* out_forest);
cuml_status_t cuml_rf_regressor_fit(cuml_handle_t handle,
                                    cuml_rf_training_streams_t streams,
                                    const float* input,
                                    int32_t rows,
                                    int32_t columns,
                                    const float* labels,
                                    int32_t n_trees,
                                    int32_t bootstrap,
                                    float max_samples,
                                    uint64_t seed,
                                    int32_t max_depth,
                                    int32_t max_leaves,
                                    float max_features,
                                    int32_t max_n_bins,
                                    int32_t min_samples_leaf,
                                    int32_t min_samples_split,
                                    float min_impurity_decrease,
                                    int32_t max_batch_size,
                                    int32_t split_criterion,
                                    int32_t input_row_major,
                                    cuml_rf_forest_t* out_forest);
cuml_status_t cuml_rf_classifier_predict(cuml_handle_t handle,
                                         cuml_rf_forest_t forest,
                                         const float* input,
                                         int32_t rows,
                                         int32_t columns,
                                         int64_t* predictions);
cuml_status_t cuml_rf_regressor_predict(cuml_handle_t handle,
                                        cuml_rf_forest_t forest,
                                        const float* input,
                                        int32_t rows,
                                        int32_t columns,
                                        float* predictions);
cuml_status_t cuml_rf_forest_destroy(cuml_rf_forest_t forest);

#ifdef __cplusplus
}
#endif
#endif
