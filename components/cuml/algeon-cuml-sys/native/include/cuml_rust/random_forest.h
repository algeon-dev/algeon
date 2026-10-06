#ifndef CUML_RUST_RANDOM_FOREST_H
#define CUML_RUST_RANDOM_FOREST_H

#include "cuml_rust/forest_inference.h"
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
// Imports the fitted forest into nvForest on the handle's device and stream
// without a checkpoint round trip. Score output of a classifier is one
// probability per class, averaged across trees; class-index output resolves
// ties to the lowest index. A regressor has one score and no class index. The
// intermediate Treelite model is freed before returning.
cuml_status_t cuml_rf_forest_import(cuml_handle_t handle,
                                    cuml_rf_forest_t forest,
                                    cuml_forest_output_t output,
                                    cuml_forest_model_t* out_model,
                                    cuml_forest_metadata_t* out_metadata);
cuml_status_t cuml_rf_forest_destroy(cuml_rf_forest_t forest);

#ifdef __cplusplus
}
#endif
#endif
