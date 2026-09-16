#ifndef CUML_RUST_FOREST_INFERENCE_H
#define CUML_RUST_FOREST_INFERENCE_H

#include "cuml_rust/handle.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUML_FOREST_FORMAT_XGBOOST_JSON = 0,
  CUML_FOREST_FORMAT_XGBOOST_UBJSON = 1,
  CUML_FOREST_FORMAT_XGBOOST_LEGACY = 2,
  CUML_FOREST_FORMAT_LIGHTGBM = 3,
  CUML_FOREST_FORMAT_TREELITE_CHECKPOINT = 4
} cuml_forest_format_t;

typedef enum {
  CUML_FOREST_PRECISION_FLOAT32 = 0,
  CUML_FOREST_PRECISION_FLOAT64 = 1
} cuml_forest_precision_t;

typedef enum {
  CUML_FOREST_OUTPUT_SCORES = 0,
  CUML_FOREST_OUTPUT_CLASS_INDEX = 1
} cuml_forest_output_t;

// num_outputs is nvForest's per-row output stride. A class-index model writes
// its index into the first slot of each row; the other slots are unspecified.
typedef struct {
  int32_t num_features;
  int32_t num_trees;
  int32_t num_outputs;
  cuml_forest_precision_t precision;
} cuml_forest_metadata_t;

typedef struct cuml_forest_model cuml_forest_model;
typedef cuml_forest_model* cuml_forest_model_t;

// Host only. Parses the source model, proves nvForest accepts it with a host
// import, and returns a Treelite checkpoint owned by the shim. Free it with
// cuml_forest_bytes_free.
cuml_status_t cuml_forest_inspect(const uint8_t* bytes, size_t len, cuml_forest_format_t format,
                                  cuml_forest_metadata_t* out_metadata,
                                  uint8_t** out_checkpoint, size_t* out_checkpoint_len);
cuml_status_t cuml_forest_bytes_free(uint8_t* bytes);

// Imports a checkpoint at its native precision onto the handle's device and
// stream through the current RMM resource.
cuml_status_t cuml_forest_import(cuml_handle_t handle, const uint8_t* checkpoint, size_t len,
                                 cuml_forest_output_t output, cuml_forest_model_t* out_model,
                                 cuml_forest_metadata_t* out_metadata);
// input is row-major rows x columns and output holds rows x num_outputs
// elements, both of the model precision on the handle's device. Work is
// enqueued on the handle stream without synchronization.
cuml_status_t cuml_forest_predict(cuml_handle_t handle, cuml_forest_model_t model,
                                  const void* input, int32_t rows, int32_t columns, void* output);
cuml_status_t cuml_forest_close(cuml_forest_model_t model);

// Reports the nvForest version compiled into libcuml_rust.so from NVForest_VERSION_* headers.
cuml_status_t cuml_forest_inference_version_get(int32_t* major, int32_t* minor, int32_t* patch);

#ifdef __cplusplus
}
#endif
#endif
