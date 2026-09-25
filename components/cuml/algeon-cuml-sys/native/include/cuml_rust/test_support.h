#ifndef CUML_RUST_TEST_SUPPORT_H
#define CUML_RUST_TEST_SUPPORT_H

#include "error.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cuml_test_exception_kind {
  CUML_TEST_ATTRIBUTED_OOM = 0,
  CUML_TEST_RMM_OOM = 1,
  CUML_TEST_RMM_BAD_ALLOC = 2,
  CUML_TEST_CUDA_ERROR = 3,
  CUML_TEST_RAFT_EXCEPTION = 4,
  CUML_TEST_STD_BAD_ALLOC = 5,
  CUML_TEST_OVERFLOW = 6,
  CUML_TEST_INVALID_ARGUMENT = 7,
  CUML_TEST_UNKNOWN = 8
} cuml_test_exception_kind_t;

cuml_status_t cuml_test_raise_error(cuml_status_t status, const char* message);
cuml_status_t cuml_test_raise_exception(cuml_test_exception_kind_t kind);
// Host Treelite GTIL oracle over a Treelite checkpoint with the default
// postprocessor, single-threaded. Writes rows x num_class values.
cuml_status_t cuml_test_forest_gtil_predict(const uint8_t* checkpoint, size_t len, const double* input,
                                            int32_t rows, int32_t columns, double* output,
                                            size_t output_capacity, size_t* out_written);

#ifdef __cplusplus
}
#endif
#endif
