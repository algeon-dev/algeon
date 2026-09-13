#ifndef CUML_RUST_ERROR_H
#define CUML_RUST_ERROR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUML_SUCCESS = 0,
  CUML_ERROR_NULL = 1,
  CUML_ERROR_INVALID_ARGUMENT = 2,
  CUML_ERROR_INVALID_STATE = 3,
  CUML_ERROR_OUT_OF_MEMORY = 4,
  CUML_ERROR_HOST_OUT_OF_MEMORY = 5,
  CUML_ERROR_CUDA = 6,
  CUML_ERROR_FATAL_CUDA = 7,
  CUML_ERROR_SIZE_OVERFLOW = 8,
  CUML_ERROR_UNSUPPORTED = 9,
  CUML_ERROR_UNKNOWN = 10
} cuml_status_t;

enum { CUML_ERROR_MESSAGE_CAPACITY = 512 };

typedef enum {
  CUML_ERROR_FAILURE_SOURCE_PHYSICAL = 0,
  CUML_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT = 1,
  CUML_ERROR_FAILURE_SOURCE_UNKNOWN = 2
} cuml_error_failure_source_t;

typedef struct {
  cuml_status_t status;
  cuml_error_failure_source_t failure_source;
  uint64_t domain_id;
  uint64_t member_id;
  uint64_t generation;
  uint64_t requested_bytes;
  uint64_t alignment;
  int32_t cuda_error;
  char message[CUML_ERROR_MESSAGE_CAPACITY];
} cuml_error_detail_t;

cuml_error_detail_t const* cuml_last_error_detail(void);
char const* cuml_last_error_message(void);
void cuml_clear_error(void);

#ifdef __cplusplus
}
#endif
#endif
