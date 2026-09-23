#ifndef CUML_RUST_TYPES_H
#define CUML_RUST_TYPES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUML_MATRIX_ROW_MAJOR = 0,
  CUML_MATRIX_COLUMN_MAJOR = 1
} cuml_matrix_layout_t;

typedef struct cuml_handle cuml_handle;
typedef cuml_handle* cuml_handle_t;

#ifdef __cplusplus
}
#endif
#endif
