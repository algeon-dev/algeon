#ifndef CUML_RUST_HANDLE_H
#define CUML_RUST_HANDLE_H

#include "types.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

cuml_status_t cuml_handle_create(int32_t device, void* cuda_stream,
                                 void* device_memory_resource,
                                 void* memory_resource_owner,
                                 cuml_handle_t* out_handle);
cuml_status_t cuml_handle_destroy(cuml_handle_t handle);

#ifdef __cplusplus
}
#endif
#endif
