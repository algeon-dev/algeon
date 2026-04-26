#pragma once

#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cugraph_rs_runtime_resource_handle
    cugraph_rs_runtime_resource_handle_t;

cugraph_rs_runtime_resource_handle_t *
cugraph_rs_runtime_resource_handle_create(void *cuda_stream,
                                          void *device_memory_resource);

cugraph_resource_handle_t *cugraph_rs_runtime_resource_handle_get(
    cugraph_rs_runtime_resource_handle_t *handle);

void cugraph_rs_runtime_resource_handle_destroy(
    cugraph_rs_runtime_resource_handle_t *handle);

#ifdef __cplusplus
}
#endif
