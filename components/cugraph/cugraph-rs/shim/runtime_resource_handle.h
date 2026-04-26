#pragma once

// Crate-local C++ shim: cugraph-sys stays limited to upstream C API bindings,
// while cugraph-rs needs to build a raft::handle_t from a cudf Runtime stream
// and RMM memory resource before handing it to cugraph_create_resource_handle().

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
