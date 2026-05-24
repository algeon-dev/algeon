#ifndef CUDF_RUST_GDS_H
#define CUDF_RUST_GDS_H

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Probe whether the cuFile driver can be opened by this process.
///
/// This is a deployment capability probe. It does not register any file or
/// buffer, and it closes the driver before returning on success.
cudf_status_t cudf_gds_driver_available(void);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_GDS_H
