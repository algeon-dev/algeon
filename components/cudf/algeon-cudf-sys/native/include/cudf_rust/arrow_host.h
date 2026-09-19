#ifndef CUDF_RUST_ARROW_HOST_H
#define CUDF_RUST_ARROW_HOST_H

#include "column.h"
#include "execution_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Import one host-resident Arrow C Data Interface column through libcudf's
/// host Arrow column importer.
///
/// `schema` must point to an `ArrowSchema`, and `array` must point to the
/// matching `ArrowArray`. `options` must name an explicit valid device,
/// non-NULL stream, and non-NULL memory resource. The shim synchronizes the
/// stream before returning because the Arrow host roots are borrowed only for
/// the duration of this call.
cudf_status_t cudf_arrow_host_column_import_ex(void const* schema,
                                               void const* array,
                                               cudf_exec_options_t const* options,
                                               cudf_column_handle* out);

#ifdef __cplusplus
}
#endif

#endif  // CUDF_RUST_ARROW_HOST_H
