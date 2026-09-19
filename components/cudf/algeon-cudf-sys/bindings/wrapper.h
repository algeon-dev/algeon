// Single source wrapper for checked-in cudf-sys raw bindings.
//
// The test-support include is enabled only by the root bindings helper while
// producing src/test_support.rs. It is not part of the production ABI input.
#include <cudf_rust/cudf_rust.h>

#ifdef CUDF_RUST_BINDINGS_TEST_SUPPORT
#include <cudf_rust/test_support.h>
#endif
