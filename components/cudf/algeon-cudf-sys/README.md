# algeon-cudf-sys

`algeon-cudf-sys` (library `cudf_sys`) is the raw C ABI of `libcudf_rust`, the
C++ shim Algeon builds around libcudf. It declares opaque handles, size-tagged
option structs and status codes; every call is `unsafe`, and nothing here owns
or frees a handle. The crate builds only on Linux.

[`algeon-cudf`](https://docs.rs/algeon-cudf) wraps these calls in owned handles
that carry their execution context, and applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion).
The rules for code that touches raw handles are in
[`SAFETY.md`](https://github.com/algeon-dev/algeon/blob/main/components/cudf/algeon-cudf-sys/SAFETY.md).

The build script locates headers and libraries through `CUDF_INSTALL_DIR`
(an installed cuDF/RMM prefix), `CUDF_CPP_BUILD_DIR` (a configured build tree)
or `ALGEON_NATIVE_DIR`; the install prefix takes precedence. Building that
prefix is described in
[Building for an application](https://algeon.dev/build/rust-dependency).
