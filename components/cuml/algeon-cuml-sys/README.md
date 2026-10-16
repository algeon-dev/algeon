# algeon-cuml-sys

`algeon-cuml-sys` (library `cuml_sys`) declares the C ABI of `libcuml_rust`,
the C++ shim Algeon builds around cuML, and links it. The shim exposes the
algorithms Algeon uses, catches native exceptions at the boundary and keeps
the failure detail for the caller to copy out.

[`algeon-cuml`](https://docs.rs/algeon-cuml) owns the handles, models and
stream pools these functions create; applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion).

On Linux, the build script links an installed shim from `CUML_INSTALL_DIR` or from
`cuml-install` under `ALGEON_NATIVE_DIR`, and takes the matching cuDF and
cuVS prefixes from `CUDF_INSTALL_DIR` and `CUVS_INSTALL_DIR` or the same
root, the layout that
[Building for an application](https://algeon.dev/build/rust-dependency)
produces.
