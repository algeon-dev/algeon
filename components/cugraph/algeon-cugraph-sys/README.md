# algeon-cugraph-sys

`algeon-cugraph-sys` (library `cugraph_sys`) is the checked-in Rust declaration
of the cuGraph C API that Algeon links, generated with bindgen from the fork's
headers. The build script resolves the include, link and runtime directories
of one cuGraph installation and checks that its version matches the crate.
Every function is `unsafe`, and nothing here owns a graph or a result. The
crate builds only on Linux.

[`algeon-cugraph`](https://docs.rs/algeon-cugraph) owns the graph and result
handles declared here and maps native status codes to typed errors;
applications use [`algeon-datafusion`](https://docs.rs/algeon-datafusion).

The build script takes an installed cuGraph from `cugraph-install` under
`ALGEON_NATIVE_DIR`, with the matching cuDF prefix from `CUDF_INSTALL_DIR` or
the same root. In a source checkout it can instead read a configured cuGraph
build tree named by `CUGRAPH_CPP_BUILD_DIR` or `CUGRAPH_CMAKE_BUILD_DIR`,
with cuDF and cuVS taken from `CUDF_INSTALL_DIR` and `CUVS_INSTALL_DIR`.
When none of those variables is set, a source checkout configures and builds
cuGraph itself; `CUGRAPH_SYS_AUTO_BUILD=0` disables that and requires an
existing build tree. Building the prefix is described in
[Building for an application](https://algeon.dev/build/rust-dependency).
