# algeon-cuvs-sys

`algeon-cuvs-sys` (library `cuvs_sys`) binds the `libcuvs` C API for Algeon.
The declarations are checked in; the build script locates cuVS through its
CMake package configuration and compiles a small C probe against the selected
headers, so a prefix whose k-means or PCA parameter struct layout differs
from the declarations fails at build time.

[`algeon-cuvs`](https://docs.rs/algeon-cuvs) builds its resource handle and
index types on these declarations; applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion).

The build script selects the cuVS prefix from `CUVS_ROOT` or, when that is
unset, from `cuvs-install` under `ALGEON_NATIVE_DIR`, as
[Building for an application](https://algeon.dev/build/rust-dependency) sets
it up. Discovery needs Linux, CMake and a CUDA toolkit, found through
`CUDAToolkit_ROOT`, `CUDA_HOME`, `CUDA_PATH` or `nvcc` on `PATH`.
