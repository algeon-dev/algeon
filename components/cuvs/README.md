# algeon-cuvs

`algeon-cuvs` (library `cuvs`) runs cuVS vector algorithms on device-resident
cuDF columns: brute-force k-NN, IVF-Flat, IVF-PQ and CAGRA search, pairwise
distance, top-k selection, k-means and PCA. A `CuvsResources` handle binds
every call to one `cudf::Runtime` device, stream and memory resource. Each
call states who owns its input and output columns and when the device work on
them completes, and it checks tensor layout and output lengths before it
returns. Applications reach these algorithms as
SQL table functions through [`algeon-datafusion`](https://docs.rs/algeon-datafusion);
the [cuVS reference](https://algeon.dev/cuvs) lists them.

The crate builds against Algeon's fork of cuVS, upstream `v26.08.01` plus the
changes listed under [Fork changes](https://algeon.dev/reference/forks#cuvs).
`algeon-cuvs-sys` declares the C API and checks two parameter struct layouts
at build time.
