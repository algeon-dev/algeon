# algeon-cuml

`algeon-cuml` (library `cuml`) runs a fixed set of cuML algorithms on
device-resident cuDF data: HDBSCAN, PCA, TSVD, quasi-Newton logistic
regression, random-forest classification and regression, UMAP, and nvForest
inference over imported XGBoost, LightGBM and Treelite models. Fitted random
forests predict through the same nvForest path. A `CumlResources` handle binds
every native call to one `cudf::Runtime` device, stream and memory resource,
and each call revalidates that binding before it runs. Applications reach
these algorithms, except PCA, as SQL table functions through
[`algeon-datafusion`](https://docs.rs/algeon-datafusion); the
[cuML reference](https://algeon.dev/cuml) lists them, and SQL PCA runs
through cuVS.

The crate builds against upstream cuML `v26.08.00`, pinned without a fork;
[Fork changes](https://algeon.dev/reference/forks) records the one nvForest
patch the native build applies. `algeon-cuml-sys` declares the C ABI of the
shim and links it.
