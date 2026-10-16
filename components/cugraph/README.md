# algeon-cugraph

`algeon-cugraph` (library `cugraph`) runs cuGraph traversal, centrality,
community, similarity, connectivity, structure and layout algorithms on
graphs built from cuDF columns. A graph is bound to one `cudf::Runtime`
device, stream and memory resource. cuGraph sizes its temporary buffers from
that resource's query budget, and an allocation failure reports the requested
bytes and the reservation that refused it. Applications reach these algorithms
as SQL table functions
through [`algeon-datafusion`](https://docs.rs/algeon-datafusion); the
[cuGraph reference](https://algeon.dev/cugraph) lists them.

The crate builds against Algeon's fork of cuGraph, upstream `v26.08.00` plus
the changes listed under [Fork changes](https://algeon.dev/reference/forks#cugraph).
`algeon-cugraph-sys` holds the raw bindings.
