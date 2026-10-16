# algeon-query-runtime

`algeon-query-runtime` owns the attempt lifecycle, admission to a GPU, the
per-device memory ledger and its immutable grants, resident caches, and the
observation snapshot that reports and metrics read. It depends on
`algeon-query-planning` and `algeon-cudf`, and `algeon-query-engine` and
`algeon-datafusion` build on it. Applications depend on `algeon-datafusion`,
which re-exports the runtime types they need.

[Memory and query lifetime](https://algeon.dev/design/admission) and
[runtimes and threads](https://algeon.dev/design#runtimes-and-threads)
document the contracts.
