---
sidebar_position: 1
---

# Public Surface

These are public contracts — treat them as stable:

- Report row schemas, TSV column ordering, and the
  `NATIVE_*_TSV_HEADER` / `NATIVE_*_TSV_SCHEMA_VERSION` constants.
- `NATIVE_CUDF_DATAFUSION_METRIC_NAMES` and the `NativeCudfExec` /
  `NativeCudfComposedExec` / `NativeCudfGpuProducerExec` wrappers.
- Workload env-var names and item shapes (TPC-H / TPC-DS / ClickBench).
- The closed `Error` / `ErrorCode` categories and structured `Facts` in
  `src/error.rs`.

:::note[Scaffold]
This page is a scaffold. Expand each contract with its schema and versioning
notes.
:::
