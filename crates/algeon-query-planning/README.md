# algeon-query-planning

`algeon-query-planning` holds the native IR that Algeon compiles DataFusion
plans into, plus the optimizer rules, capability proof and admission contracts
over it. It depends on neither DataFusion nor any RAPIDS library, so a plan
can be analyzed and
rejected without a GPU. `algeon-query-runtime` and `algeon-query-engine` build
on it. Applications depend on
[`algeon-datafusion`](https://docs.rs/algeon-datafusion), which re-exports the
planning types they need.

The contracts are documented under Design on algeon.dev:
[module boundaries](https://algeon.dev/design#crates-and-ownership),
[native lowering](https://algeon.dev/design/native-lowering) and
[memory and query lifetime](https://algeon.dev/design/admission).
