# algeon-interop

`algeon-interop` (`rapids_interop`) is a published implementation dependency
of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion). See the
[API documentation boundary](https://algeon.dev/design#api-documentation).

The crate owns shared buffer, execution-binding and completion contracts for
Algeon GPU components. Producers and consumers retain storage through the
completion obligations documented on the exported types.
