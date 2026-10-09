# algeon-build-support

`algeon-build-support` (`rapids_build_support`) is a published implementation dependency
of Algeon. Applications use
[`algeon-datafusion`](https://docs.rs/algeon-datafusion). See the
[API documentation boundary](https://algeon.dev/design#api-documentation).

The crate owns shared Cargo build-script discovery and native-link metadata.
It uses only the Rust standard library so native packages can depend on it
before their own libraries have been built.
