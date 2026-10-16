# algeon-build-support

`algeon-build-support` (library `rapids_build_support`) is the build-script
helper shared by the Algeon crates that link RAPIDS libraries; it discovers
native prefixes and emits link metadata. It uses only the Rust standard library so those
crates can depend on it before any native library exists. The `-sys` crates,
`algeon-query-engine` and `algeon-datafusion` list it as a build dependency.
An application lists it only to bake a native RUNPATH into its own binary;
otherwise `LD_LIBRARY_PATH` supplies runtime discovery.
