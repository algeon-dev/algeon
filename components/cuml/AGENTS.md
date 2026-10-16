# AGENTS.md

[README.md](README.md) and [algeon-cuml-sys/README.md](algeon-cuml-sys/README.md)
name the crates, the supported algorithms, the upstream pin and native
discovery. The `CumlResources` rustdoc owns the runtime, memory and stream-pool
contracts; [Fork changes](../../website/docs/reference/forks.md) records the
nvForest patch the native build applies, and
[Building from source](../../website/docs/build/from-source.md#build-cuml)
the native build.

After a change to the shim headers under `algeon-cuml-sys/native`, regenerate
the checked-in bindings with `scripts/dev/bindings.sh cuml`. Generated files
are not edited by hand, and no crate build runs bindgen.
