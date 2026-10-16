# AGENTS.md

[README.md](README.md) and
[algeon-cugraph-sys/README.md](algeon-cugraph-sys/README.md) name the crates,
the fork pin and native discovery. The crate rustdoc owns the handle, graph
type-state and interop contracts;
[GPU table functions](../../website/docs/design/gpu-table-functions.mdx) owns
ownership across the cuDF boundary, and
[Building from source](../../website/docs/build/from-source.md#build-cugraph)
the native build.

After a change to the fork's C headers, regenerate the checked-in bindings
with `scripts/dev/bindings.sh cugraph`. Generated files are not edited by
hand, and no crate build runs bindgen.
