# AGENTS.md

[README.md](README.md) and [algeon-cuvs-sys/README.md](algeon-cuvs-sys/README.md)
name the crates, the fork pin and native discovery. The `CuvsResources` and
prepared-lease rustdoc own the binding, ownership and completion contracts;
[GPU table functions](../../website/docs/design/gpu-table-functions.mdx) owns
ownership across the cuDF boundary, and
[Building from source](../../website/docs/build/from-source.md#build-cuvs)
the native build.

After a change to the pinned cuVS C headers, regenerate the checked-in
bindings with `scripts/dev/bindings.sh cuvs`. Generated files are not edited
by hand, and no crate build runs bindgen.
