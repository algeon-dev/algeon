# AGENTS.md

[DESIGN.md](DESIGN.md) owns the layering, execution, ownership, memory-domain
and ABI contracts of this component, and [README.md](README.md) names the
crates and the fork pin. Changes to unsafe code, lifetimes, raw handles or
thread traits follow [algeon-cudf-sys/SAFETY.md](algeon-cudf-sys/SAFETY.md).

After a change under `algeon-cudf-sys/native/include`, regenerate the
checked-in bindings with `scripts/dev/bindings.sh cudf`. Generated files are
not edited by hand, and no crate build runs bindgen.
