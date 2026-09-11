# cuGraph component

`components/cugraph` is the Algeon Rust graph integration component
built around [`libcugraph`](https://github.com/algeon-dev/cugraph). It is
part of the root Cargo workspace, not a second repository or workspace.
It exists only to support the native backend of
[Algeon](https://github.com/algeon-dev/algeon/tree/main/components/cugraph).
Its Rust package and import names are workspace contracts, not an external
semantic-versioning or publication promise.

## Focus

- Runtime-bound graph ownership and algorithm execution.
- Canonical prepared cuDF ingress and result egress with completion ordering.
- Allocation preflight facts for bounded Algeon admission.
- Structured errors that preserve machine-readable native failure context.

## Workspace

- `components/cugraph/algeon-cugraph/` - the internal safe execution crate, packaged as `cugraph` (import `cugraph`).
- `components/cugraph/algeon-cugraph-sys/` - low-level FFI bindings around the cuGraph C API (import `cugraph_sys`).
- `components/cugraph/cugraph/` - the pinned **[forked cuGraph](https://github.com/algeon-dev/cugraph)** source submodule used to build `libcugraph`.

Native defaults are repo-relative: `target/native/cudf-install` and
`target/native/cuvs-install` provide the shared cuDF/RMM and cuVS prefixes, and
`target/native/cugraph-build` is the cuGraph CMake build tree. Use the root
`scripts/build/build_libcugraph.sh` helper and root `scripts/check_all.sh` gate.

## validation

```bash
cargo clippy -p cugraph -p cugraph-sys --all-targets --all-features -- -D warnings
cargo +nightly fmt --all --check
cargo nt -E 'package(cugraph) | package(cugraph-sys)'
```

`cugraph-sys` intentionally generates bindings from the selected native
headers in `build.rs`; the selected headers and include paths are part of its
package/native contract.
