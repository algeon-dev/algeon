# cudf-nexus-sys

`cudf-nexus-sys` (`cudf_sys`) is the Linux-only raw C ABI for the cuDF shim
library, `libcudf_rust`. It exposes nominal opaque handles and C-compatible
types; every raw call is unsafe. Applications should use the safe owner,
runtime, and DataFrame APIs in [`cudf`](../README.md) instead.

The safe crate owns lifetime, execution, and error-policy decisions. This
crate owns the namespaced C headers, generated raw Rust declarations, and the
native shim boundary. It deliberately does not turn native handles into a
general-purpose Rust ownership API.

## Native libcudf discovery

Set one native-discovery environment variable for every build. An installed
prefix is the normal choice:

```bash
CUDF_INSTALL_DIR=~/code/cudf-nexus/cudf-install \
  cargo check -p cudf-nexus-sys
```

For development against an already configured cuDF build tree, use:

```bash
CUDF_CPP_BUILD_DIR=/path/to/cudf/build \
  cargo check -p cudf-nexus-sys
```

When both variables are present, `CUDF_INSTALL_DIR` takes precedence. Do not
mix headers, `libcudf`, or `librmm` from different prefixes: a process must
use one compatible libcudf/RMM runtime identity.

## Production and test hooks

The default build is the production ABI. It excludes `cudf_test_*` declarations
and test-only native controls. The `test-hooks` feature is only for this
workspace's integration and failure-path tests:

```bash
CUDF_INSTALL_DIR=~/code/cudf-nexus/cudf-install \
  cargo test -p cudf-nexus-sys --features test-hooks --test build_link_contract
```

With that feature, generated test declarations are available under
`cudf_sys::test_support`; they are not re-exported from the production root.
The production master header is
[`cudf_rust/cudf_rust.h`](native/include/cudf_rust/cudf_rust.h); test support
is a separate explicit header and is not part of the normal production
surface.

## Headers and bindings

The C headers under [`native/include/cudf_rust`](native/include/cudf_rust) are
the single source of truth for the raw ABI. Checked-in Rust declarations in
`src/bindings.rs` (and, for test hooks, `src/test_support.rs`) are generated
from [`bindings/wrapper.h`](bindings/wrapper.h) with pinned bindgen-cli 0.72.1.
Consumer builds never run bindgen.

From the repository root, regenerate after a C-header ABI change and verify
that checked-in output is current with these exact commands:

```bash
./cudf-nexus-sys/bindings/generate.sh
./cudf-nexus-sys/bindings/generate.sh --check
```

Do not edit generated binding files by hand.

## Focused validation

Use the same explicit prefix for every check. Run the no-feature check before
an all-features build so feature unification cannot hide a production
dependency on test hooks.

```bash
CUDF_INSTALL_DIR=~/code/cudf-nexus/cudf-install \
  cargo check -p cudf-nexus-sys
CUDF_INSTALL_DIR=~/code/cudf-nexus/cudf-install \
  cargo doc -p cudf-nexus-sys --no-deps
./cudf-nexus-sys/bindings/generate.sh --check

flock /tmp/cudf-gpu.lock \
  env CUDF_INSTALL_DIR=~/code/cudf-nexus/cudf-install \
  cargo test -p cudf-nexus-sys --features test-hooks --no-fail-fast
```

See [`SAFETY.md`](SAFETY.md) before introducing or changing raw-handle code.
