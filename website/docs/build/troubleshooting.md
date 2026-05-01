---
sidebar_position: 7
description: Failure modes actually hit while validating this guide, with root causes — not workarounds.
---

# Troubleshooting

Every entry below is a failure mode observed on real hardware, with the root
cause and the correct fix. If you hit something not listed, prefer diagnosing
which layer's contract broke over patching symptoms downstream.

## `cargo build` fails: mold or clang missing

```text
error: linker `clang` not found
# or
collect2: fatal error: cannot find 'ld'
clang: error: invalid linker name in argument '-fuse-ld=mold'
```

`datafusion-nexus` selects clang + mold in `.cargo/config.toml`. Install both
host packages (and the matching `~/.cargo` linker setting when building sibling
crates) and re-run the Cargo command:

```bash
sudo apt-get install -y clang mold
mold --version
clang --version
```

See [Prerequisites](./prerequisites.md#rust-linker-mold).

## CMake refuses to configure: version too old

```text
CMake Error: CMake 3.30.4 or higher is required. You are running version 3.28.x
```

Ubuntu 24.04's archive cmake is too old for the RAPIDS 26.06 stack. Install
from the Kitware repository ([Prerequisites](./prerequisites.md#cmake--3304)).

## `cargo build` warns: libcudf appears stale

```text
warning: libcudf appears stale: N source file(s) under cudf/cpp/{src,include}
are newer than .../libcudf.so
```

`cudf-nexus-sys` compares libcudf source mtimes against the installed
`libcudf.so`. Two distinct causes:

1. You actually edited or pulled `cudf/` sources — rebuild:
   `cd cudf-nexus/cudf && ./build.sh libcudf`.
2. `CUDF_INSTALL_DIR` points at a **different (older) install tree** than
   the one you rebuilt — tests would load the stale libcudf. Check
   `echo $CUDF_INSTALL_DIR` and realign it with the tree that
   `./build.sh libcudf` actually installed into.

## GPU op aborts: `cudaErrorNoKernelImageForDevice`

The binary loads and links fine, then the first kernel launch aborts. The
libcudf or libcugraph build did not include full kernels for the GPU it is
running on — `CMAKE_CUDA_ARCHITECTURES=NATIVE` bakes kernels **only** for
the GPU present at build time. Rebuild **both** native libraries listing every
deployment architecture (`CUDF_CMAKE_CUDA_ARCHITECTURES` for libcudf,
`CMAKE_CUDA_ARCHITECTURES` for `build_libcugraph.sh`), and verify coverage of
each `.so`:

```bash
for so in "$CUDF_INSTALL_DIR"/lib/libcudf.so \
          "$NEXUS_ROOT"/cugraph-nexus/cugraph/cpp/build/libcugraph.so; do
  echo "== $so =="
  cuobjdump --list-elf "$so" | grep -oE 'sm_[0-9]+' | sort | uniq -c
done
```

Comparable, large counts per architecture are healthy; ~a dozen entries for
an architecture means stubs only.

## cuGraph configure aborts: private RMM detected

`scripts/build_libcugraph.sh` refuses to finish when the configure resolved an
RMM or rapids_logger other than the one exported by `CUDF_INSTALL_DIR`, or when
CPM-built copies linger under `cugraph/cpp/build/_deps`. This protects the
one-RMM-per-process invariant ([Environment](./environment.md#dependency-relationship)).

Usual cause: the build dir predates the external-RMM configure. Remove
`cugraph/cpp/build` and re-run the script. Do not delete the check or point
the build at a private RMM — a process that loads two `librmm.so` copies
corrupts memory-resource state at runtime.

## cuGraph configure demands spdlog / fmt: prefix not fully bootstrapped

```text
CMake Error ... rapids_logger-dependencies.cmake ... find_dependency(spdlog)
Could not find a package configuration file provided by "spdlog"
```

cuGraph does a real `find_package(rmm)`, whose closure pulls in
`rapids_logger`, which in turn `find_dependency()`s `fmt` and `spdlog`. Those
packages are **not** part of the libcudf install; `bootstrap_cudf_install.sh`
adds `rapids_logger`, `nvtx3`, `fmt`, and `spdlog` to the prefix together (it
installs them from the cudf C++ build tree's `_deps/`). This error means the
prefix has `rapids_logger` but is missing its `fmt`/`spdlog` closure — i.e. the
bootstrap did not run or could not complete.

`build_libcugraph.sh` runs the bootstrap automatically, so the usual fix is
simply to run it (or the bootstrap alone) with the cudf `cpp/build` tree
intact:

```bash
cd "$NEXUS_ROOT/cugraph-nexus" && bash scripts/build_libcugraph.sh
# or just the prefix-completion step:
CUDF_INSTALL_DIR="$CUDF_INSTALL_DIR" bash scripts/bootstrap_cudf_install.sh
```

Only if the cudf build tree (`cudf/cpp/build/_deps`) was deleted — leaving the
bootstrap nothing to install from — rebuild libcudf first (keeping
`cpp/build`), then re-run the bootstrap:

```bash
cd "$NEXUS_ROOT/cudf-nexus/cudf" && ./build.sh libcudf
```

## SCC fails on Blackwell (sm_120): `cudaErrorInvalidDeviceFunction`

```text
cugraph_strongly_connected_components failed:
  copy_if failed on 2nd step: cudaErrorInvalidDeviceFunction
```

Known upstream defect of clean source-built cuGraph v26.06 on Blackwell
(`sm_120`, e.g. RTX 50-series / RTX PRO 6000)
([rapidsai/cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543)) —
CCCL 3.4.0's CUB select tuning has no policy for compute capability 12.0.
The pinned `cugraph` fork routes SCC through a host-side fallback on
single-GPU `sm_120` while keeping the C API result shape. If you see this,
you built upstream cuGraph (or an old fork commit) instead of the pinned
submodule.

## `nvcc` errors against the host compiler

`nvcc` rejects host compilers outside the range its CUDA toolkit supports —
either an older toolkit against a newer gcc (e.g. an `rsqrt` `noexcept`
mismatch on trivial `.cu` files), or a gcc newer than the toolkit allows.
RAPIDS 26.06 pins **gcc 14**, but it must be paired with a toolkit that
supports it. The recommended pairing is
CUDA 13.x plus gcc 14 (CUDA 13.0 supports up to gcc 15); CUDA 12.2 supports gcc
only through 12.2. Confirm `CC`/`CXX`/`CUDAHOSTCXX` select a compiler accepted
by the exact toolkit and that `/usr/local/cuda` resolves to that toolkit
(`readlink -f /usr/local/cuda`) before building the stack. The standard
[Environment](./environment.md) gcc-14 settings assume the recommended CUDA
13.x pairing.

## Datafusion Nexus tests fail under bare parallel Cargo

`datafusion-nexus` GPU tests use process-global CUDA and query-service state.
Do not diagnose failures with broad `cargo test`, even with
`--test-threads=1`: serial threads cannot restore the engine's process-global
allocator or policy fixtures. The former `cargo-diagnostic` runner now fails
closed for this reason. Use the canonical runner instead:

```bash
bash scripts/check_all.sh
```

A bare `cargo nextest run` also does not take the repository-wide GPU lock; use
`check_all.sh --lane gpu-shared` or `check_all.sh --lane gpu-exclusive` for a
locked targeted GPU lane.

## A build works in your shell but fails in CI (or vice versa)

Inherited environment. Re-run the failing step under `env -i` with only
`env.sh` ([Environment](./environment.md#validated-against-shell-state-leakage));
diff the outcomes to find the leaking variable.

## Cargo recompiles cudf-nexus-sys after only PATH changed

`cudf-nexus-sys` no longer treats ambient `PATH` as a rebuild input. It tracks
explicit `CC`/`CXX`, cuDF/CUDA discovery variables, lint mode, optional
`CUDF_RUST_CLANG_FORMAT` / `CUDF_RUST_CLANG_TIDY` / `CUDF_RUST_CMAKE`, and the
selected runtime libraries. A PATH-only change (venv activate, agent tool
shim, PATH normalization) should leave `cudf-nexus-sys` fresh when source and those
explicit inputs are unchanged.

If you still see a PATH-driven invalidation, you are likely on an older
`cudf-nexus` revision. Confirm the sibling checkout includes the fingerprint
matrix and re-run:

```bash
cd "${NEXUS_ROOT}/cudf-nexus"
bash scripts/check_fingerprint_matrix.sh
```

Diagnose the first dirty unit without publishing secrets:

```bash
CARGO_LOG=cargo::core::compiler::fingerprint=info \
  cargo nextest run -p cudf-nexus-sys --no-run 2>&1 \
  | rg 'fingerprint dirty|dirty:|Compiling' \
  | sed -E 's/(PATH=)[^ ]*/\1<redacted>/g'
```

## `target/` is hundreds of gigabytes

Historical feature hashes, full-debuginfo test links, and interrupted mold
temps accumulate under `target/debug/deps`. Inventory and clean explicitly:

```bash
bash scripts/dev/clean_target_artifacts.sh
bash scripts/dev/clean_target_artifacts.sh --mold-only --yes
# only when you accept a cold rebuild:
bash scripts/dev/clean_target_artifacts.sh --clean-cargo-target --yes
```

Do not delete `target/` while `cargo`, `rustc`, or `mold` is running. Prefer
the single-harness edit loop in [Tests](./tests.md#one-package-one-harness)
before paying for an all-feature warm-up.
