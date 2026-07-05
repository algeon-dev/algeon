---
sidebar_position: 6
description: Failure modes actually hit while validating this guide, with root causes — not workarounds.
---

# Troubleshooting

Every entry below is a failure mode observed on real hardware, with the root
cause and the correct fix. If you hit something not listed, prefer diagnosing
which layer's contract broke over patching symptoms downstream.

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

`cudf-sys` compares libcudf source mtimes against the installed
`libcudf.so`. Two distinct causes:

1. You actually edited or pulled `cudf/` sources — rebuild:
   `cd cudf-rs/cudf && ./build.sh libcudf`.
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
          "$NEXUS_ROOT"/cugraph-rs/cugraph/cpp/build/libcugraph.so; do
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
one-RMM-per-process invariant ([Environment](./environment.md#why-one-shared-install-prefix)).

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
cd "$NEXUS_ROOT/cugraph-rs" && bash scripts/build_libcugraph.sh
# or just the prefix-completion step:
CUDF_INSTALL_DIR="$CUDF_INSTALL_DIR" bash scripts/bootstrap_cudf_install.sh
```

Only if the cudf build tree (`cudf/cpp/build/_deps`) was deleted — leaving the
bootstrap nothing to install from — rebuild libcudf first (keeping
`cpp/build`), then re-run the bootstrap:

```bash
cd "$NEXUS_ROOT/cudf-rs/cudf" && ./build.sh libcudf
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
RAPIDS 26.06 pins **gcc 14**, which is in range across CUDA 12.2–13.2
(CUDA 13.0 supports up to gcc 15). Confirm `CC`/`CXX`/`CUDAHOSTCXX` point at
`gcc-14`/`g++-14` ([Environment](./environment.md)) and that
`/usr/local/cuda` resolves to a supported toolkit
(`readlink -f /usr/local/cuda`) before building the stack.

## Parquet fixture tests are flaky in parallel

Some `cudf-rs` Parquet fixture tests share `/tmp` files and a Python
subprocess. If they fail intermittently under parallel execution, run them
single-threaded to confirm: `cargo test --all-features -- --test-threads=1`.

## A build works in your shell but fails in CI (or vice versa)

Inherited environment. Re-run the failing step under `env -i` with only
`env.sh` ([Environment](./environment.md#validated-against-shell-state-leakage));
diff the outcomes to find the leaking variable.
