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
libcudf/libcugraph build did not include full kernels for the GPU it is
running on — `CMAKE_CUDA_ARCHITECTURES=NATIVE` bakes kernels **only** for
the GPU present at build time. Rebuild listing every deployment
architecture, and verify coverage:

```bash
cuobjdump --list-elf "$CUDF_INSTALL_DIR/lib/libcudf.so" \
  | grep -oE 'sm_[0-9]+' | sort | uniq -c
```

Comparable, large counts per architecture are healthy; ~a dozen entries for
an architecture means stubs only.

## cuGraph configure aborts: private RMM detected

`scripts/build_cpp.sh` refuses to finish when the configure resolved an RMM
or rapids_logger other than the one exported by `CUDF_INSTALL_DIR`, or when
CPM-built copies linger under `cugraph/cpp/build/_deps`. This protects the
one-RMM-per-process invariant ([Environment](./environment.md#why-one-shared-install-prefix)).

Usual cause: the build dir predates the external-RMM configure. Remove
`cugraph/cpp/build` and re-run the script. Do not delete the check or point
the build at a private RMM — a process that loads two `librmm.so` copies
corrupts memory-resource state at runtime.

## Downstream configure demands spdlog: stale export metadata

```text
CMake Error ... rapids_logger-dependencies.cmake ... find_dependency(spdlog)
Could not find a package configuration file provided by "spdlog"
```

A libcudf build directory that predates a change in how a dependency is
linked (for example rapids_logger switching static → shared) can install
stale package metadata: the old configure recorded spdlog/fmt in the export
set, and the re-run keeps installing a `rapids_logger-dependencies.cmake`
that demands them. A shared rapids_logger links spdlog privately and its
installed package requires no spdlog at all.

Incremental libcudf rebuilds are only trustworthy while dependency link
modes are unchanged. After pulling a `cudf` submodule bump that changes
them, rebuild clean:

```bash
rm -rf "$NEXUS_ROOT/cudf-rs/cudf/cpp/build" "$CUDF_INSTALL_DIR"
cd "$NEXUS_ROOT/cudf-rs/cudf" && ./build.sh libcudf
```

## SCC fails on RTX 50-series: `cudaErrorInvalidDeviceFunction`

```text
cugraph_strongly_connected_components failed:
  copy_if failed on 2nd step: cudaErrorInvalidDeviceFunction
```

Known upstream defect of clean source-built cuGraph v26.06 on `sm_120`
([rapidsai/cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543)) —
CCCL 3.4.0's CUB select tuning has no policy for compute capability 12.0.
The pinned `cugraph` fork routes SCC through a host-side fallback on
single-GPU `sm_120` while keeping the C API result shape. If you see this,
you built upstream cuGraph (or an old fork commit) instead of the pinned
submodule.

## `nvcc` errors against the host compiler

An older CUDA toolkit's `nvcc` can reject a newer gcc/glibc (e.g. an `rsqrt`
`noexcept` mismatch on trivial `.cu` files). Check which toolkit
`/usr/local/cuda` resolves to (`readlink -f /usr/local/cuda`) and point
`CUDA_HOME` at one that compiles a trivial kernel with your gcc before
building the stack.

## Parquet fixture tests are flaky in parallel

Some `cudf-rs` Parquet fixture tests share `/tmp` files and a Python
subprocess. If they fail intermittently under parallel execution, run them
single-threaded to confirm: `cargo test --all-features -- --test-threads=1`.

## A build works in your shell but fails in CI (or vice versa)

Inherited environment. Re-run the failing step under `env -i` with only
`env.sh` ([Environment](./environment.md#validated-against-shell-state-leakage));
diff the outcomes to find the leaking variable.
