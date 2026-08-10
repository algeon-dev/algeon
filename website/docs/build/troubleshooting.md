---
sidebar_position: 7
description: The native-library failure modes whose fix is not documented anywhere else — cryptic configure aborts and runtime kernel errors.
---

# Troubleshooting

Toolchain and setup problems are documented where the tool is: missing
mold/clang, CMake and host-compiler versions in
[Prerequisites](./prerequisites.md); stale-install and shell-leakage issues in
[Environment](./environment.md); test-runner, fingerprint, and `target/`-size
issues in [Tests](./tests.md). This page keeps only the failures whose error
text does not point at the cause — all four were observed on real hardware.

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
