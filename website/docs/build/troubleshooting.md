---
sidebar_position: 6
description: The native-library failure modes whose fix is not documented anywhere else — cryptic configure aborts and runtime kernel errors.
---

# Troubleshooting

Toolchain and setup problems are documented where the tool is: missing
mold/clang, CMake and host-compiler versions in
[Prerequisites](./prerequisites.md); checkout-local native selectors in
[Building from Source](./from-source.md#2-configure-the-build-shell); and
test-runner, fingerprint, and `target/`-size issues in
[Tests](./tests.md#maintainer-reference).
This page keeps the failures whose error text does not point at the cause.

## cuGraph configure aborts: private RMM detected

`scripts/build/build_libcugraph.sh` refuses to finish when the configure resolved an
RMM or rapids_logger other than the one exported by `CUDF_INSTALL_DIR`, or when
CPM-built copies linger under `target/native/cugraph-build/_deps`. This protects the
one-RMM-per-process invariant
([native dependency relationship](./from-source.md#2-configure-the-build-shell)).

Usual cause: the build dir predates the external-RMM configure. Remove
`target/native/cugraph-build` and re-run the script. Do not delete the check or point
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
installs them from the canonical `target/native/cudf-build/_deps/`). This error means the
prefix has `rapids_logger` but is missing its `fmt`/`spdlog` closure — i.e. the
bootstrap did not run or could not complete.

`scripts/build/build_libcugraph.sh` runs the bootstrap automatically, so the usual fix is
simply to run it (or the bootstrap alone) with the canonical cuDF build cache
intact:

```bash
cd "$NEXUS_ROOT" && bash scripts/build/build_libcugraph.sh
# or just the prefix-completion step:
CUDF_INSTALL_DIR="$CUDF_INSTALL_DIR" bash scripts/build/bootstrap_cudf_install.sh
```

Only if the canonical cuDF build cache (`target/native/cudf-build/_deps`) was deleted — leaving the
bootstrap nothing to install from — rebuild libcudf first (keeping
`target/native/cudf-build`), then re-run the bootstrap:

```bash
cd "$NEXUS_ROOT" && bash scripts/build/build_libcudf.sh
```

## Native build rejects a CUDA architecture contract

The cuVS and cuGraph helpers stop before an expensive downstream build when a
dependency was built with another `CMAKE_CUDA_ARCHITECTURES` value or contains
a different full-kernel real cubin set. Small embedded dependency stub sets are
ignored. The check compares the native layers against each other only; it does
not compare them against the GPU installed in this host.

Choose one value and rebuild in dependency order:

```bash
export CMAKE_CUDA_ARCHITECTURES=NATIVE # or an explicit list such as 89-real;120a-real
bash scripts/build/build_libcudf.sh
bash scripts/build/build_libcuvs.sh
bash scripts/build/build_libcugraph.sh
```

The old cuDF-only `CUDF_CMAKE_CUDA_ARCHITECTURES` selector is intentionally
rejected. The wrapper maps the shared CMake selector to cuDF internally.

The record each layer carries, and how to inspect it, is described under
[The architecture contract record](./from-source.md#the-architecture-contract-record).

## GPU op aborts: `cudaErrorNoKernelImageForDevice`

The binary loads and links fine, then the first kernel launch aborts. The
native build did not include full kernels for the GPU it is running on —
`CMAKE_CUDA_ARCHITECTURES=NATIVE` bakes kernels **only** for the GPU present at
build time. Set one explicit deployment architecture list and rebuild cuDF,
cuVS, and cuGraph in that order. The build helpers then confirm the three
layers agree. For deeper inspection, list each `.so`:

```bash
for so in "$CUDF_INSTALL_DIR"/lib/libcudf.so \
          "$CUVS_INSTALL_DIR"/lib/libcuvs.so \
          "$NEXUS_ROOT"/target/native/cugraph-build/libcugraph.so; do
  echo "== $so =="
  cuobjdump --list-elf "$so" | grep -oE 'sm_[0-9]+[a-z]?' | sort | uniq -c
done
```

Comparable, large counts per architecture are healthy; ~a dozen entries for
an architecture means stubs only.

## SCC fails on Blackwell (sm_120): `cudaErrorInvalidDeviceFunction`

```text
cugraph_strongly_connected_components failed:
  copy_if failed on 2nd step: cudaErrorInvalidDeviceFunction
```

Known upstream cuGraph defect on Blackwell
(`sm_120`, e.g. RTX 50-series / RTX PRO 6000)
([rapidsai/cugraph#5543](https://github.com/rapidsai/cugraph/issues/5543)) —
CCCL 3.4.0's CUB select tuning has no policy for compute capability 12.0.
The pinned `cugraph` fork routes SCC through a host-side fallback on
single-GPU `sm_120` while keeping the C API result shape. If you see this,
you built upstream cuGraph (or an old fork commit) instead of the pinned
submodule.
