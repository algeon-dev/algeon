---
sidebar_position: 2
description: Install the upstream native-build prerequisites, then add the small set of tools required by this checkout.
---

# Prerequisites

This page covers building from source. The
[published server image](./docker.md#run-the-published-image) needs only an
NVIDIA driver, the NVIDIA Container Toolkit and a CPU that meets the
[CPU requirement](#cpu-requirement).

Start with the [RAPIDS install guide](https://docs.rapids.ai/install/) for your
Linux distribution. It owns NVIDIA driver, CUDA toolkit, GPU support,
host-compiler, CMake, and Ninja compatibility. The upstream dependency
manifests are [cuDF](https://github.com/NVIDIA/cudf/blob/main/dependencies.yaml),
[cuVS](https://github.com/NVIDIA/cuvs/blob/main/dependencies.yaml),
[cuML](https://github.com/rapidsai/cuml/blob/main/dependencies.yaml), and
[cuGraph](https://github.com/rapidsai/cugraph/blob/main/dependencies.yaml).
The exact native sources in this checkout are the authority for this build.
Their `dependencies.yaml` and `CMakeLists.txt` files take precedence if they
differ from those upstream references.

Linux is the only supported platform.

## CPU requirement

Every workspace build pins `target-cpu=x86-64-v3` in the root
[`.cargo/config.toml`](https://github.com/algeon-dev/algeon/blob/main/.cargo/config.toml),
and the compiled code uses AVX2, FMA, BMI1, and BMI2 instructions. Check the
build host and every host that will run the binary before starting the native
builds:

```bash
grep -o -m1 avx2 /proc/cpuinfo
/lib64/ld-linux-x86-64.so.2 --help | grep x86-64-v3
```

The first command prints `avx2` on a CPU with AVX2. The second, from glibc
2.33 or newer, prints `x86-64-v3 (supported, searched)` when the CPU
implements the whole level. Intel CPUs older than Haswell and AMD CPUs older
than Zen may lack AVX2, and on those `algeon_server` and the test binaries can
exit with `SIGILL` before `main`, with no diagnostic.

To build for another CPU, change the `target-cpu` value in the
`[target.x86_64-unknown-linux-gnu]` table of `.cargo/config.toml` before the
first Rust build, for example to `x86-64-v4` on a host with AVX-512. Changing
it later recompiles the whole workspace. Docker packaging does not read that
file. It takes the target from `.config/policies/cpu-packaging.toml`, and the
[Docker image](./docker.md) it builds targets x86-64-v3 only.

## Native build environment

The recursive checkout builds its pinned `libcudf`, `libcuvs`, `libcuml`,
and `libcugraph` sources locally. Before continuing, use the RAPIDS guidance to
install a compatible NVIDIA driver, CUDA toolkit, host compiler, CMake, and
Ninja. `nvcc` must be available from the CUDA installation you intend to use.

Install the checkout-specific host tools:

```bash
sudo apt-get install -y build-essential git clang mold patchelf
```

The root [`.cargo/config.toml`](https://github.com/algeon-dev/algeon/blob/main/.cargo/config.toml)
selects `clang` and `mold` for normal source builds.

The [build shell](./from-source.md#2-configure-the-build-shell) on the next
page sets `CC`, `CXX`, and `CUDAHOSTCXX`. The native build scripts and CMake
read those variables. The example uses `/usr/bin/gcc-14` and
`/usr/bin/g++-14`, so confirm both paths exist before starting, or decide now
which RAPIDS-compatible host compiler you will substitute in that block:

```bash
ls /usr/bin/gcc-14 /usr/bin/g++-14
```

### Pin the toolchain in Cargo `[env]`

The build shell exports these machine-specific toolchain variables into the
current shell only. Write them once into a Cargo `[env]` table outside the
checkout so that all shell sessions and `scripts/check_all.sh` compile with the
identical toolchain regardless of whether the build shell environment was
sourced: use
`<parent-of-checkout>/.cargo/config.toml` to cover every checkout and worktree
that [`scripts/dev/new_worktree.sh`](https://github.com/algeon-dev/algeon/blob/main/scripts/dev/new_worktree.sh)
creates as siblings, or `~/.cargo/config.toml` to apply it to all of this
user's Cargo projects:

```toml
[env]
CC = { value = "/usr/bin/gcc-14", force = true }
CXX = { value = "/usr/bin/g++-14", force = true }
CUDACXX = { value = "/usr/local/cuda/bin/nvcc", force = true }
CUDAHOSTCXX = { value = "/usr/bin/g++-14", force = true }
CUDA_HOME = { value = "/usr/local/cuda", force = true }
CUDA_PATH = { value = "/usr/local/cuda", force = true }
CUDAToolkit_ROOT = { value = "/usr/local/cuda", force = true }
CMAKE_CUDA_ARCHITECTURES = { value = "NATIVE", force = true }
```

Match these values to the build shell block. `force = true` gives Cargo the
configured value even when the shell exports a different spelling of the
toolchain path, preserving build-script fingerprints across shells. The gate
prints the effective values before starting checks. Change the table when
selecting another toolchain. A command-specific architecture override passes
the value on the Cargo command line:

```bash
cargo --config 'env.CMAKE_CUDA_ARCHITECTURES.value="<architecture-list>"'
```

Changing architectures also requires rebuilding the native libraries.

Native binaries target the build GPU by default. When they must run on a
different GPU, choose the architecture list before the first native build as
described in [Building from Source](./from-source.md#choosing-cuda-architectures).

## Registry application build environment

An application using `algeon-datafusion` from crates.io still needs the
[feature-selected native installations](./rust-dependency.md#2-build-the-required-native-libraries),
a CUDA toolkit and a compatible host toolchain. The `cuvs` dependency, also
required by `cugraph` and `cuml`, uses CMake package discovery and compiles a C
ABI probe. cuGraph discovery uses `readelf` and `ldd`. Install binutils and the
Linux loader tools. The safe cuGraph crate compiles its C++20 runtime shim
through `cc` and its CUDA edge-count shim through CMake and Ninja. These builds
use `CXX`, `CUDACXX` and `CUDAHOSTCXX` with the installed cuGraph, cuDF/RMM and
CUDA headers. The CUDA shim preserves explicit `CMAKE_CUDA_ARCHITECTURES`
selections; `NATIVE` uses the real architectures in the selected cuGraph library.
Preserve the `include/` trees when staging native install prefixes.

Registry consumers use the installed cuDF, cuGraph and cuML shims and checked-in
FFI bindings. They do not rebuild those RAPIDS libraries or run bindgen. Stable
Rust at or above the MSRV (1.95) is sufficient; nightly Rust and repository test
runners are required only for contributor checks.

The repository's `.cargo/config.toml` applies to Cargo commands run within the
checkout. An external application's CPU target and linker come from its own
Cargo configuration. Configure them for the hosts that will execute that
application. The repository's x86-64-v3 setting is not carried by registry
packages.

## Rust toolchain and test runner

Readers who only run the server or embed the library need stable Rust. The rest
of this section is for running the validation gate.

Install Rust with the [official Rustup instructions](https://rustup.rs/). The
workspace declares its MSRV in `Cargo.toml` (currently Rust 1.95). Nightly is
needed only for formatting. Install the latest stable nextest. The repository
does not pin its version. `scripts/check_all.sh` checks that nextest runs and
requires the exact `cargo-hack` version below before starting validation:

```bash
rustup toolchain install nightly
cargo install --locked cargo-nextest
cargo install --locked --version 0.6.45 cargo-hack
```

Enable nextest's run store once per machine, in its own user config (not this
repository's), so `-R latest` reruns and `cargo nextest replay` have runs to
work from:

```toml
# ~/.config/nextest/config.toml
[experimental]
record = true

[record]
enabled = true
```

Retention defaults to 30 days, 100 runs, or 1 GB, whichever comes first.

Continue with [Building from Source](./from-source.md).
