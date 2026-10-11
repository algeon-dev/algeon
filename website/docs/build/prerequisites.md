---
sidebar_position: 2
description: Install the upstream native-build prerequisites, then add the small set of tools required by this checkout.
---

# Prerequisites

Start with the [RAPIDS install guide](https://docs.rapids.ai/install/) for your
Linux distribution. It owns NVIDIA driver, CUDA toolkit, GPU support,
host-compiler, CMake, and Ninja compatibility. The upstream dependency
manifests are [cuDF](https://github.com/NVIDIA/cudf/blob/main/dependencies.yaml),
[cuVS](https://github.com/NVIDIA/cuvs/blob/main/dependencies.yaml),
[cuML](https://github.com/rapidsai/cuml/blob/main/dependencies.yaml), and
[cuGraph](https://github.com/rapidsai/cugraph/blob/main/dependencies.yaml).
The exact native sources in this checkout are the authority for this build;
their `dependencies.yaml` and `CMakeLists.txt` files take precedence if they
differ from those upstream references.

This page covers only what connects that upstream environment to DataFusion
Algeon. Linux is the only supported platform.

## CPU requirement

Every workspace build pins `target-cpu=x86-64-v4` in the root
[`.cargo/config.toml`](https://github.com/algeon-dev/algeon/blob/main/.cargo/config.toml),
so the compiled code uses AVX-512 instructions. On a CPU without x86-64-v4,
`algeon_server` and the test binaries can exit with `SIGILL` before `main`,
with no diagnostic. Check the build host and every host that will run the
binary before starting the native builds:

```bash
grep -o -m1 avx512f /proc/cpuinfo
/lib64/ld-linux-x86-64.so.2 --help | grep x86-64-v4
```

The first command prints `avx512f` on a CPU with AVX-512. The second, from
glibc 2.33 or newer, prints `x86-64-v4 (supported, searched)` when the CPU
implements the whole level.

Several AWS GPU instance families have AMD Zen 2 or Zen 3 cores, which
implement AVX2 but not AVX-512. According to the
[accelerated computing instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html),
G4ad and G5 use the AMD EPYC 7R32, and G6, G6e, Gr6, P5, and P5e use the
AMD EPYC 7R13.

To build for another CPU, change the `target-cpu` value in the
`[target.x86_64-unknown-linux-gnu]` table of `.cargo/config.toml`, for example
to `x86-64-v3`, before the first Rust build. Changing it later recompiles the
whole workspace. Docker packaging does not read that file: it takes the target
from `.config/policies/cpu-packaging.toml`, and the
[Docker image](./docker.md) it builds targets x86-64-v4 only.

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
page sets `CC`, `CXX`, and `CUDAHOSTCXX`; the native build scripts and CMake
read those variables. The example uses `/usr/bin/gcc-14` and
`/usr/bin/g++-14`, so confirm both paths exist before starting, or decide now
which RAPIDS-compatible host compiler you will substitute in that block:

```bash
ls /usr/bin/gcc-14 /usr/bin/g++-14
```

### Pin the toolchain in Cargo `[env]`

The build shell exports these machine-specific toolchain variables into the
current shell only. Write them once into a Cargo `[env]` table outside the
checkout so every shell, and `scripts/check_all.sh`, build with the same
toolchain whether or not the build shell block was run: use
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
selecting another toolchain. A command-specific architecture override uses
`cargo --config 'env.CMAKE_CUDA_ARCHITECTURES.value="<architecture-list>"'`;
changing architectures also requires rebuilding the native libraries.

Native binaries target the build GPU by default. When they must run on a
different GPU, choose the architecture list before the first native build as
described in [Building from Source](./from-source.md#choosing-cuda-architectures).

## Rust toolchain and test runner

Readers who only run the server need stable Rust; the rest of this section
is for running the validation gate.

Install Rust with the [official Rustup instructions](https://rustup.rs/). The
workspace declares its MSRV in `Cargo.toml` (currently Rust 1.95); nightly is
needed only for formatting. Install the latest stable nextest; the repository
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
