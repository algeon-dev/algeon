---
sidebar_position: 5
sidebar_label: Docker
description: Run the published Algeon Flight SQL server image, or build an image from locally built native artifacts, and smoke-test Flight SQL.
---

# Docker

The published image `ghcr.io/algeon-dev/algeon-server` runs the standalone
Flight SQL server with cuGraph, cuVS and cuML SQL. Building your own image
packages the server with the RAPIDS libraries of a local source build, for
other GPU architectures, other features or a modified checkout.

## Run the published image

From the directory that holds your Parquet files:

```bash
docker run --rm --gpus all -p 127.0.0.1:50051:50051 -v "$PWD:/data:ro" ghcr.io/algeon-dev/algeon-server
```

The host needs:

- a CPU that implements x86-64-v3, as described under
  [CPU requirement](./prerequisites.md#cpu-requirement);
- an NVIDIA driver R580 or newer, the first branch that supports CUDA 13;
- the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  which `--gpus all` uses to expose the driver to the container;
- a GPU from the table below.

The image carries kernel images for five targets. A GPU of compute capability
8.9 runs the `sm_86` kernels, because a kernel image runs on GPUs of the same
major version and an equal or higher minor version.

| Compute capability | Kernel image | GPUs |
| --- | --- | --- |
| 8.0 | `sm_80` | A100, A30 |
| 8.6 | `sm_86` | A10, A10G, A40, RTX A6000, GeForce RTX 30 series |
| 8.9 | `sm_86` | L4, L40S, RTX 6000 Ada, GeForce RTX 40 series |
| 9.0 | `sm_90a` | H100, H200 |
| 10.0, 10.3 | `sm_100f` | B200, B300 |
| 12.0 | `sm_120a` | RTX PRO 6000 Blackwell, GeForce RTX 50 series |

Turing GPUs (T4, GeForce RTX 20 series) and older ones are not supported;
their first GPU query fails with `no kernel image is available for execution
on the device`. Only compute capability 12.0 has been verified on hardware.
The other rows follow from the kernel images in the release build and have not
run on those GPUs.

Each Parquet file and each directory of Parquet files at the top level of the
mounted `/data` becomes a table at startup, so `events.parquet` is queried as
`events`. [Data root](../integration/configuration.mdx#data-root)
gives the naming rules. Without a mount the server starts with no tables. The
server listens on host loopback port 50051 with authentication disabled;
submit a query with the
[client quickstart](../integration/flight-sql-server.mdx#quickstart).
Configuration files, authentication and TLS work as described under
[Run the server](#run-the-server).

## Build your own image

:::note[Building an image requires a source build]

Complete [Building from Source](./from-source.md) before following this
section. `docker/build-image.sh` packages the locally built native libraries.
It cannot create their cuDF, cuVS, cuML, or cuGraph artifacts itself.

:::

The image build selects `cugraph`, `cuvs`, and `cuml` by default. NVML device
diagnostics are opt-in through `FEATURES`. A `FEATURES=cuml` payload also
stages the canonical cuVS runtime dependency.

`docker/build-image.sh` is the single entry point. On the host it builds
`algeon_server`, stages the binary and the locally built cuDF/RMM libraries
into `docker/dist`, then adds cuVS, cuGraph, and cuML libraries for the
selected features. cuGraph always brings its cuVS runtime dependency. When
`strip` is available it runs `strip --strip-unneeded` on the staged files, so
they can be smaller than the source install; `NO_STRIP=1` keeps the symbols.
When `patchelf` is available it rewrites RUNPATHs to `/app/lib` and
`$ORIGIN`.

The script then checks the staged loader closure with
`docker/validate-loader-closure.sh`, stages the [license texts](#licenses),
writes `payload-manifest.json` and `SHA256SUMS`, and runs `docker build` with
`docker/dist` as the build context. Inside the build, the harvest stage checks
the manifest and `SHA256SUMS` against the hashes verified on the host and
every other file against `SHA256SUMS`, rejects a CUDA runtime image older than
the toolkit that built the payload, reruns the closure check, and copies out
only the CUDA toolkit libraries the server links or dlopens. The runtime stage
starts from `ubuntu:24.04`, the OS of the CUDA images, and reruns the closure
check on the assembled image. The `nvidia/cuda` base images would add only the
CUDA forward-compatibility libraries, which an R580 or newer driver does not
need.

## Prerequisites

The default full image expects these root-owned artifacts, plus the matching
`cudf-build`, `cuvs-build`, `cugraph-build` and `cuml-build` trees, which
hold the license files of the fetched native dependencies:

```text
algeon/
├── target/native/cudf-install/lib/libcudf.so
├── target/native/cudf-install/lib/librmm.so
├── target/native/cudf-install/lib/libcudf_rust.so
├── target/native/cuvs-install/lib/libcuvs.so
├── target/native/cugraph-install/lib/libcugraph.so
├── target/native/cuml-install/lib/libcuml.so
├── target/native/cuml-install/lib/libcuml_rust.so
├── target/native/cuml-install/lib/libtreelite.so
└── target/native/cuml-install/lib/libnvforest.so
```

The cuDF script also installs `libcudf_rust_rmm_interop.so` beside the shim.
`docker/build-image.sh` stages both automatically. It reads the
matching install prefixes and build trees under `ALGEON_NATIVE_DIR`; the
`CUDF_CPP_BUILD_DIR`, `CUVS_CPP_BUILD_DIR`, `CUGRAPH_CMAKE_BUILD_DIR` and
`CUML_CPP_BUILD_DIR` selectors of the native build scripts move a build tree.

The cuDF-only `FEATURES=''` and `FEATURES=nvml` images omit cuVS, cuML, and
cuGraph, so they need none of those native artifacts. `nvml` adds GPU
utilization, NVML framebuffer and device identity diagnostics. It loads
`libnvidia-ml.so.1` from the driver at run time, and admission never reads it.

`readelf` and `ldd` are required for the loader-closure checks. `patchelf` and
`strip` (binutils) are recommended. Without `patchelf` the image resolves its
libraries through `LD_LIBRARY_PATH=/app/lib`. The build records the CUDA
toolkit release of the `nvcc` found through `CUDA_HOME`, `CUDA_PATH` or
`PATH`, in that order. The native install prefixes carry no toolkit record,
so the native libraries are assumed to come from that same toolkit.

Install Docker and configure GPU containers with the
[NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The default CUDA runtime image is CUDA 13.2.1. The harvest stage fails when
the `CUDA_RUNTIME_IMAGE` major differs from the build toolkit's or its minor
is older. The harvested runtime libraries are minor-version compatible within
CUDA 13, so the host needs an NVIDIA driver that supports CUDA 13 (R580 or
newer).

### GPU architectures

The bundled cuDF, cuVS, cuGraph, and cuML libraries contain kernels only for
the `CMAKE_CUDA_ARCHITECTURES` value they were built with. The default
`NATIVE` value covers the build host's GPU and nothing else. To package an
image for other GPUs, choose the architecture list before the first native
build, as described in
[Choosing CUDA architectures](./from-source.md#choosing-cuda-architectures),
and inspect the recorded value with the commands in
[The architecture contract record](./from-source.md#the-architecture-contract-record).
`docker/build-image.sh` copies the requested and real values of the cuDF record
into `payload-manifest.json` and the requested value into the
`dev.algeon.cuda-architectures` label, and refuses to stage a cuDF install
without a record.

The published image covers the [supported GPU table](#run-the-published-image)
with this list, exported before the first native build:

```bash
export CMAKE_CUDA_ARCHITECTURES="80-real;86-real;90a-real;100f-real;120a-real"
```

## Build the image

From the `algeon` checkout:

```bash
cd "$ALGEON_ROOT"
bash docker/build-image.sh
```

The default image carries the Flight SQL server with cuVS, cuGraph and cuML
SQL. An unset `FEATURES` selects that list; `FEATURES=''` selects no optional
feature and builds the cuDF-only server. Add `nvml` to any list for device
diagnostics:

```bash
FEATURES='' bash docker/build-image.sh
FEATURES=nvml bash docker/build-image.sh
FEATURES=cugraph,cuml,cuvs,nvml bash docker/build-image.sh
```

The tag is `<version>-<cpu baseline>` for the default feature set, such as
`algeon-server:0.1.2-x86-64-v3`. The empty feature list appends `-cudf`, and
any other feature set appends its sorted features joined by `-`. The builds
above produce `algeon-server:0.1.2-x86-64-v3-cudf`,
`algeon-server:0.1.2-x86-64-v3-nvml` and
`algeon-server:0.1.2-x86-64-v3-cugraph-cuml-cuvs-nvml`, and leave the default
image in place. The build record, payload manifest and `dev.algeon.features`
label record the selected list, which is empty for `FEATURES=''`.
`IMAGE_REPOSITORY` replaces `algeon-server`. The image labels record the
version, source revision, whether the worktree had uncommitted changes,
repository URL, creation time, features, CUDA toolkit release, requested CUDA
architectures, CPU baseline, and the two base images. A build from a worktree with uncommitted changes also prints a
`warning:` line:

```bash
docker inspect --format '{{json .Config.Labels}}' algeon-server:0.1.2-x86-64-v3
```

A server built without the `cugraph` feature still installs the cuGraph SQL
frontend and returns a structured `required_feature_disabled` error when
cuGraph execution is requested. Set `ALGEON_SERVER_CUGRAPH_ENABLED=false` to
hide it.

The script always runs its own clean build of `algeon_server`, from an
isolated working directory and a clean `CARGO_HOME`, with the fixed target
triple and `target-cpu=x86-64-v3` from `.config/policies/cpu-packaging.toml`.
It writes a build record (binary hash, policy fingerprint, features, target
triple, codegen flags, version, source revision and CUDA toolkit release)
beside the binary. An ordinary `target/release` build has no such record and
is not reused. A packaging build with `FEATURES=cugraph,cuml,cuvs,nvml`
typically takes 7 to 8 minutes with a warm native prefix and compiler cache,
or 3 minutes when built with `FEATURES=nvml`.

`SKIP_CARGO_BUILD=1` re-stages a payload from a previous packaging build only:

```bash
SKIP_CARGO_BUILD=1 bash docker/build-image.sh
```

It reuses the artifact only when the recorded build record still agrees;
re-staging an existing payload takes under a minute.
`docker/dist/payload-manifest.json` records the staged files, checksums and
policy fingerprint. The script fails before `docker build` when the staged
files or `SHA256SUMS` disagree with that manifest. `CPU_PACKAGE_DIST_DIR`
moves the staging directory, which is also the build context.

What ends up in the image:

| Layer | Source | How it gets in |
| --- | --- | --- |
| cuDF, RMM, and cuGraph `.so` | `target/native/cudf-install`, `target/native/cugraph-install` | bundled by `build-image.sh` |
| cuVS `.so` | `target/native/cuvs-install` | bundled: `libcuvs.so` for any GPU SQL feature, `libcuvs_c.so` for `cuvs` |
| cuML `.so` | `target/native/cuml-install` | bundled for `cuml`: `libcuml.so`, `libcuml_rust.so`, `libtreelite.so`, `libnvforest.so` |
| `libcudf_rust*.so` FFI bridges | cudf-sys CMake `OUT_DIR` | bundled by `build-image.sh` |
| CUDA toolkit (`libcudart`, `libcublas`, `libnvrtc`, and others) | `nvidia/cuda:*-runtime` (`CUDA_RUNTIME_IMAGE`) | harvested in the Dockerfile through `ldd` |
| License texts | native build trees, Cargo registry, `nvidia/cuda:*-runtime` | `/app/licenses`, see [Licenses](#licenses) |
| Base OS | `ubuntu:24.04` (`RUNTIME_BASE_IMAGE`) | runtime stage base |
| `libgomp1`, `libbrotli1`, `tini` | Ubuntu apt | installed in the runtime stage |

The image size depends on the features and the CUDA architecture list, so it
is measured per build with `docker image ls`.

### Licenses

`/app/licenses` holds the license texts of everything the image bundles.
`INVENTORY.tsv` lists each component with its version, declared license,
the staged files that bundle it, and its license files. `FORKS.tsv` names the
repository and the checked-out commit of each staged cuDF, cuVS, cuGraph or
cuML submodule, and links to [Fork changes](https://algeon.dev/reference/forks)
for what the forks change:

| Directory | Contents |
| --- | --- |
| `algeon/` | Algeon `LICENSE` and `NOTICE` |
| `rapids/<layer>/` | `LICENSE` of the cuDF, cuVS, cuGraph or cuML submodule for each staged layer |
| `native/<layer>/<package>/` | license files of each package that CPM or FetchContent fetched into the layer's build tree, except GoogleTest and rapids-cmake, which never reach a staged library |
| `rust/<crate>-<version>/` | license files of each crate that `cargo metadata` shows linked into `algeon_server`; a crate that ships none gets the standard text of its declared license, from `docker/license-texts` or, for Apache-2.0, the repository `LICENSE`; MIT and BSD-3-Clause texts name the crate's `cargo metadata` authors as copyright holders, or `the <crate> authors` and its repository when it lists none |
| `cuda/` | `NGC-DL-CONTAINER-LICENSE` and the copyright file, which holds the CUDA EULA, of each Debian package that ships a harvested CUDA library; `LIBRARIES.tsv` maps each library to its package and license file |

`LIBRARIES.tsv` maps `libnvtx3interop.so.1` to `native/cudf/nvtx3/LICENSE.txt`
instead of its package's copyright file. NVTX is Apache-2.0 WITH
LLVM-exception, as the SPDX line of `nvToolsExt.h` in the same package
states, and the CUDA EULA's list of redistributable files does not name it.
The Dockerfile keeps such exceptions in one commented table in the harvest
stage and fails the build when the package stops declaring that license.

The canonical build captures `cargo metadata` beside the binary and binds its
hash to the build record, so `SKIP_CARGO_BUILD=1` reuses the same crate list.
Staging fails when a staged library belongs to no listed component, a
component has no license source, or a staged submodule has modified tracked
files. The host-staged files are part of
`payload-manifest.json` and `SHA256SUMS`; the harvest stage fails when a
harvested CUDA library has no owning package with a copyright file.

:::warning[Build machine versus target host CPU]
This image targets x86-64-v3. Both the build machine and every target host must
support that instruction set. Otherwise the binary may terminate with `SIGILL`
before `main`. See [CPU requirement](./prerequisites.md#cpu-requirement).
:::

## Run the server

The image starts with the server defaults: GPU 0, authentication disabled,
and a listener on port 50051 inside the container. Publish the port on host
loopback only, and mount the directory that holds your Parquet files:

```bash
docker run --rm --gpus all -p 127.0.0.1:50051:50051 \
  -v "$PWD/data:/data:ro" \
  algeon-server:0.1.2-x86-64-v3
```

The image sets `ALGEON_SERVER_BIND=0.0.0.0:50051` because a loopback listener
inside the container is unreachable through `-p`. The `127.0.0.1:` prefix on
`-p` keeps the unauthenticated endpoint off other host interfaces. Before
publishing it more widely, enable
[authentication](../integration/configuration.mdx#authentication) and
[TLS](../integration/configuration.mdx#tls).

For settings beyond the defaults, mount a TOML file and name it with
`ALGEON_SERVER_CONFIG_FILE`:

```bash
docker run --rm --gpus all -p 127.0.0.1:50051:50051 \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/server.toml:/etc/algeon/server.toml:ro" \
  -e ALGEON_SERVER_CONFIG_FILE=/etc/algeon/server.toml \
  algeon-server:0.1.2-x86-64-v3
```

Arguments after the image name go to `algeon_server`. Validate a mounted
file without starting the server or touching a GPU:

```bash
docker run --rm \
  -v "$PWD/server.toml:/etc/algeon/server.toml:ro" \
  -e ALGEON_SERVER_CONFIG_FILE=/etc/algeon/server.toml \
  algeon-server:0.1.2-x86-64-v3 --check-config
```

It prints the effective configuration with the source of each value and exits
1 on an invalid file. `--version` and `hash-password` work the same way; give
`hash-password` stdin with `docker run -i`. The
[command line](../integration/configuration.mdx#command-line) reference lists
the arguments.

The [configuration reference](../integration/configuration.mdx#complete-example)
lists every key. Any scalar key can also be passed as an `ALGEON_SERVER_*`
variable with `-e`. Because the image's `ALGEON_SERVER_BIND` overrides `bind` in a
mounted file, the container port is set with
`-e ALGEON_SERVER_BIND=0.0.0.0:<port>`. The healthcheck reads that variable.
The image's `ALGEON_SERVER_DATA_ROOT=/data` likewise overrides `[data] root`.
The server runs as UID 1001 in the container; mounted
files must be readable by that user.

The top-level Parquet files and directories under `/data` become tables at
startup, as described under
[Data root](../integration/configuration.mdx#data-root). Other
table locations are paths inside the container too:

```sql
CREATE OR REPLACE EXTERNAL TABLE archived_events
STORED AS PARQUET LOCATION '/data/archive/events.parquet';
```

To register tables at every start, mount a SQL file of DDL statements and
point [`init_sql`](../integration/configuration.mdx#startup-sql) at its
container path.

Startup is eager: CUDA/RMM admission initializes before the Flight SQL port is
opened. The server is ready when logs include:

```text
query admission controller initialized
starting Algeon Flight SQL server
```

The image has a TCP healthcheck for `ALGEON_SERVER_BIND`:

```bash
docker ps --filter ancestor=algeon-server:0.1.2-x86-64-v3
```

On `SIGTERM` the server allows `observability.shutdown_grace_secs`, 30 s by
default, to finalize runtime artifacts. `docker stop` sends `SIGKILL` after
10 s unless told otherwise, so stop the container with `docker stop -t 30`, or
with a timeout longer than a configured `shutdown_grace_secs`.

## Smoke-test Flight SQL

With the container running on `127.0.0.1:50051`, use the repository client.
It needs only the default `tools` features and no native libraries:

```bash
cargo run -p tools -- \
  flight-sql-query grpc://127.0.0.1:50051 "SELECT 1 AS one"
```

Expected output:

```text
one
1
```

For a full-feature image, run PageRank over inline rows:

```bash
cargo run -p tools -- \
  flight-sql-query grpc://127.0.0.1:50051 \
  "SELECT * FROM cugraph_pagerank(edges => (SELECT * FROM (VALUES (1,2),(2,3),(3,1),(3,4)) AS t(src, dst)))"
```

The result has a `vertex` and a `value` column with one row per vertex.

## Publish a release image

Maintainers publish `ghcr.io/algeon-dev/algeon-server:<version>` from a clean
clone of tag `v<version>`, with the release
`CMAKE_CUDA_ARCHITECTURES` list from [GPU architectures](#gpu-architectures)
set for the whole native build and both base images pinned by digest:

```bash
CUDA_RUNTIME_IMAGE='nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:<digest>' \
RUNTIME_BASE_IMAGE='ubuntu:24.04@sha256:<digest>' \
  bash docker/build-image.sh
bash docker/publish-image.sh --dry-run algeon-server:<version>-x86-64-v3
bash docker/publish-image.sh algeon-server:<version>-x86-64-v3
```

`docker/publish-image.sh` neither rebuilds nor logs in. It refuses an image
unless `dev.algeon.worktree-dirty` is `false`, the revision label equals the
commit of tag `v<version>` in the checkout, the features are the shipped
default list, `dev.algeon.cuda-architectures` equals the release list, and
both base image labels carry a digest. It also tags and pushes `:latest` when
`v<version>` is the highest `vX.Y.Z` tag in the checkout by numeric
comparison, ignoring pre-release tags; for an older release it pushes only
`:<version>` and says that `:latest` stays. `--dry-run` runs the same checks
and prints the planned tags; otherwise the script tags and pushes and prints
the pushed digest. `--repository` replaces `ghcr.io/algeon-dev/algeon-server`.

## Troubleshooting

- The container exits at startup with `no NVIDIA driver is available to this
  process`: it was started without `--gpus all`, or the host lacks the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- The host driver predates CUDA 13. Install driver R580 or newer. The symptom
  depends on how the NVIDIA Container Toolkit injects the GPU. With its
  default runtime hook, `docker run` fails before the server starts with
  `unsatisfied condition: cuda>=13.0`, from the image's `NVIDIA_REQUIRE_CUDA`.
  In CDI mode the toolkit ignores `NVIDIA_REQUIRE_*`, so the container starts
  and the server exits with `cudaGetDeviceCount failed: ...; the NVIDIA driver
  supports CUDA X.Y and the CUDA runtime is 13.2`, where X.Y is the driver's
  CUDA version.
- `docker/build-image.sh` cannot find `libcudf.so`, `librmm.so`, or
  `libcugraph.so`: finish the source build first, or set `ALGEON_NATIVE_DIR` to
  the root containing the matching install prefixes.
- Staging fails with `ldd could not resolve libcugraph.so to a canonical
  libcuvs.so` (or `libcuml.so`): the installed library's RUNPATH does not name
  the cuVS prefix it was built against. `scripts/build/build_libcugraph.sh`
  and `build_libcuml.sh` write that RUNPATH at install time. Rebuild the
  library with the current scripts, in the order on
  [Building from Source](./from-source.md#3-build-the-native-stack).
- The container starts but the first GPU query fails with
  `cudaErrorNoKernelImageForDevice`: the native libraries were built without
  the deployment GPU's architecture. Rebuild them as described under
  [GPU architectures](#gpu-architectures).
- The harvest stage fails with `the payload was built with CUDA toolkit`:
  set the `CUDA_RUNTIME_IMAGE` environment variable to an image of the
  toolkit's CUDA major and the same or a newer minor.
- Staging fails with `license staging failed`: the message names the build
  tree, fetched package or crate without a license source, or the staged
  library without a listed component. Keep the native build trees beside their
  install prefixes, and add a standard text to `docker/license-texts` for a
  crate that declares a license there is no text for.
- A plain `docker build` fails with `missing packaging build args`: the image
  needs the verified payload and build arguments that `docker/build-image.sh`
  supplies.
