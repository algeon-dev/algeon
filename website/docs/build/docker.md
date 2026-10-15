---
sidebar_position: 5
sidebar_label: Docker
description: Build the Algeon Flight SQL server image from locally built native artifacts and smoke-test Flight SQL.
---

# Building with Docker

The Docker image packages the standalone Flight SQL server plus the exact
locally built RAPIDS libraries it was linked against.

:::note[Docker requires a source build]

Complete [Building from Source](./from-source.md) before following this page.
`docker/build-image.sh` packages the locally built native libraries.
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
`docker/validate-loader-closure.sh`, writes `payload-manifest.json` and
`SHA256SUMS`, and runs `docker build` with `docker/dist` as the build context.
Inside the build, the harvest stage checks the manifest and `SHA256SUMS`
against the hashes verified on the host and every other file against
`SHA256SUMS`, rejects a CUDA runtime image older than
the toolkit that built the payload, reruns the closure check, and copies out
only the CUDA toolkit libraries the server links or dlopens. The runtime stage
reruns the closure check on the assembled image.

## Prerequisites

The default full image expects these root-owned artifacts:

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
matching install prefixes under `ALGEON_NATIVE_DIR`.

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
The default runtime images are CUDA 13.2.1. The harvest stage fails when
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
`algeon-server:0.1.0-x86-64-v3`. The empty feature list appends `-cudf`, and
any other feature set appends its sorted features joined by `-`. The builds
above produce `algeon-server:0.1.0-x86-64-v3-cudf`,
`algeon-server:0.1.0-x86-64-v3-nvml` and
`algeon-server:0.1.0-x86-64-v3-cugraph-cuml-cuvs-nvml`, and leave the default
image in place. The build record, payload manifest and `dev.algeon.features`
label record the selected list, which is empty for `FEATURES=''`.
`IMAGE_REPOSITORY` replaces `algeon-server`. The image labels record the
version, source revision, whether the worktree had uncommitted changes,
repository URL, creation time, features, CUDA toolkit release and CPU
baseline. A build from a worktree with uncommitted changes also prints a
`warning:` line:

```bash
docker inspect --format '{{json .Config.Labels}}' algeon-server:0.1.0-x86-64-v3
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
is not reused. On 2026-10-14 the `FEATURES=cugraph,cuml,cuvs,nvml` build took
7 min 20 s end to end, 5 min 21 s of it in Cargo with a warm native prefix and
compiler cache. The `FEATURES=nvml` build took 3 min 03 s.

`SKIP_CARGO_BUILD=1` re-stages a payload from a previous packaging build only:

```bash
SKIP_CARGO_BUILD=1 bash docker/build-image.sh
```

It reuses the artifact only when the recorded build record still agrees; a
reuse run took 41 s.
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
| CUDA toolkit (`libcudart`, `libcublas`, `libnvrtc`, and others) | `nvidia/cuda:*-runtime` | harvested in the Dockerfile through `ldd` |
| `libgomp1`, `libbrotli1`, `tini` | Ubuntu apt | installed in the runtime stage |

On 2026-10-14 `docker images` reported 3.69 GB disk usage and 1.31 GB
content size for the `FEATURES=cugraph,cuml,cuvs,nvml` image, and 1.32 GB and
361 MB for the `FEATURES=nvml` image.

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
  algeon-server:0.1.0-x86-64-v3
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
  algeon-server:0.1.0-x86-64-v3
```

Arguments after the image name go to `algeon_server`. Validate a mounted
file without starting the server or touching a GPU:

```bash
docker run --rm \
  -v "$PWD/server.toml:/etc/algeon/server.toml:ro" \
  -e ALGEON_SERVER_CONFIG_FILE=/etc/algeon/server.toml \
  algeon-server:0.1.0-x86-64-v3 --check-config
```

It prints the effective configuration with the source of each value and exits
1 on an invalid file. `--version` and `hash-password` work the same way; give
`hash-password` stdin with `docker run -i`. The
[command line](../integration/configuration.mdx#command-line) reference lists
the arguments.

The [configuration reference](../integration/configuration.mdx#complete-example)
lists every key. Any scalar key can also be passed as an `ALGEON_SERVER_*`
variable with `-e`. The image's `ALGEON_SERVER_BIND` overrides `bind` in a
mounted file, so change the container port with
`-e ALGEON_SERVER_BIND=0.0.0.0:<port>`. The healthcheck reads that variable.
The server runs as UID 1001 in the container, so mounted
files must be readable by that user.

Table locations are paths inside the container. With the data directory
mounted at `/data`:

```sql
CREATE OR REPLACE EXTERNAL TABLE events
STORED AS PARQUET LOCATION '/data/events.parquet';
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
docker ps --filter ancestor=algeon-server:0.1.0-x86-64-v3
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

## Troubleshooting

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
  set the `CUDA_RUNTIME_IMAGE` and `CUDA_BASE_IMAGE` environment variables to
  images of the toolkit's CUDA major and the same or a newer minor.
- Startup fails with a missing CUDA library such as `libcublas.so.13`: rebuild
  the image with matching `CUDA_RUNTIME_IMAGE` and `CUDA_BASE_IMAGE`, keeping
  both on the same CUDA minor.
- A plain `docker build` fails with `missing packaging build args`: the image
  needs the verified payload and build arguments that `docker/build-image.sh`
  supplies.
