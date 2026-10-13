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
`docker/stage-docker-payload.sh` packages the locally built native libraries.
It cannot create their cuDF, cuVS, cuML, or cuGraph artifacts itself.

:::

Docker staging selects `cugraph`, `cuvs`, `cuml`, and `nvml` by default.
A `FEATURES=cuml` payload also stages the canonical cuVS runtime dependency.

The image build has two phases:

1. `docker/stage-docker-payload.sh` runs on the host. It builds
   `algeon_server`, stages the binary, copies the locally built
   cuDF/RMM libraries, then adds cuVS, cuGraph, and cuML libraries for the
   selected features. cuGraph always brings its cuVS runtime dependency.
   It strips staged artifacts with `strip --strip-unneeded` when `strip` is
   available. The Docker copies can therefore be smaller than the source
   install, while `NO_STRIP=1` preserves available symbols. It rewrites
   RUNPATHs when `patchelf` is available.
2. `docker/build-image.sh` runs `docker build` after verifying the staged
   provenance. It copies that payload into an
   NVIDIA CUDA runtime image, harvests only the CUDA toolkit libraries the
   server actually links or dlopens, and emits the final runtime image.

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
`docker/stage-docker-payload.sh` stages both automatically. Docker consumes the
matching install prefixes under `ALGEON_NATIVE_DIR`.

The lean `FEATURES=nvml` image omits cuVS, cuML, and cuGraph, so it
needs neither optional native artifact. Production examples keep `nvml` enabled so live
device telemetry is available without affecting admission decisions.

`readelf` and `ldd` are required for the loader-closure checks. `patchelf` and
`strip` (binutils) are recommended. Without `patchelf` the image resolves its
libraries through `LD_LIBRARY_PATH=/app/lib`.

Install Docker and configure GPU containers with the
[NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image is built against the CUDA 13.2 toolkit. The harvested runtime
libraries are minor-version compatible within CUDA 13, so the host needs an
NVIDIA driver that supports CUDA 13 (R580 or newer).

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

# Default image: Flight SQL server + cuVS/cuGraph/cuML SQL + NVML diagnostics.
bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

For a smaller cuDF-only server image, omit `cugraph`, `cuvs`, and `cuml` when
staging the payload:

```bash
FEATURES=nvml bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

A server built without the `cugraph` feature still installs the cuGraph SQL
frontend and returns a structured `required_feature_disabled` error when
cuGraph execution is requested. Set `ALGEON_SERVER_CUGRAPH_ENABLED=false` to
hide it.

Staging always runs its own clean build of `algeon_server`, from an isolated
working directory and a clean `CARGO_HOME`, with the fixed target triple and
`target-cpu=x86-64-v4` from `.config/policies/cpu-packaging.toml`. It writes a
build record (binary hash, policy fingerprint, features, target triple and
codegen flags) beside the binary. An ordinary `target/release` build has no
such record and is not reused. Expect about five minutes on a warm native
prefix. It took 5 min 04 s on 2026-10-09.

`SKIP_CARGO_BUILD=1` re-stages a payload from a previous packaging build only:

```bash
SKIP_CARGO_BUILD=1 bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

It reuses the artifact only when the recorded build record still agrees.
`docker/dist/payload-manifest.json` records the staged files, checksums and
policy fingerprint, and `docker/build-image.sh` fails when its baseline,
target or hashes disagree with that manifest.

What ends up in the image:

| Layer | Source | How it gets in |
| --- | --- | --- |
| cuDF, RMM, and cuGraph `.so` | `target/native/cudf-install`, `target/native/cugraph-install` | bundled by `stage-docker-payload.sh` |
| cuVS `.so` | `target/native/cuvs-install` | bundled: `libcuvs.so` for any GPU SQL feature, `libcuvs_c.so` for `cuvs` |
| cuML `.so` | `target/native/cuml-install` | bundled for `cuml`: `libcuml.so`, `libcuml_rust.so`, `libtreelite.so`, `libnvforest.so` |
| `libcudf_rust*.so` FFI bridges | cudf-sys CMake `OUT_DIR` | bundled by `stage-docker-payload.sh` |
| CUDA toolkit (`libcudart`, `libcublas`, `libnvrtc`, and others) | `nvidia/cuda:*-runtime` | harvested in the Dockerfile through `ldd` |
| `libgomp1`, `libbrotli1`, `tini` | Ubuntu apt | installed in the runtime stage |

With the default features, `docker images` reported 3.68 GB disk usage and
1.31 GB content size for the image on 2026-10-09.

:::warning[Build machine versus target host CPU]
This image targets x86-64-v4. Both the build machine and every target host must
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
  algeon-server:x86-64-v4
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
  algeon-server:x86-64-v4
```

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
docker ps --filter ancestor=algeon-server:x86-64-v4
```

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

- `docker/stage-docker-payload.sh` cannot find `libcudf.so`, `librmm.so`, or
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
- Startup fails with a missing CUDA library such as `libcublas.so.13`: rebuild
  the image with matching `CUDA_RUNTIME_IMAGE` and `CUDA_BASE_IMAGE` build
  args, keeping both on the same CUDA minor.
