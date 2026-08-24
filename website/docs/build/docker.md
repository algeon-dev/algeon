---
sidebar_position: 4
sidebar_label: Docker
description: Build the DataFusion Nexus Flight SQL server image from locally built native artifacts and smoke-test Flight SQL.
---

# Building with Docker

The Docker image packages the standalone Flight SQL server plus the exact
forked RAPIDS libraries it was linked against. It is a **packaging step after a
source build**, not a replacement for building the component crates under
`components/` and the root workspace from source.

:::note[Docker requires a source build]

Complete [Building from Source](./from-source.md) before following this page.
`docker/stage-docker-payload.sh` packages the locally built native libraries;
it cannot create their cuDF, cuVS, or cuGraph artifacts itself.

:::

The image build has two phases:

1. `docker/stage-docker-payload.sh` runs on the host. It builds
   `datafusion_nexus_server`, stages the binary, copies the locally built
   forked cuDF/RMM libraries, then adds cuVS and cuGraph libraries for the
   selected features. cuGraph always brings its cuVS runtime dependency.
   It also bundles optional KvikIO/nvCOMP libraries when they are present.
   It strips staged artifacts with `strip --strip-unneeded` when `strip` is
   available; the Docker copies can therefore be smaller than the source
   install, while `NO_STRIP=1` preserves available symbols. It rewrites
   RUNPATHs when `patchelf` is available.
2. `docker/build-image.sh` runs `docker build` after verifying the staged
   provenance. It copies that payload into an
   NVIDIA CUDA runtime image, harvests only the CUDA toolkit libraries the
   server actually links or dlopens, and emits the final runtime image.

## Prerequisites

Complete [Building from Source](./from-source.md) first for the features you
want to package. The default full image expects these root-owned artifacts:

```text
datafusion-nexus/
├── target/native/cudf-install/lib/libcudf.so
├── target/native/cudf-install/lib/librmm.so
├── target/native/cuvs-install/lib/libcuvs.so
└── target/native/cugraph-build/libcugraph.so
```

The Cargo build also produces `libcudf_rust.so` and
`libcudf_rust_rmm_interop.so`; `docker/stage-docker-payload.sh` stages both automatically.

The lean `FEATURES=iceberg,nvml` image omits cuVS and cuGraph, so it
needs neither optional native artifact. Production examples keep `nvml` enabled so live
device telemetry is available without affecting admission decisions.

Install Docker and configure GPU containers with the
[NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The runtime host must satisfy NVIDIA's documented driver and container-runtime
requirements. The full `cugraph` image is portable only to GPU architectures
included when `components/cugraph` was built; see
[Building from Source](./from-source.md#3-build-the-native-stack) when you
need to cover additional deployment GPUs.

## Build the image

From the `datafusion-nexus` checkout:

```bash
cd "$NEXUS_ROOT"

# Full image: Flight SQL server + Iceberg + cuVS/cuGraph SQL + NVML diagnostics.
bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

For a smaller cuDF-only server image, omit both `cugraph` and `cuvs` when
staging the payload:

```bash
FEATURES=iceberg,nvml bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

For this lean image, set `NEXUS_SERVER_CUGRAPH_ENABLED=false` when using direct
`docker run` or before using `docker/run.sh`. A server built without the
`cugraph` feature exposes the frontend but returns a structured
`required_feature_disabled` error when cuGraph execution is requested.

If you already have a current release binary and only need to restage the
runtime payload:

```bash
SKIP_CARGO_BUILD=1 bash docker/stage-docker-payload.sh
bash docker/build-image.sh
```

Skipping compilation does not trust an arbitrary `target/release` binary. It
requires an artifact created by the canonical packaging helper whose binary
hash, policy fingerprint, features, target, and codegen flags still agree.

Docker packaging performs a controlled clean build of the `x86-64-v4` server artifact. The helper
invokes Cargo from an isolated working directory and clean `CARGO_HOME` with
the fixed target triple and `target-cpu=x86-64-v4`, recording a canonical build record.
The resulting `docker/dist/payload-manifest.json` records the staged files, checksums,
and artifact-affecting policy fingerprint. Docker hard-fails if its baseline, target, or hashes
disagree with that manifest. `docker/build-image.sh` verifies the manifest and supplies build arguments automatically.

:::warning[Build machine versus target host CPU]
This image targets x86-64-v4. Both the build machine and every target host must
support that instruction set; otherwise the binary may terminate with `SIGILL`
before `main`.
:::

## Run the server

For the minimal no-catalog server, run the image directly:

```bash
flock /tmp/cudf-gpu.lock \
  docker run --rm --gpus all -p 50051:50051 datafusion-nexus-server:x86-64-v4
```

Startup is eager: CUDA/RMM admission initializes before the Flight SQL port is
opened. The server is ready when logs include:

```text
query admission controller initialized
starting datafusion-nexus Flight SQL server
```

The image has a TCP healthcheck for `NEXUS_SERVER_BIND`:

```bash
docker ps --filter ancestor=datafusion-nexus-server:x86-64-v4
```

### Local fixture launcher

`docker/run.sh` is a convenience launcher for local fixture-style runs. It uses
host networking, enables cuGraph SQL by default, sets bounded native memory
knobs, and wires the local Iceberg REST/RustFS defaults. It defaults to
`datafusion-nexus-server:x86-64-v4`; set `NEXUS_SERVER_IMAGE` explicitly for a
different repository or tag:

```bash
flock /tmp/cudf-gpu.lock bash docker/run.sh
```

Print the exact command without running it:

```bash
bash docker/run.sh --dry-run
```

Use direct `docker run` when you want a minimal server with no Iceberg catalog
environment. Use `docker/run.sh` when you want the local REST catalog defaults
and the cuGraph SQL surface enabled. For a lean image, run:

```bash
NEXUS_SERVER_CUGRAPH_ENABLED=false flock /tmp/cudf-gpu.lock bash docker/run.sh
```

## Smoke-test Flight SQL

With the container running on `127.0.0.1:50051`, use the repo client:

```bash
cargo run -p nexus-tools --all-features -- \
  flight-sql-query http://127.0.0.1:50051 "SELECT 1 AS one"
```

Expected output:

```text
one
1
```

For a full image launched with `NEXUS_SERVER_CUGRAPH_ENABLED=true`:

```bash
cargo run -p nexus-tools --all-features -- \
  flight-sql-query http://127.0.0.1:50051 \
  "SELECT COUNT(*) AS algorithms FROM gpu_list_functions() WHERE provider = 'cugraph'"
```

Expected output starts with:

```text
algorithms
27
```

## Troubleshooting

- `docker/stage-docker-payload.sh` cannot find `libcudf.so`, `librmm.so`, or
  `libcugraph.so`: finish the source build first, or set `CUDF_INSTALL_DIR` /
  `CUGRAPH_CMAKE_BUILD_DIR` to the correct artifact locations.
- The container starts but the first GPU query fails with
  `cudaErrorNoKernelImageForDevice`: rebuild the forked native libraries with
  CUDA architectures covering the deployment GPU.
- Startup fails with a missing CUDA library such as `libcublas.so.13`: rebuild
  the image with matching `CUDA_RUNTIME_IMAGE` and `CUDA_BASE_IMAGE` build
  args, keeping both on the same CUDA minor.
- `docker/run.sh` starts a server with `iceberg_enabled=true`: this is expected
  for that wrapper. Direct `docker run` leaves Iceberg disabled unless you pass
  `NEXUS_ICEBERG_*` variables.
