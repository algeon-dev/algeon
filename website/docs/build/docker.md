---
sidebar_position: 5
sidebar_label: Docker
description: Build the DataFusion Nexus Flight SQL server image from locally built native artifacts, run it with NVIDIA Container Toolkit, and smoke-test Flight SQL.
---

# Building with Docker

The Docker image packages the standalone Flight SQL server plus the exact
forked RAPIDS libraries it was linked against. It is a **packaging step after a
source build**, not a replacement for building `cudf-rs`, `cugraph-rs`, and
`datafusion-nexus` from source.

The image build has two phases:

1. `docker/build.sh` runs on the host. It builds
   `datafusion_nexus_server`, stages the binary, copies the locally built
   forked cuDF/RMM libraries, copies cuGraph/cuVS when the `cugraph` feature is
   enabled, and bundles optional KvikIO/nvCOMP libraries when they are present.
   It strips artifacts when `strip` is available and rewrites RUNPATHs when
   `patchelf` is available.
2. `docker build` runs the Dockerfile. It copies that staged payload into an
   NVIDIA CUDA runtime image, harvests only the CUDA toolkit libraries the
   server actually links or dlopens, and emits the final runtime image.

## Prerequisites

Complete [Building from Source](./from-source.md) first for the features you
want to package. The default full image expects these sibling artifacts:

```text
<NEXUS_ROOT>/
├── cudf-rs/cudf-install/lib/libcudf.so
├── cudf-rs/cudf-install/lib/librmm.so
├── cugraph-rs/cugraph/cpp/build/libcugraph.so
└── datafusion-nexus/
```

The lean `FEATURES=server,iceberg,nvml` image omits cuGraph and does not need
the `cugraph-rs` artifact. Production examples keep `nvml` enabled so live
device telemetry is available without affecting admission decisions.

You also need Docker with NVIDIA Container Toolkit installed on the runtime
host:

```bash
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

The default image uses CUDA 13.2 runtime layers. The host driver must support
CUDA 13, and the forked native libraries must include kernels for the GPU
architecture you deploy on. The full `cugraph` image is currently only portable
to architectures included when `cugraph-rs` was built.

## Build the image

From the `datafusion-nexus` checkout:

```bash
cd "$NEXUS_ROOT/datafusion-nexus"

# Full image: Flight SQL server + Iceberg + cuGraph SQL + NVML diagnostics.
bash docker/build.sh
docker build -t datafusion-nexus-server:latest docker/
```

For a smaller cuDF-only server image, omit `cugraph` when staging the payload:

```bash
FEATURES=server,iceberg,nvml bash docker/build.sh
docker build -t datafusion-nexus-server:latest docker/
```

For this lean image, keep cuGraph disabled at runtime. Use direct `docker run`
or set `NEXUS_SERVER_CUGRAPH_ENABLED=false` before using `docker/run.sh`; a
server built without the `cugraph` feature rejects
`NEXUS_SERVER_CUGRAPH_ENABLED=true`.

If you already have a current release binary and only need to restage the
runtime payload:

```bash
SKIP_CARGO_BUILD=1 bash docker/build.sh
docker build -t datafusion-nexus-server:latest docker/
```

`docker/build.sh` refuses host-tuned CPU codegen flags such as
`target-cpu=native`, `-march=native`, `+aes`, and `+avx512`; redistributable
images use the repository's fixed `x86-64-v2` CPU baseline.

## Run the server

For the minimal no-catalog server, run the image directly:

```bash
flock /tmp/cudf-gpu.lock \
  docker run --rm --gpus all -p 50051:50051 datafusion-nexus-server:latest
```

Startup is eager: CUDA/RMM admission initializes before the Flight SQL port is
opened. The server is ready when logs include:

```text
query admission controller initialized
starting datafusion-nexus Flight SQL server
```

The image has a TCP healthcheck for `NEXUS_SERVER_BIND`:

```bash
docker ps --filter ancestor=datafusion-nexus-server:latest
```

### Local fixture launcher

`docker/run.sh` is a convenience launcher for local fixture-style runs. It uses
host networking, enables cuGraph SQL by default, sets bounded native memory
knobs, and wires the local Iceberg REST/RustFS defaults:

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
cargo run -p datafusion-nexus-tools -- \
  flight-sql-query http://127.0.0.1:50051 "SELECT 1 AS one"
```

Expected output:

```text
one
1
```

For a full image launched with `NEXUS_SERVER_CUGRAPH_ENABLED=true`:

```bash
cargo run -p datafusion-nexus-tools -- \
  flight-sql-query http://127.0.0.1:50051 \
  "SELECT COUNT(*) AS algorithms FROM cugraph_list_algorithms()"
```

Expected output starts with:

```text
algorithms
23
```

## Troubleshooting

- `docker/build.sh` cannot find `libcudf.so`, `librmm.so`, or
  `libcugraph.so`: finish the source build first, or set `CUDF_INSTALL_DIR` /
  `CUGRAPH_BUILD_DIR` to the correct artifact locations.
- The container starts but the first GPU query fails with
  `cudaErrorNoKernelImageForDevice`: rebuild the forked native libraries with
  CUDA architectures covering the deployment GPU.
- Startup fails with a missing CUDA library such as `libcublas.so.13`: rebuild
  the image with matching `CUDA_RUNTIME_IMAGE` and `CUDA_BASE_IMAGE` build
  args, keeping both on the same CUDA minor.
- `docker/run.sh` starts a server with `iceberg_enabled=true`: this is expected
  for that wrapper. Direct `docker run` leaves Iceberg disabled unless you pass
  `NEXUS_ICEBERG_*` variables.
