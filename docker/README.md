# datafusion-nexus — GPU query engine container

A drop-and-run Arrow Flight SQL service that executes DataFusion physical plans
on the GPU via cuDF (and optionally cuGraph). This directory builds a
self-contained image: you give it a GPU and (optionally) a catalog, it serves
SQL on `:50051`.

## Why this image is unusual

datafusion-nexus links **forked, locally built** RAPIDS C++ libraries — a
modified cuDF, cuGraph, RMM and cuVS. The `.so` files in the public
`rapidsai/*` and `nvidia/cuda` images are the *stock* upstream builds and are
**ABI-incompatible** with this binary. So the image cannot install RAPIDS from a
base layer; it must bundle the exact libraries the binary was linked against.

The split:

| Layer | Source | How it gets in |
|---|---|---|
| Forked cuDF/cuGraph/RMM/cuVS `.so` | local `../cudf-rs`, `../cugraph-rs` builds | **bundled** by `build.sh` → `dist/lib/` |
| `libcudf_rust.so` FFI bridge | cudf-sys CMake `OUT_DIR` | **bundled** by `build.sh` |
| CUDA toolkit (`libcudart`, `libcublas`, `libnvrtc`, …) | official `nvidia/cuda:*-runtime` | **harvested** in-Dockerfile via `ldd` |
| `libgomp1`, `libbrotli1`, `ca-certificates`, `tini` | Ubuntu apt | installed in final stage |

## Build

```bash
# 1. Stage the binary + forked libs (compiles, strips, rehomes RUNPATHs).
bash docker/build.sh                          # full: server,iceberg,cugraph
FEATURES=server,iceberg bash docker/build.sh  # lean: cuDF only, ~3x smaller image
SKIP_CARGO_BUILD=1 bash docker/build.sh        # reuse an existing target/release build

# 2. Build the image (feature-agnostic: bundles whatever build.sh staged,
#    harvests only the CUDA libs the binary actually links).
docker build -t datafusion-nexus-server docker/
```

Prerequisites: the forked libs must already be built —
`../cudf-rs/cudf-install/lib/libcudf.so` and (for cugraph)
`../cugraph-rs/cugraph/cpp/build/libcugraph.so`. Override their locations with
`CUDF_INSTALL_DIR` / `CUGRAPH_BUILD_DIR`. `patchelf` and `binutils` (strip) on
the host are recommended; without them the image falls back to `LD_LIBRARY_PATH`.

## Image size

The cuGraph feature is the dominant cost — it roughly **triples** the image,
mostly via the CUDA math libraries it pulls (`libcublasLt` alone is ~517 MB):

| Build | Forked libs | Harvested CUDA | ≈ Final image |
|---|---|---|---|
| **lean** (`server,iceberg`) | ~180 MB | ~110 MB (cudart + nvrtc) | **~0.6 GB** |
| **full** (`+cugraph`) | ~480 MB | ~1.2 GB (+ cublasLt/cusparse/cusolver/curand/nvJitLink) | **~2 GB** |

All artifacts are stripped (`strip --strip-unneeded`, ~35–40% off). Set
`NO_STRIP=1` to keep symbols for debugging.

## GPU architecture & platform support

GPU kernels are baked into the forked libs as fatbins. **Coverage differs by
library** because of how each fork was compiled:

- **cuDF** — `libcudf.so` ships cubins for **sm_75, 80, 86, 89, 90, 100, 110,
  120** plus PTX. Portable across Turing → Blackwell, and JIT-able beyond.
- **cuGraph / cuVS** — currently built for **sm_120 (Blackwell) ONLY**, no PTX.
  **cuGraph queries will fail on any non-Blackwell GPU.** This is a build-time
  setting in `cugraph-rs` (`CMAKE_CUDA_ARCHITECTURES`), not something the
  Dockerfile can fix. To make the cuGraph image portable, rebuild the forked
  cuGraph with a multi-arch list before `build.sh`, e.g.:
  ```bash
  CMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" \
    bash ../cugraph-rs/scripts/build_libcugraph.sh
  ```
  (Verify with `cuobjdump --list-elf libcugraph.so | grep -oE 'sm_[0-9]+'`.)
  The lean cuDF-only image has no such restriction.

Other axes:

- **CPU arch**: the bundled libs are **x86_64 only**. arm64 (Grace/Jetson)
  requires rebuilding the forked libs and the binary on arm64; the Dockerfile
  itself is arch-neutral (change the `nvidia/cuda` base tags).

### CPU instruction-set portability (x86-64-v2 floor)

The binary must not contain CPU instructions the deploy host lacks, or it dies
with an opaque `SIGILL`. The subtlety: `ahash` (a cuDF dependency) and *all*
Rust/LLVM codegen pick their instruction set at **compile time** from
`target_feature` — there is no runtime SIMD dispatch. The danger is building
with `-C target-cpu=native` / `-march=native` on a modern build host (AVX-512 +
AES-NI), which bakes those into unguarded code.

This repo therefore pins a fixed **x86-64-v2** floor (SSE4.2 + POPCNT, ~2008+,
covers essentially every GPU server):

- **Rust**: `.cargo/config.toml` sets `target-cpu=x86-64-v2`. v2 excludes `aes`,
  so ahash compiles its **portable fallback** hasher. `build.sh` refuses
  host-tuned flags (`native`, `+aes`, `+avx512`). The server also calls
  `ensure_cpu_baseline()` at startup and exits with a clear message (not SIGILL)
  on an older CPU. The image carries `LABEL org.datafusion-nexus.cpu-baseline`.
- **C++ forked libs**: cuDF/cuGraph are built today with empty `CMAKE_CXX_FLAGS`
  (conservative — 0 AES-NI / 0 AVX-512 in `libcudf.so`), so they are already
  portable. Keep it that way: **never pass `-march=native`** to their builds;
  pin `-march=x86-64-v2` if you want the floor to be explicit there too. The
  AES-NI / AVX you *do* see in the binary comes from ring/aws-lc/zstd, which are
  **runtime-dispatched** (CPU-checked before use) and safe on any x86-64.

To trade portability for CPU speed, build a tuned variant locally
(`RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release`) and ship it under a
distinct tag — analogous to per-GPU-arch images. Not worth it for a GPU engine
where CPU SIMD is off the hot path.
- **CUDA / driver**: built against the CUDA **13.0** toolkit. The harvested
  runtime libs are minor-version forward-compatible within CUDA 13.x. The host
  needs an NVIDIA driver new enough for CUDA 13 (**≥ 580**) and the
  **nvidia-container-toolkit** installed (`--gpus all` / `--runtime=nvidia`).
  Retarget a different CUDA minor via the `CUDA_RUNTIME_IMAGE` /
  `CUDA_BASE_IMAGE` build args (keep both on the same minor).

## Run

```bash
docker run --rm --gpus all -p 50051:50051 datafusion-nexus-server
```

`docker/run.sh` is a fuller launcher that wires the common env vars (catalog,
memory policy, cuGraph) and forwards host AWS credentials. The complete,
code-verified env-var reference lives in the Dockerfile header. Highlights:

- `NEXUS_SERVER_BIND` (default `0.0.0.0:50051`), `NEXUS_SERVER_LOG`,
  `NEXUS_SERVER_MAX_CONCURRENT_QUERIES`, `NEXUS_SERVER_GPU_DEVICES`.
- Native memory: `NEXUS_SERVER_NATIVE_MEMORY_POLICY` plus the bounded
  `NEXUS_SERVER_NATIVE_*` knobs.
- Iceberg: `NEXUS_ICEBERG_CATALOG_KIND` (`glue`|`rest`) + `NEXUS_ICEBERG_*`,
  `AWS_DEFAULT_REGION` (**required** for KvikIO S3).
- cuGraph: `NEXUS_SERVER_CUGRAPH_ENABLED=true` (no effect on a lean image).

Startup is **eager**: the server initializes CUDA/RMM before binding, so a
missing or incompatible GPU fails fast. Readiness is signalled by the log line
`query admission controller initialized` followed by `starting datafusion-nexus
Flight SQL server`, after which `:50051` accepts connections. The built-in
`HEALTHCHECK` TCP-probes that port (the port only opens after CUDA init).

## Troubleshooting

- `libcublas.so.13: cannot open shared object file` at startup → the harvest
  stage didn't capture a CUDA lib; confirm the `-runtime` base tag matches the
  binary's CUDA major and rebuild.
- cuGraph queries error with an invalid-device / no-kernel-image message →
  non-Blackwell GPU against the sm_120-only cuGraph build (see above).
- `CUDA error: no kernel image is available for execution` on cuDF → GPU older
  than sm_75; rebuild cuDF including that arch.
