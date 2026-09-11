# GPU query engine container

A drop-and-run Arrow Flight SQL service that executes DataFusion physical plans
on the GPU via cuDF (and optionally cuGraph and cuVS). This directory builds a
self-contained image: you give it a GPU and (optionally) a catalog, it serves
SQL on `:50051`.

## Why this image is unusual

algeon-datafusion links **forked, locally built** RAPIDS C++ libraries — a
modified cuDF, cuGraph, RMM and cuVS. The `.so` files in the public
`rapidsai/*` and `nvidia/cuda` images are the *stock* upstream builds and are
**ABI-incompatible** with this binary. So the image cannot install RAPIDS from a
base layer; it must bundle the exact libraries the binary was linked against.

The split:

| Layer | Source | How it gets in |
|---|---|---|
| Forked cuDF/cuGraph/RMM `.so` | root `target/native/cudf-install` and `target/native/cugraph-install` | **bundled** by `stage-docker-payload.sh` → `dist/lib/` |
| Canonical cuVS `.so` | root `target/native/cuvs-install` | **bundled** by `stage-docker-payload.sh`; `libcuvs.so` for either accelerator and `libcuvs_c.so` for the `cuvs` feature |
| `libcudf_rust*.so` FFI bridges | cudf-sys CMake `OUT_DIR` | **bundled** by `stage-docker-payload.sh` |
| CUDA toolkit (`libcudart`, `libcublas`, `libnvrtc`, …) | official `nvidia/cuda:*-runtime` | **harvested** in-Dockerfile via `ldd` |
| `libgomp1`, `libbrotli1`, `ca-certificates`, `tini` | Ubuntu apt | installed in final stage |

## Build

```bash
# 1. Stage the binary + forked libs (compiles; strips/re-homes RUNPATHs when
#    host tools are available).
bash docker/stage-docker-payload.sh                         # cugraph,cuvs,nvml
FEATURES=nvml bash docker/stage-docker-payload.sh            # lean: cuDF only, ~3x smaller image
FEATURES=cuvs bash docker/stage-docker-payload.sh             # cuVS without cuGraph
SKIP_CARGO_BUILD=1 bash docker/stage-docker-payload.sh       # reuse only hash-verified canonical artifact

# 2. Build the image. This verifies the payload manifest and all file hashes, then
#    supplies the required Docker build inputs automatically.
bash docker/build-image.sh
```

The canonical build performs a controlled clean build that does not reuse ambient
Cargo/C/C++ codegen configuration. `.config/policies/cpu-packaging.toml` supplies
the target triple and exact Rust codegen flags. `scripts/build/cpu_packaging_policy.py`
launches Cargo from an isolated working directory and clean `CARGO_HOME`, and
passes only an allowlisted environment, emitting a canonical build record. The staged
payload includes `payload-manifest.json` with relative paths, file hashes, and the
artifact-affecting policy fingerprint. `SKIP_CARGO_BUILD=1` succeeds only when
the build record and canonical artifact still match the selected policy.

Prerequisites: the forked libs must already be built —
`target/native/cudf-install/lib/libcudf.so`,
`target/native/cuvs-install/lib/libcuvs.so`, and (for cugraph)
`target/native/cugraph-install/lib/libcugraph.so`. Set `ALGEON_NATIVE_DIR` when
the matching `cudf-install`, `cuvs-install`, and `cugraph-install` prefixes live
under a different native root. cuGraph must have been configured against that
root's cuVS prefix. `readelf` and `ldd` are mandatory
closure checks; `patchelf` and `binutils` (strip) on
the host are recommended; without them the image falls back to `LD_LIBRARY_PATH`.

Packaging fails if cuGraph resolves a different or private `_deps` cuVS,
`libcuvs_c.so` is missing for the `cuvs` feature, any dependency is unresolved,
or the final realpath closure contains more than one cuVS/RMM/shared-logger
identity. `librapids_logger.so` is bundled only when a staged ELF has a
`DT_NEEDED` edge; static/absorbed logger builds deliberately omit it.

## Serving cuVS SQL

Use a `cuvs` image for `cuvs_brute_force_knn`, `cuvs_kmeans`, and
`cuvs_pca`. Their relation arguments are parenthesized SQL subqueries, so the
same call shape works over normal tables, Parquet, Iceberg, or cuGraph-derived
relations. The bounded server runs them inside the admitted query domain, whose
allocator enforces the query's memory cap.

## Image size

The cuGraph feature is the dominant cost — it roughly **triples** the image,
mostly via the CUDA math libraries it pulls (`libcublasLt` alone is ~517 MB):

| Build | Forked libs | Harvested CUDA | ≈ Final image |
|---|---|---|---|
| **lean** (`nvml`) | ~180 MB | ~110 MB (cudart + nvrtc) | **~0.6 GB** |
| **full** (`+cugraph,+cuvs`) | ~480 MB | ~1.2 GB (+ cublasLt/cusparse/cusolver/curand/nvJitLink) | **~2 GB** |

When `strip` is available, staged artifacts are stripped
(`strip --strip-unneeded`, ~35–40% off). Set `NO_STRIP=1` to keep symbols for
debugging.

## GPU architecture & platform support

GPU kernels are baked into the forked libs as fatbins. **Coverage differs by
library** because of how each fork was compiled:

- **cuDF** — `libcudf.so` ships cubins for **sm_75, 80, 86, 89, 90, 100, 110,
  120** plus PTX. Portable across Turing → Blackwell, and JIT-able beyond.
- **cuGraph / cuVS** — currently built for **sm_120 (Blackwell) ONLY**, no PTX.
  **cuGraph queries will fail on any non-Blackwell GPU.** This is a build-time
  setting in `cugraph` (`CMAKE_CUDA_ARCHITECTURES`), not something the
  Dockerfile can fix. To make the cuGraph image portable, rebuild the forked
  cuGraph with a multi-arch list before `build.sh`, e.g.:
  ```bash
  CMAKE_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual" \
    bash scripts/build/build_libcugraph.sh
  ```
  (Verify with `cuobjdump --list-elf libcugraph.so | grep -oE 'sm_[0-9]+'`.)
  The lean cuDF-only image has no such restriction.

Other axes:

- **CPU arch**: the bundled libs are **x86_64 only**. arm64 (Grace/Jetson)
  requires rebuilding the forked libs and the binary on arm64; the Dockerfile
  itself is arch-neutral (change the `nvidia/cuda` base tags).

### CPU instruction-set portability (x86-64-v4 default)

This image targets **x86-64-v4**. Confirm that both the build machine and every
deployment host support that ISA; otherwise it can terminate with `SIGILL`
before `main`. Packaging fixes that target and rejects ambient codegen flags so
the staged binary and image labels remain verifiable.

- **CUDA / driver**: built against the CUDA **13.2** toolkit. The harvested
  runtime libs are minor-version forward-compatible within CUDA 13.x. The host
  needs an NVIDIA driver new enough for CUDA 13 (**≥ 580**) and the
  **nvidia-container-toolkit** installed (`--gpus all` / `--runtime=nvidia`).
  Retarget a different CUDA minor via the `CUDA_RUNTIME_IMAGE` /
  `CUDA_BASE_IMAGE` build args (keep both on the same minor).

## Run

```bash
docker run --rm --gpus all -p 50051:50051 algeon-server:x86-64-v4
```

`docker/run.sh` is a fuller launcher that wires the common env vars (catalog,
bounded resource profile, cuGraph) and forwards host AWS credentials. The complete,
code-verified env-var reference lives in the Dockerfile header. It defaults to
the default `:x86-64-v4` tag; set `ALGEON_SERVER_IMAGE` explicitly to use a
different repository or tag. Highlights:

- `ALGEON_SERVER_BIND` (default `0.0.0.0:50051`), `ALGEON_SERVER_LOG`,
  `ALGEON_SERVER_GPU_DEVICES`.
- Native memory: automatic bounded capacity is resolved per selected GPU;
  set the three explicit `ALGEON_SERVER_NATIVE_*` profile knobs together only
  for a fixed experiment.
- Iceberg: `ALGEON_ICEBERG_CATALOG_KIND` (`glue`|`rest`) + `ALGEON_ICEBERG_*`,
  `AWS_DEFAULT_REGION` (**required** for KvikIO S3).
- cuGraph: `ALGEON_SERVER_CUGRAPH_ENABLED=true` requires an image built with
  `cugraph`; set it to `false` for a lean image.

Startup is **eager**: the server initializes CUDA/RMM before binding, so a
missing or incompatible GPU fails fast. Readiness is signalled by the log line
`query admission controller initialized` followed by `starting algeon-datafusion
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
