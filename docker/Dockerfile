# syntax=docker/dockerfile:1

# ===========================================================================
# datafusion-nexus  —  Arrow Flight SQL service  (GPU query engine)
# ===========================================================================
#
# This image ships a Rust binary that links FORKED RAPIDS C++ libraries
# (cuDF / cuGraph / RMM / cuVS) built from local sibling checkouts. Those
# forked `.so` files are NOT in any public `rapidsai/*` or `nvidia/cuda`
# image, so we bundle the exact libraries the binary was linked against.
# The unmodified CUDA *toolkit* libraries (libcudart, libcublas, libnvrtc,
# ...) are harvested from an official `nvidia/cuda:*-runtime` image instead.
#
# BUILD FLOW (two host-side steps, then docker build):
#   1. bash docker/build.sh
#        -> stages docker/dist/{bin/datafusion_nexus_server, lib/*.so}
#           (forked libs only; stripped; RUNPATHs rehomed)
#   2. docker build -t datafusion-nexus-server docker/
#
# IMAGE STAGES:
#   cuda-harvest : an nvidia/cuda:-runtime image. The staged binary + forked
#                  libs are dropped in, then `ldd` resolves exactly which CUDA
#                  toolkit libraries this binary actually needs and copies just
#                  those out. A cuDF-only binary pulls ~110 MB (cudart+nvrtc);
#                  a cuGraph binary pulls ~1.2 GB (adds cublasLt/cusparse/...).
#   runtime      : a minimal nvidia/cuda:-base image + the forked libs + the
#                  harvested CUDA subset. No conda, no RAPIDS, no build tools.
# ===========================================================================

# CUDA 13.x. The forked libs were built against the CUDA 13.2 toolkit; the
# harvested runtime libs are minor-version forward-compatible. Override these
# to retarget a different CUDA minor / Ubuntu base / pinned digest.
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.2.1-base-ubuntu24.04
ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_PROFILE
ARG CPU_PROFILE_KIND
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_PROVENANCE_SHA256

# ===========================================================================
# Stage 1 — harvest the CUDA toolkit subset this binary actually links
# ===========================================================================
FROM ${CUDA_RUNTIME_IMAGE} AS cuda-harvest
USER root

ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_PROFILE
ARG CPU_PROFILE_KIND
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_PROVENANCE_SHA256

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils && \
    rm -rf /var/lib/apt/lists/*

# Drop the staged payload in at its final runtime location so the binary's
# RUNPATH (/app/lib, set by build.sh) resolves the forked libs during ldd.
COPY dist/bin/datafusion_nexus_server /app/datafusion_nexus_server
COPY dist/lib/                        /app/lib/
COPY dist/cpu-provenance.json         /app/cpu-provenance.json
COPY dist/cpu-provenance.labels       /app/cpu-provenance.labels

# Refuse an image whose payload, declared policy identity, or hashes do not
# match the clean packaging output staged by docker/build.sh.
RUN printf '%s  %s\n' "${CPU_BINARY_SHA256}" /app/datafusion_nexus_server | sha256sum -c - && \
    printf '%s  %s\n' "${CPU_PROVENANCE_SHA256}" /app/cpu-provenance.json | sha256sum -c - && \
    grep -Fx "cpu_baseline=${CPU_BASELINE}" /app/cpu-provenance.labels && \
    grep -Fx "cpu_arch=${CPU_ARCH}" /app/cpu-provenance.labels && \
    grep -Fx "cpu_profile=${CPU_PROFILE}" /app/cpu-provenance.labels && \
    grep -Fx "cpu_profile_kind=${CPU_PROFILE_KIND}" /app/cpu-provenance.labels && \
    grep -Fx "cpu_target=${CPU_TARGET}" /app/cpu-provenance.labels && \
    grep -Fx "binary_sha256=${CPU_BINARY_SHA256}" /app/cpu-provenance.labels

# Resolve the binary's full dependency closure and copy only the libraries
# that live under the CUDA toolkit prefix. The forked libs are already present
# in /app/lib, so they resolve locally and are (correctly) NOT harvested here.
RUN set -eu; mkdir -p /cuda-collect; \
    LD_LIBRARY_PATH=/app/lib ldd /app/datafusion_nexus_server \
        | awk '$3 ~ /^\/usr\/local\/cuda/ { print $3 }' \
        | sort -u \
        | xargs -r -I{} cp -Lv {} /cuda-collect/; \
    # NVRTC + its builtins are dlopen'd by cuDF's JIT (jitify) and never appear
    # in ldd. Copy them with their symlink chains so any soname dlopen resolves.
    cp -av /usr/local/cuda/lib64/libnvrtc.so*          /cuda-collect/ 2>/dev/null || true; \
    cp -av /usr/local/cuda/lib64/libnvrtc-builtins.so* /cuda-collect/ 2>/dev/null || true; \
    echo "== harvested CUDA libs ($(du -sh /cuda-collect | cut -f1)) =="; \
    ls -la /cuda-collect

# ===========================================================================
# Stage 2 — minimal GPU runtime image
# ===========================================================================
FROM ${CUDA_BASE_IMAGE} AS runtime
USER root

ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_PROFILE
ARG CPU_PROFILE_KIND
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_PROVENANCE_SHA256

# Runtime system libs the forked RAPIDS libs need that are NOT in the CUDA
# -base image: libgomp1 (OpenMP, cuVS/cuGraph), libbrotli1 (Arrow/Parquet in
# libcudf), ca-certificates (TLS to S3/Glue). tini = correct PID 1 / signals.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libbrotli1 ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# Policy and artifact identity are supplied only by the verified canonical
# packaging command printed by docker/build.sh.
LABEL org.datafusion-nexus.cpu-baseline="${CPU_BASELINE}" \
      org.datafusion-nexus.cpu-arch="${CPU_ARCH}" \
      org.datafusion-nexus.cpu-profile="${CPU_PROFILE}" \
      org.datafusion-nexus.cpu-profile-kind="${CPU_PROFILE_KIND}" \
      org.datafusion-nexus.cpu-target="${CPU_TARGET}" \
      org.datafusion-nexus.binary-sha256="${CPU_BINARY_SHA256}" \
      org.datafusion-nexus.provenance-sha256="${CPU_PROVENANCE_SHA256}"

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --no-create-home ${USERNAME}

WORKDIR /app

# Forked RAPIDS libs (bundled) + harvested CUDA toolkit subset, side by side.
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/lib/      /app/lib/
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /cuda-collect/ /app/lib/
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/datafusion_nexus_server /app/datafusion_nexus_server
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/cpu-provenance.json /app/cpu-provenance.json

# RUNPATH already points at /app/lib (binary) and $ORIGIN (libs); keep
# LD_LIBRARY_PATH as a belt-and-suspenders fallback for hosts without patchelf.
ENV LD_LIBRARY_PATH=/app/lib \
    RUST_BACKTRACE=1 \
    NEXUS_SERVER_BIND=0.0.0.0:50051 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ===========================================================================
# Runtime configuration — environment variables read by the server
# (verified against src/server/config.rs; names are load-bearing)
# ===========================================================================
#
# ── Server core ─────────────────────────────────────────────────────────────
#   NEXUS_SERVER_BIND                  listen addr           (default 0.0.0.0:50051)
#   NEXUS_SERVER_LOG                   tracing filter        (default info)
#   NEXUS_SERVER_MAX_CONCURRENT_QUERIES  concurrent stmts    (default 1; preferred)
#   NEXUS_SERVER_MAX_IN_FLIGHT         legacy alias of the above
#   NEXUS_SERVER_GPU_DEVICES           CUDA ordinals, csv    (default 0)
#   NEXUS_SERVER_DEVICE_BUDGET_FRACTION  0<f<=1 of GPU mem for native budgets
#   NEXUS_SERVER_SESSION_PROFILE       native | cpu_fallback
#   NEXUS_SERVER_STATEMENT_TICKET_TTL_SECS                   (default 300)
#   NEXUS_SERVER_CONFIG_FILE           optional TOML overlay ([auth],[native])
#
# ── Native GPU execution (bounded mode) ─────────────────────────────────────
#   NEXUS_SERVER_NATIVE_MEMORY_POLICY                bounded | unbounded
#   NEXUS_SERVER_NATIVE_DEVICE_BUDGET_BYTES          ┐ set all three together,
#   NEXUS_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES       │ or omit all (auto-sized)
#   NEXUS_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK     ┘
#   NEXUS_SERVER_NATIVE_MAX_RETRY_ATTEMPTS
#   NEXUS_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS
#   NEXUS_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT
#   NEXUS_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE   conservative | local_parquet_throughput
#   NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE      strict_bounded | adaptive
#   NEXUS_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE
#                                      datafusion_exact_host_welford | gpu_native_tolerant
#
# ── Iceberg catalog (feature `iceberg`; omit all to disable) ────────────────
#   NEXUS_ICEBERG_CATALOG_KIND         glue | rest
#   NEXUS_ICEBERG_CATALOG_NAME  NEXUS_ICEBERG_NAMESPACE  NEXUS_ICEBERG_WAREHOUSE
#   NEXUS_ICEBERG_TABLES               optional "alias=table,table2"
#   NEXUS_ICEBERG_GLUE_CATALOG_ID  NEXUS_ICEBERG_GLUE_URI   (kind=glue)
#   NEXUS_ICEBERG_REST_URI  NEXUS_ICEBERG_REST_PREFIX  NEXUS_ICEBERG_REST_TOKEN  (kind=rest)
#   NEXUS_ICEBERG_S3_ENDPOINT / _REGION / _PATH_STYLE
#   NEXUS_ICEBERG_S3_ACCESS_KEY_ID / _SECRET_ACCESS_KEY / _SESSION_TOKEN  (static creds)
#   NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE=default_chain         (use AWS SDK default chain)
#   NEXUS_LAKEHOUSE_CACHE_ROOT / _READ / _POLICY / _MAX_BYTES
#   NEXUS_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS                (default 60)
#   AWS_DEFAULT_REGION                 REQUIRED for KvikIO S3 (reads ONLY this var;
#                                      AWS_REGION alone -> unsigned reqs -> 403)
#   AWS_REGION  AWS_PROFILE  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
#
# ── DataFusion session (SessionConfig::from_env; single underscores) ────────
#   DATAFUSION_EXECUTION_TARGET_PARTITIONS   DATAFUSION_EXECUTION_BATCH_SIZE
#
# ── cuGraph SQL DSL (image must be built with the cugraph feature) ──────────
#   NEXUS_SERVER_CUGRAPH_ENABLED       true | false
#   NEXUS_SERVER_CUGRAPH_DIRECTED  NEXUS_SERVER_CUGRAPH_RENUMBER
#   NEXUS_SERVER_CUGRAPH_CONSTRUCTION_POLICY   python_cugraph | raw_libcugraph
#
# NOTE: NEXUS_NATIVE_PIPELINE_EXECUTION_MODE is a BENCHMARK env var only; the
# server uses NEXUS_SERVER_NATIVE_PIPELINE_EXECUTION_MODE.
# ===========================================================================

EXPOSE 50051

# Liveness: the gRPC port only opens after CUDA/RMM init succeeds, so a TCP
# connect is a sufficient readiness signal. Generous start-period covers cold
# GPU init + first cuDF load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/${NEXUS_SERVER_BIND##*:}' || exit 1

USER ${USERNAME}

# tini = PID 1 (reaps zombies, forwards SIGTERM for graceful shutdown).
# ulimit -c 0: disable core dumps so a crash exits immediately and surfaces the
# panic on stderr instead of freezing in do_coredump on GPU nodes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "ulimit -c 0 && exec /app/datafusion_nexus_server"]
