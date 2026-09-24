# syntax=docker/dockerfile:1

# ===========================================================================
# algeon-datafusion: Arrow Flight SQL service  (GPU query engine)
# ===========================================================================
#
# This image ships a Rust binary that links FORKED RAPIDS C++ libraries
# (cuDF / cuGraph / RMM / cuVS / cuML) built from the pinned monorepo sources. Those
# forked `.so` files are NOT in any public `rapidsai/*` or `nvidia/cuda`
# image, so we bundle the exact libraries the binary was linked against.
# The unmodified CUDA *toolkit* libraries (libcudart, libcublas, libnvrtc,
# ...) are harvested from an official `nvidia/cuda:*-runtime` image instead.
#
# BUILD FLOW (two host-side steps):
#   1. bash docker/stage-docker-payload.sh
#        -> stages docker/dist/{bin/algeon_server, lib/*.so}
#           (forked libs only; stripped; RUNPATHs rehomed)
#   2. bash docker/build-image.sh
#
# IMAGE STAGES:
#   cuda-harvest : an nvidia/cuda:-runtime image. The staged binary + forked
#                  libs are dropped in, then `ldd` resolves exactly which CUDA
#                  toolkit libraries this binary actually needs and copies just
#                  those out. A cuDF-only binary pulls ~110 MB (cudart+nvrtc);
#                  cuML adds CUDA math dependencies (cuSolver, cuBLAS,
#                  cuBLASLt, cuSparse); image size with cuML is not measured.
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
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_MANIFEST_SHA256

# ===========================================================================
# Stage 1: harvest the CUDA toolkit subset this binary actually links
# ===========================================================================
FROM ${CUDA_RUNTIME_IMAGE} AS cuda-harvest
USER root

ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_MANIFEST_SHA256

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils && \
    rm -rf /var/lib/apt/lists/*

# Drop the staged payload in at its final runtime location so the binary's
# RUNPATH (/app/lib, set by stage-docker-payload.sh) resolves the forked libs during ldd.
COPY dist/bin/algeon_server /app/algeon_server
COPY dist/lib/                        /app/lib/
COPY dist/payload-manifest.json       /app/payload-manifest.json

# Refuse an image whose payload, declared policy identity, or hashes do not
# match the clean packaging output staged by docker/stage-docker-payload.sh.
RUN printf '%s  %s\n' "${CPU_BINARY_SHA256}" /app/algeon_server | sha256sum -c - && \
    printf '%s  %s\n' "${CPU_MANIFEST_SHA256}" /app/payload-manifest.json | sha256sum -c - && \
    grep -Fq "\"baseline\": \"${CPU_BASELINE}\"" /app/payload-manifest.json && \
    grep -Fq "\"image_arch\": \"${CPU_ARCH}\"" /app/payload-manifest.json && \
    grep -Fq "\"target_triple\": \"${CPU_TARGET}\"" /app/payload-manifest.json && \
    grep -Fq "\"binary_sha256\": \"${CPU_BINARY_SHA256}\"" /app/payload-manifest.json

# Resolve every staged ELF's dependency closure and copy only the libraries
# that live under the CUDA toolkit prefix. The forked libs are already present
# in /app/lib, so they resolve locally and are (correctly) NOT harvested here.
# glibc ldd renders libnvtx3interop's DT_AUXILIARY [visibility=hidden] marker
# as `visibility=hidden => not found`; it is not a DT_NEEDED dependency.
RUN set -eu; mkdir -p /cuda-collect; \
    closure="$(for elf in /app/algeon_server /app/lib/*.so*; do \
        test -f "$elf" || continue; \
        LD_LIBRARY_PATH=/app/lib ldd "$elf"; \
    done)"; \
    echo "$closure"; \
    test -z "$(echo "$closure" \
        | awk '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found")"; \
    test "$(echo "$closure" | awk '$1 ~ /^librmm\.so/ && $3 ~ /^\// { print $3 }' \
        | xargs -r -n1 readlink -f | sort -u | wc -l)" -eq 1; \
    logger_count="$(echo "$closure" | awk '$1 ~ /^librapids_logger\.so/ && $3 ~ /^\// { print $3 }' \
        | xargs -r -n1 readlink -f | sort -u | wc -l)"; \
    staged_logger_count="$(find /app/lib -maxdepth 1 -name 'librapids_logger.so*' -exec readlink -f {} \; \
        | sort -u | wc -l)"; \
    test "$staged_logger_count" -le 1; \
    if test "$staged_logger_count" -eq 1; then \
        test "$logger_count" -eq 1; \
        resolved_logger="$(echo "$closure" | awk '$1 ~ /^librapids_logger\.so/ && $3 ~ /^\// { print $3; exit }')"; \
        staged_logger="$(find /app/lib -maxdepth 1 -name 'librapids_logger.so*' -print -quit)"; \
        test "$(readlink -f "$resolved_logger")" = "$(readlink -f "$staged_logger")"; \
    else \
        test "$logger_count" -eq 0; \
    fi; \
    if test -e /app/lib/libcuvs.so; then \
        test "$(echo "$closure" | awk '$1 ~ /^libcuvs\.so/ && $3 ~ /^\// { print $3 }' \
            | xargs -r -n1 readlink -f | sort -u | wc -l)" -eq 1; \
        resolved_cuvs="$(echo "$closure" | awk '$1 ~ /^libcuvs\.so/ && $3 ~ /^\// { print $3; exit }')"; \
        test "$(readlink -f "$resolved_cuvs")" = "$(readlink -f /app/lib/libcuvs.so)"; \
    fi; \
    ! readelf -d /app/algeon_server /app/lib/*.so* 2>/dev/null \
        | grep -E 'Shared library: \[libraft\.so'; \
    test -z "$(readelf -d /app/algeon_server /app/lib/*.so* 2>/dev/null \
        | awk -F'[][]' '/\(NEEDED\)/ && $2 ~ /^libcudart\.so/ && $2 != "libcudart.so.13" { print $2 }')"; \
    echo "$closure" \
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
# Stage 2: minimal GPU runtime image
# ===========================================================================
FROM ${CUDA_BASE_IMAGE} AS runtime
USER root

ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_MANIFEST_SHA256

# Runtime system libs the forked RAPIDS libs need that are NOT in the CUDA
# -base image: libgomp1 (OpenMP, cuVS/cuGraph), libbrotli1 (Arrow/Parquet in
# libcudf), ca-certificates (TLS to S3/Glue). tini = correct PID 1 / signals.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libbrotli1 ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# Policy and artifact identity are supplied only by the verified canonical
# packaging inputs verified by docker/build-image.sh.
LABEL dev.algeon.cpu-baseline="${CPU_BASELINE}" \
      dev.algeon.cpu-arch="${CPU_ARCH}" \
      dev.algeon.cpu-target="${CPU_TARGET}" \
      dev.algeon.binary-sha256="${CPU_BINARY_SHA256}" \
      dev.algeon.manifest-sha256="${CPU_MANIFEST_SHA256}"

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --no-create-home ${USERNAME}

WORKDIR /app

# Forked RAPIDS libs (bundled) + harvested CUDA toolkit subset, side by side.
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/lib/      /app/lib/
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /cuda-collect/ /app/lib/
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/algeon_server /app/algeon_server
COPY --from=cuda-harvest --chown=${USERNAME}:${USERNAME} /app/payload-manifest.json /app/payload-manifest.json

# Re-check after assembling the final image: the base-stage loader closure must
# still resolve the staged canonical RMM/cuVS identities with no missing edge.
RUN set -eu; \
    closure="$(for elf in /app/algeon_server /app/lib/*.so*; do \
        test -f "$elf" || continue; \
        LD_LIBRARY_PATH=/app/lib ldd "$elf"; \
    done)"; \
    echo "$closure"; \
    test -z "$(echo "$closure" \
        | awk '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found")"; \
    test "$(echo "$closure" | awk '$1 ~ /^librmm\.so/ && $3 ~ /^\// { print $3 }' \
        | xargs -r -n1 readlink -f | sort -u | wc -l)" -eq 1; \
    logger_count="$(echo "$closure" | awk '$1 ~ /^librapids_logger\.so/ && $3 ~ /^\// { print $3 }' \
        | xargs -r -n1 readlink -f | sort -u | wc -l)"; \
    staged_logger_count="$(find /app/lib -maxdepth 1 -name 'librapids_logger.so*' -exec readlink -f {} \; \
        | sort -u | wc -l)"; \
    test "$staged_logger_count" -le 1; \
    if test "$staged_logger_count" -eq 1; then \
        test "$logger_count" -eq 1; \
        resolved_logger="$(echo "$closure" | awk '$1 ~ /^librapids_logger\.so/ && $3 ~ /^\// { print $3; exit }')"; \
        staged_logger="$(find /app/lib -maxdepth 1 -name 'librapids_logger.so*' -print -quit)"; \
        test "$(readlink -f "$resolved_logger")" = "$(readlink -f "$staged_logger")"; \
    else \
        test "$logger_count" -eq 0; \
    fi; \
    if test -e /app/lib/libcuvs.so; then \
        test "$(echo "$closure" | awk '$1 ~ /^libcuvs\.so/ && $3 ~ /^\// { print $3 }' \
            | xargs -r -n1 readlink -f | sort -u | wc -l)" -eq 1; \
        resolved_cuvs="$(echo "$closure" | awk '$1 ~ /^libcuvs\.so/ && $3 ~ /^\// { print $3; exit }')"; \
        test "$(readlink -f "$resolved_cuvs")" = "$(readlink -f /app/lib/libcuvs.so)"; \
    fi

# RUNPATH already points at /app/lib (binary) and $ORIGIN (libs); keep
# LD_LIBRARY_PATH as a belt-and-suspenders fallback for hosts without patchelf.
ENV LD_LIBRARY_PATH=/app/lib \
    RUST_BACKTRACE=1 \
    ALGEON_SERVER_BIND=0.0.0.0:50051 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ===========================================================================
# Runtime configuration: environment variables read by the server
# (verified against crates/algeon-datafusion/src/server/config/; names are load-bearing)
# ===========================================================================
#
# ── Server core ─────────────────────────────────────────────────────────────
#   ALGEON_SERVER_BIND                  listen addr           (default 0.0.0.0:50051)
#   ALGEON_SERVER_LOG                   tracing filter        (default info)
#   ALGEON_SERVER_GPU_DEVICES           CUDA ordinals, csv    (default 0)
#   ALGEON_SERVER_STATEMENT_TICKET_TTL_SECS                   (default 300)
#   ALGEON_SERVER_CONFIG_FILE           optional TOML overlay ([admission],[native],[observability])
#
# ── Native GPU execution (bounded per-device admission) ─────────────────────
#   ALGEON_SERVER_NATIVE_DEVICE_BUDGET_BYTES          ┐ set all three together,
#   ALGEON_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES       │ or omit all (auto-sized)
#   ALGEON_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK     ┘
#   ALGEON_SERVER_NATIVE_MAX_RETRY_ATTEMPTS
#   ALGEON_SERVER_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS
#   ALGEON_SERVER_NATIVE_MAX_SOURCE_READS_IN_FLIGHT
#   ALGEON_SERVER_NATIVE_SOURCE_CHUNK_PLANNING_MODE   conservative | local_parquet_throughput
#   ALGEON_SERVER_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE
#                                      datafusion_exact_host_welford | gpu_native_tolerant
#
# ── Iceberg catalog (feature `iceberg`; omit all to disable) ────────────────
#   ALGEON_ICEBERG_CATALOG_KIND         glue | rest
#   ALGEON_ICEBERG_CATALOG_NAME  ALGEON_ICEBERG_NAMESPACE  ALGEON_ICEBERG_WAREHOUSE
#   ALGEON_ICEBERG_TABLES               optional "alias=table,table2"
#   ALGEON_ICEBERG_GLUE_CATALOG_ID  ALGEON_ICEBERG_GLUE_URI   (kind=glue)
#   ALGEON_ICEBERG_REST_URI  ALGEON_ICEBERG_REST_PREFIX  ALGEON_ICEBERG_REST_TOKEN  (kind=rest)
#   ALGEON_ICEBERG_S3_ENDPOINT / _REGION / _PATH_STYLE
#   ALGEON_ICEBERG_S3_ACCESS_KEY_ID / _SECRET_ACCESS_KEY / _SESSION_TOKEN  (static creds)
#   ALGEON_ICEBERG_S3_CREDENTIAL_SOURCE=default_chain         (use AWS SDK default chain)
#   ALGEON_LAKEHOUSE_CACHE_ROOT / _READ / _POLICY / _MAX_BYTES
#   ALGEON_ICEBERG_SCAN_PLANNING_TIMEOUT_SECS                (default 60)
#   AWS_DEFAULT_REGION                 REQUIRED for KvikIO S3 (reads ONLY this var;
#                                      AWS_REGION alone -> unsigned reqs -> 403)
#   AWS_REGION  AWS_PROFILE  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
#
# ── DataFusion session (SessionConfig::from_env; single underscores) ────────
#   DATAFUSION_EXECUTION_TARGET_PARTITIONS   DATAFUSION_EXECUTION_BATCH_SIZE
#
# ── cuGraph SQL DSL (image must be built with the cugraph feature) ──────────
#   ALGEON_SERVER_CUGRAPH_ENABLED       true | false
#   ALGEON_SERVER_CUGRAPH_DIRECTED  ALGEON_SERVER_CUGRAPH_RENUMBER
#   ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY   python_cugraph | raw_libcugraph
#
# ===========================================================================

EXPOSE 50051

# Liveness: the gRPC port only opens after CUDA/RMM init succeeds, so a TCP
# connect is a sufficient readiness signal. Generous start-period covers cold
# GPU init + first cuDF load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/${ALGEON_SERVER_BIND##*:}' || exit 1

USER ${USERNAME}

# tini = PID 1 (reaps zombies, forwards SIGTERM for graceful shutdown).
# ulimit -c 0: disable core dumps so a crash exits immediately and surfaces the
# panic on stderr instead of freezing in do_coredump on GPU nodes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "ulimit -c 0 && exec /app/algeon_server"]
