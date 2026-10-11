# syntax=docker/dockerfile:1

# ===========================================================================
# algeon-datafusion: Arrow Flight SQL service  (GPU query engine)
# ===========================================================================
#
# This image ships a Rust binary that links locally built RAPIDS C++ libraries
# (cuDF / cuGraph / RMM / cuVS / cuML) for this checkout. The
# image bundles the exact libraries the binary was linked against.
# The unmodified CUDA *toolkit* libraries (libcudart, libcublas, libnvrtc,
# ...) are harvested from an official `nvidia/cuda:*-runtime` image instead.
#
# BUILD FLOW (two host-side steps):
#   1. bash docker/stage-docker-payload.sh
#        -> stages docker/dist/{bin/algeon_server, lib/*.so}
#           (native libs only; stripped; RUNPATHs rehomed)
#   2. bash docker/build-image.sh
#
# IMAGE STAGES:
#   cuda-harvest : an nvidia/cuda:-runtime image. The staged binary + native
#                  libs are dropped in, then `ldd` resolves exactly which CUDA
#                  toolkit libraries this binary actually needs and copies just
#                  those out. A cuDF-only binary pulls ~110 MB (cudart+nvrtc);
#                  cuML adds CUDA math dependencies (cuSolver, cuBLAS,
#                  cuBLASLt, cuSparse); the full-feature image measured
#                  3.68 GB on 2026-10-09.
#   runtime      : a minimal nvidia/cuda:-base image + the native libs + the
#                  harvested CUDA subset. No conda, no RAPIDS, no build tools.
# ===========================================================================

# CUDA 13.x. The native libs were built against the CUDA 13.2 toolkit; the
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

# libgomp1 lets the closure check below resolve the OpenMP dependency of the
# staged cuVS, cuGraph and cuML libraries; the runtime stage installs it too.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Drop the staged payload in at its final runtime location so the binary's
# RUNPATH (/app/lib, set by stage-docker-payload.sh) resolves the native libs during ldd.
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
# that live under the CUDA toolkit prefix. The native libs are already present
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
        | awk '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found"')"; \
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

# Runtime system libs the bundled RAPIDS libs need that are NOT in the CUDA
# -base image: libgomp1 (OpenMP, cuVS/cuGraph), libbrotli1 (Arrow/Parquet in
# libcudf). tini = correct PID 1 / signals. The server makes no outbound TLS
# connections, so no CA bundle is installed.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libbrotli1 tini && \
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

# Bundled RAPIDS libs + harvested CUDA toolkit subset, side by side.
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
        | awk '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found"')"; \
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

# Configuration reference: website/docs/integration/configuration.mdx
# (https://algeon.dev/integration/configuration).
# The image sets ALGEON_SERVER_BIND=0.0.0.0:50051 so `docker run -p` reaches the listener.
# Every other setting comes from a mounted TOML file named by ALGEON_SERVER_CONFIG_FILE
# or from ALGEON_SERVER_* variables derived from TOML paths.

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
