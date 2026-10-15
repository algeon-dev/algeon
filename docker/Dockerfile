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
# Build with docker/build-image.sh, which supplies the context and build args.
#
# IMAGE STAGES:
#   cuda-harvest : an nvidia/cuda:-runtime image. It checks the staged
#                  payload against its manifest and checksums, then copies out
#                  only the CUDA toolkit libraries in the binary's `ldd`
#                  closure. A cuDF-only binary pulls 116 MiB (cudart+nvrtc);
#                  cuML adds CUDA math dependencies (cuSolver, cuBLAS,
#                  cuBLASLt, cuSparse).
#   runtime      : a minimal nvidia/cuda:-base image + the native libs + the
#                  harvested CUDA subset. No conda, no RAPIDS, no build tools.
# ===========================================================================

# CUDA 13.x. The harvest stage rejects a runtime image whose CUDA major differs
# from the toolkit that built the payload or whose minor is older. Override
# these to retarget a different CUDA minor / Ubuntu base / pinned digest.
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.2.1-base-ubuntu24.04

# ===========================================================================
# Stage 1: harvest the CUDA toolkit subset this binary actually links
# ===========================================================================
FROM ${CUDA_RUNTIME_IMAGE} AS cuda-harvest
USER root

# libgomp1 lets the closure check below resolve the OpenMP dependency of the
# staged cuVS, cuGraph and cuML libraries; the runtime stage installs it too.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Declaring the build args after the apt layers keeps those layers cached when
# only the payload changes.
ARG CPU_MANIFEST_SHA256
ARG CPU_CHECKSUMS_SHA256
ARG ALGEON_FEATURES
ARG CUDA_TOOLKIT_VERSION

RUN test -n "$CPU_MANIFEST_SHA256" && test -n "$CPU_CHECKSUMS_SHA256" || { \
        echo "ERROR: missing packaging build args; build the image with docker/build-image.sh, which stages and verifies the payload." >&2; \
        exit 1; \
    }

# nvidia/cuda images set CUDA_VERSION. Binaries built with CUDA 13.N need
# runtime libraries from CUDA 13.N or a later 13.x minor.
RUN set -eu; \
    toolkit="$CUDA_TOOLKIT_VERSION"; image="${CUDA_VERSION:-unknown}"; \
    toolkit_major="${toolkit%%.*}"; toolkit_minor="${toolkit#*.}"; toolkit_minor="${toolkit_minor%%.*}"; \
    image_major="${image%%.*}"; image_minor="${image#*.}"; image_minor="${image_minor%%.*}"; \
    if [ "$image_major" != "$toolkit_major" ] || ! [ "$image_minor" -ge "$toolkit_minor" ] 2>/dev/null; then \
        echo "ERROR: the payload was built with CUDA toolkit $toolkit, but the CUDA_RUNTIME_IMAGE build arg selects CUDA $image." >&2; \
        echo "Choose CUDA_RUNTIME_IMAGE and CUDA_BASE_IMAGE with CUDA major $toolkit_major and minor $toolkit_minor or newer." >&2; \
        exit 1; \
    fi

COPY bin/algeon_server /payload/bin/algeon_server
COPY lib/ /payload/lib/
COPY payload-manifest.json SHA256SUMS validate-loader-closure.sh /payload/

# docker/build-image.sh verified the manifest and SHA256SUMS on the host and
# passed in both hashes; SHA256SUMS then covers every other payload file.
RUN cd /payload && \
    printf '%s  %s\n' "$CPU_MANIFEST_SHA256" payload-manifest.json | sha256sum -c - && \
    printf '%s  %s\n' "$CPU_CHECKSUMS_SHA256" SHA256SUMS | sha256sum -c - && \
    sha256sum -c --quiet SHA256SUMS

RUN bash /payload/validate-loader-closure.sh \
        --bin /payload/bin/algeon_server --lib /payload/lib --features "$ALGEON_FEATURES"

# The native libraries resolve from /payload/lib, so only closure entries under
# /usr/local/cuda are harvested.
RUN set -eu; mkdir -p /cuda-collect; \
    for elf in /payload/bin/algeon_server $(find /payload/lib -type f); do \
        LD_LIBRARY_PATH=/payload/lib ldd "$elf"; \
    done \
        | awk '$3 ~ /^\/usr\/local\/cuda/ { print $3 }' \
        | sort -u \
        | xargs -r -I{} cp -Lv {} /cuda-collect/; \
    # cuDF's JIT dlopens NVRTC and its builtins, so they never appear in ldd.
    # Copy them with their symlink chains so any soname dlopen resolves.
    cp -av /usr/local/cuda/lib64/libnvrtc.so* /cuda-collect/; \
    cp -av /usr/local/cuda/lib64/libnvrtc-builtins.so* /cuda-collect/; \
    echo "== harvested CUDA libs ($(du -sh /cuda-collect | cut -f1)) =="; \
    ls -la /cuda-collect

# ===========================================================================
# Stage 2: minimal GPU runtime image
# ===========================================================================
FROM ${CUDA_BASE_IMAGE} AS runtime
USER root

# Runtime system libs the bundled RAPIDS libs need that are NOT in the CUDA
# -base image: libgomp1 (OpenMP, cuVS/cuGraph), libbrotli1 (Arrow/Parquet in
# libcudf). tini = correct PID 1 / signals. The server makes no outbound TLS
# connections, so no CA bundle is installed.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 libbrotli1 tini && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --no-create-home ${USERNAME}

WORKDIR /app

# The files stay root-owned; the runtime user only reads and executes them.
COPY --from=cuda-harvest /payload/lib/                 /app/lib/
COPY --from=cuda-harvest /cuda-collect/                /app/lib/
COPY --from=cuda-harvest /payload/bin/algeon_server    /app/algeon_server
COPY --from=cuda-harvest /payload/payload-manifest.json /app/payload-manifest.json

ARG ALGEON_FEATURES

# The assembled image resolves CUDA libraries from /app/lib instead of the
# runtime image, so the closure check runs again. binutils supplies readelf
# and is purged in the same layer so the image ships without it.
RUN --mount=type=bind,from=cuda-harvest,source=/payload/validate-loader-closure.sh,target=/tmp/validate-loader-closure.sh \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends binutils && \
    bash /tmp/validate-loader-closure.sh \
        --bin /app/algeon_server --lib /app/lib --features "$ALGEON_FEATURES" && \
    apt-get purge -y --auto-remove binutils && \
    rm -rf /var/lib/apt/lists/*

# LD_LIBRARY_PATH covers a payload staged without patchelf, whose RUNPATHs
# still name host paths. The nvidia/cuda base sets
# NVIDIA_DRIVER_CAPABILITIES=compute,utility, and `utility` makes the container
# toolkit mount libnvidia-ml.so.1, which the nvml feature loads.
ENV LD_LIBRARY_PATH=/app/lib \
    RUST_BACKTRACE=1 \
    ALGEON_SERVER_BIND=0.0.0.0:50051

# Configuration reference: website/docs/integration/configuration.mdx
# (https://algeon.dev/integration/configuration).
# The image sets ALGEON_SERVER_BIND=0.0.0.0:50051 so `docker run -p` reaches the listener.
# Every other setting comes from a mounted TOML file named by ALGEON_SERVER_CONFIG_FILE
# or from ALGEON_SERVER_* variables derived from TOML paths.

EXPOSE 50051

# The gRPC port opens only after CUDA and RMM initialization succeed, so a TCP
# connect signals readiness. The start period has to cover CUDA initialization
# on a cold GPU and the first load of the cuDF libraries.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/${ALGEON_SERVER_BIND##*:}' || exit 1

# Provenance comes from the payload manifest that docker/build-image.sh
# verified before passing these build args.
ARG CPU_BASELINE
ARG CPU_ARCH
ARG CPU_TARGET
ARG CPU_BINARY_SHA256
ARG CPU_MANIFEST_SHA256
ARG CPU_CHECKSUMS_SHA256
ARG ALGEON_VERSION
ARG ALGEON_REVISION
ARG ALGEON_WORKTREE_DIRTY
ARG ALGEON_SOURCE
ARG ALGEON_CREATED
ARG CUDA_TOOLKIT_VERSION
LABEL org.opencontainers.image.version="${ALGEON_VERSION}" \
      org.opencontainers.image.revision="${ALGEON_REVISION}" \
      org.opencontainers.image.source="${ALGEON_SOURCE}" \
      org.opencontainers.image.created="${ALGEON_CREATED}" \
      dev.algeon.worktree-dirty="${ALGEON_WORKTREE_DIRTY}" \
      dev.algeon.features="${ALGEON_FEATURES}" \
      dev.algeon.cuda-toolkit="${CUDA_TOOLKIT_VERSION}" \
      dev.algeon.cpu-baseline="${CPU_BASELINE}" \
      dev.algeon.cpu-arch="${CPU_ARCH}" \
      dev.algeon.cpu-target="${CPU_TARGET}" \
      dev.algeon.binary-sha256="${CPU_BINARY_SHA256}" \
      dev.algeon.manifest-sha256="${CPU_MANIFEST_SHA256}" \
      dev.algeon.checksums-sha256="${CPU_CHECKSUMS_SHA256}"

USER ${USERNAME}

# tini = PID 1 (reaps zombies, forwards SIGTERM for graceful shutdown).
# ulimit -c 0: disable core dumps so a crash exits immediately and surfaces the
# panic on stderr instead of freezing in do_coredump on GPU nodes.
# `sh -c` binds the first argument after the script to $0, so "algeon_server"
# fills $0 and container arguments such as --check-config arrive in "$@".
ENTRYPOINT ["/usr/bin/tini", "--", "sh", "-c", "ulimit -c 0 && exec /app/algeon_server \"$@\"", "algeon_server"]
CMD []
