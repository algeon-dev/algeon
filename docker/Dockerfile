# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# datafusion-cudf  —  Arrow Flight SQL service  (GPU query engine)
#
# Build flow:
#   1. Run docker/build.sh on the host to compile the binary and stage:
#        docker/dist/bin/datafusion_cudf_server
#        docker/dist/lib/libcudf_rust.so
#   2. docker build -t datafusion-cudf-server docker/
#
# Stage lib-collector: runs inside the RAPIDS image to collect the full
#   transitive set of /opt/conda .so files needed at runtime via ldd.
#   CUDA system libs (libcudart etc.) come from the runtime base image.
#
# Stage runtime: minimal nvidia/cuda base + binary + collected libs only.
# ---------------------------------------------------------------------------

ARG RAPIDS_IMAGE=rapidsai/base:26.04a-cuda13-py3.12-amd64@sha256:6bbf98bbc36879aa6467ac55f10b5997bdd325be5595156caee00cfa570930d9
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:13.1.0-base-ubuntu24.04@sha256:0f4abb216c6d33bc4932a59d8157f9ad75b818ec17e283a401d9c7ce27efb49f

# ===========================================================================
# Stage 1 — collect runtime libs from the RAPIDS conda environment
# ===========================================================================
FROM ${RAPIDS_IMAGE} AS lib-collector
USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        binutils patchelf && \
    rm -rf /var/lib/apt/lists/*

COPY dist/bin/datafusion_cudf_server /app/datafusion_cudf_server
COPY dist/lib/libcudf_rust.so        /app/lib/libcudf_rust.so

# Rewrite the binary RUNPATH to /app/lib so ldd can resolve libcudf_rust.so.
# The host-compiled binary has a RUNPATH pointing at the Cargo out-dir on the
# build host; that path does not exist in this container.
RUN patchelf --set-rpath /app/lib /app/datafusion_cudf_server

# Collect all /opt/conda-origin .so files in the full transitive closure.
# ldd resolves transitively; we filter to /opt/conda/ since CUDA system libs
# (libcudart, libcuda, ...) are provided by the CUDA_RUNTIME_IMAGE base.
RUN mkdir -p /collect/lib && \
    cp /app/lib/libcudf_rust.so /collect/lib/ && \
    LD_LIBRARY_PATH="/app/lib:/opt/conda/lib:/usr/local/cuda/lib64" \
    ldd /app/datafusion_cudf_server 2>/dev/null | \
    awk '$3 ~ /^\/opt\/conda\// { print $3 }' | sort -u | \
    xargs -r -I{} cp -n {} /collect/lib/

# libnvrtc-builtins is dlopen'd at runtime by libnvrtc; it does not appear
# in ldd output and must be copied explicitly.
RUN find /opt/conda/lib -maxdepth 1 -name 'libnvrtc-builtins.so.*' \
        -exec cp -n {} /collect/lib/ \;

# ===========================================================================
# Stage 2 — minimal GPU runtime image
# ===========================================================================
FROM ${CUDA_RUNTIME_IMAGE}
USER root

ARG USERNAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001

RUN groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --no-create-home ${USERNAME}

WORKDIR /app

COPY --from=lib-collector --chown=${USERNAME}:${USERNAME} /app/datafusion_cudf_server /app/datafusion_cudf_server
COPY --from=lib-collector --chown=${USERNAME}:${USERNAME} /collect/lib/               /app/lib/

ENV LD_LIBRARY_PATH=/app/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
    RUST_BACKTRACE=1

# ===========================================================================
# Environment variable reference
# ===========================================================================
#
# ── Server core ─────────────────────────────────────────────────────────────
#   DATAFUSION_CUDF_SERVER_BIND                   default: 0.0.0.0:50051
#   DATAFUSION_CUDF_SERVER_LOG                    default: info
#   DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT          default: 1
#
# ── Native memory budget (all three required together, or all omitted) ──────
#   DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES
#   DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES
#   DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK
#
# ── Iceberg / AWS Glue (omit all to disable) ────────────────────────────────
#   DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE         required when any Iceberg var set
#   DATAFUSION_CUDF_ICEBERG_WAREHOUSE             required  (s3://...)
#   DATAFUSION_CUDF_ICEBERG_GLUE_TABLES           required  "alias=db.tbl,tbl2"
#   DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME     default: glue
#   DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID       optional
#   AWS_DEFAULT_REGION                            required (KvikIO S3Endpoint reads ONLY this
#                                                 variable; AWS_REGION alone is not enough and
#                                                 will cause silent fallthrough to unsigned S3
#                                                 requests → 403 on private buckets)
#   AWS_REGION                                    used by iceberg-rust catalog only; set both
#   AWS_PROFILE                                   optional
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
#
# ── DataFusion session (DATAFUSION__ prefix, __ separator) ──────────────────
#   DATAFUSION__EXECUTION__TARGET_PARTITIONS
#   DATAFUSION__EXECUTION__BATCH_SIZE
#
# ── Pipeline execution mode ──────────────────────────────────────────────────
#   CUDF_NATIVE_PIPELINE_EXECUTION_MODE           strict_bounded | adaptive
#
# ── cuGraph SQL DSL (binary must be built with cugraph support) ─────────────
#   DATAFUSION_CUDF_SERVER_CUGRAPH_ENABLED        true | false
#   DATAFUSION_CUDF_SERVER_CUGRAPH_ALGORITHMS     optional allowlist by SQL
#                                                 function name, e.g.
#                                                 cugraph_pagerank,cugraph_bfs
#   DATAFUSION_CUDF_SERVER_CUGRAPH_DIRECTED       default graph direction
#   DATAFUSION_CUDF_SERVER_CUGRAPH_RENUMBER       default renumber flag
#   DATAFUSION_CUDF_SERVER_CUGRAPH_CONSTRUCTION_POLICY
#                                                 python_cugraph | raw_libcugraph
#
# ===========================================================================

EXPOSE 50051

USER ${USERNAME}

# ulimit -c 0: disable core dumps. On GPU nodes where the core_pattern pipes
# to systemd-coredump and nothing drains the pipe, a crashing process would
# otherwise freeze in do_coredump. With cores disabled it exits immediately
# and the panic/abort message surfaces on stderr.
CMD ["sh", "-c", "ulimit -c 0 && exec /app/datafusion_cudf_server"]
