#!/usr/bin/env bash
# Build the Docker image from the validated payload in docker/dist/.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "ERROR: docker/build-image.sh accepts no arguments; configure staging with environment variables." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST_DIR="${CPU_PACKAGE_DIST_DIR:-$SCRIPT_DIR/dist}"
POLICY_HELPER="$REPO_ROOT/scripts/build/cpu_packaging_policy.py"
MANIFEST="$DIST_DIR/payload-manifest.json"

[[ -f "$MANIFEST" ]] || {
    echo "ERROR: missing $MANIFEST; run docker/stage-docker-payload.sh first." >&2
    exit 1
}

eval "$(python3 "$POLICY_HELPER" verify-manifest --manifest "$MANIFEST" --dist-dir "$DIST_DIR")"
CPU_IMAGE_TAG_SUFFIX="$CPU_BASELINE"

command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker is required to build the image." >&2
    exit 1
}

IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-datafusion-nexus-server}"
IMAGE_TAG="$IMAGE_REPOSITORY:$CPU_IMAGE_TAG_SUFFIX"
docker_args=(
    build
    --build-arg "CPU_BASELINE=$CPU_BASELINE"
    --build-arg "CPU_ARCH=$CPU_IMAGE_ARCH"
    --build-arg "CPU_TARGET=$CPU_TARGET_TRIPLE"
    --build-arg "CPU_BINARY_SHA256=$CPU_BINARY_SHA256"
    --build-arg "CPU_MANIFEST_SHA256=$CPU_MANIFEST_SHA256"
)
[[ -n "${CUDA_RUNTIME_IMAGE:-}" ]] && docker_args+=(--build-arg "CUDA_RUNTIME_IMAGE=$CUDA_RUNTIME_IMAGE")
[[ -n "${CUDA_BASE_IMAGE:-}" ]] && docker_args+=(--build-arg "CUDA_BASE_IMAGE=$CUDA_BASE_IMAGE")
docker_args+=(-t "$IMAGE_TAG" "$SCRIPT_DIR")

echo "==> Building image: $IMAGE_TAG"
docker "${docker_args[@]}"
echo "==> SUCCESS: Docker image built: $IMAGE_TAG"
