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
POLICY_FILE="$REPO_ROOT/.config/policies/cpu-packaging.toml"
POLICY_HELPER="$REPO_ROOT/scripts/build/cpu_packaging_policy.py"
PROVENANCE="$DIST_DIR/cpu-provenance.json"
LABELS="$DIST_DIR/cpu-provenance.labels"

[[ -f "$PROVENANCE" ]] || {
    echo "ERROR: missing $PROVENANCE; run docker/stage-docker-payload.sh first." >&2
    exit 1
}
[[ -f "$LABELS" ]] || {
    echo "ERROR: missing $LABELS; run docker/stage-docker-payload.sh first." >&2
    exit 1
}

label_value() {
    local key="$1" value
    value=$(awk -F= -v key="$key" '$1 == key { print substr($0, index($0, "=") + 1); found=1 } END { if (!found) exit 1 }' "$LABELS") || {
        echo "ERROR: provenance labels are missing $key." >&2
        exit 2
    }
    [[ -n "$value" ]] || { echo "ERROR: provenance label $key is empty." >&2; exit 2; }
    printf '%s\n' "$value"
}

CPU_BASELINE=$(label_value cpu_baseline)
CPU_IMAGE_ARCH=$(label_value cpu_arch)
CPU_POLICY_PROFILE=$(label_value cpu_profile)
CPU_PROFILE_KIND=$(label_value cpu_profile_kind)
CPU_TARGET_TRIPLE=$(label_value cpu_target)
CPU_BINARY_SHA256=$(label_value binary_sha256)
CPU_PROVENANCE_SHA256=$(sha256sum "$PROVENANCE" | awk '{print $1}')
CPU_IMAGE_TAG_SUFFIX=$(python3 - "$PROVENANCE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as provenance_file:
    provenance = json.load(provenance_file)
try:
    print(provenance["cpu_profile"]["image_tag_suffix"])
except (KeyError, TypeError) as error:
    raise SystemExit(f"ERROR: staged provenance is missing cpu_profile.image_tag_suffix: {error}")
PY
)

python3 "$POLICY_HELPER" --policy "$POLICY_FILE" verify-image-inputs \
    --provenance "$PROVENANCE" \
    --labels "$LABELS" \
    --baseline "$CPU_BASELINE" \
    --arch "$CPU_IMAGE_ARCH" \
    --profile "$CPU_POLICY_PROFILE" \
    --profile-kind "$CPU_PROFILE_KIND" \
    --target "$CPU_TARGET_TRIPLE" \
    --binary-sha256 "$CPU_BINARY_SHA256" \
    --provenance-sha256 "$CPU_PROVENANCE_SHA256" >/dev/null

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
    --build-arg "CPU_PROFILE=$CPU_POLICY_PROFILE"
    --build-arg "CPU_PROFILE_KIND=$CPU_PROFILE_KIND"
    --build-arg "CPU_TARGET=$CPU_TARGET_TRIPLE"
    --build-arg "CPU_BINARY_SHA256=$CPU_BINARY_SHA256"
    --build-arg "CPU_PROVENANCE_SHA256=$CPU_PROVENANCE_SHA256"
)
[[ -n "${CUDA_RUNTIME_IMAGE:-}" ]] && docker_args+=(--build-arg "CUDA_RUNTIME_IMAGE=$CUDA_RUNTIME_IMAGE")
[[ -n "${CUDA_BASE_IMAGE:-}" ]] && docker_args+=(--build-arg "CUDA_BASE_IMAGE=$CUDA_BASE_IMAGE")
docker_args+=(-t "$IMAGE_TAG" "$SCRIPT_DIR")

echo "==> Building image: $IMAGE_TAG"
docker "${docker_args[@]}"
echo "==> SUCCESS: Docker image built: $IMAGE_TAG"
