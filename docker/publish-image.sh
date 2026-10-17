#!/usr/bin/env bash
# Publish a locally built server image as <repository>:<version>, and as
# <repository>:latest only when v<version> is the highest vX.Y.Z tag in this
# checkout, so publishing a fix for an older line never moves latest
# backwards. Nothing is rebuilt: the image must come from
# docker/build-image.sh in a clean checkout of tag v<version>, with the
# shipped features, the release CUDA architecture list and digest-pinned base
# images, as its labels record.
#
# Usage:
#   bash docker/publish-image.sh [--dry-run] [--repository <name>] <image>
#
# --dry-run runs every check and prints the planned tags without tagging or
# pushing. The repository defaults to ghcr.io/algeon-dev/algeon-server; the
# caller must already be logged in to its registry.

set -euo pipefail

# The release native build sets CMAKE_CUDA_ARCHITECTURES to this list; see
# website/docs/build/docker.md for the GPUs it covers.
RELEASE_CUDA_ARCHITECTURES="80-real;86-real;90a-real;100f-real;120a-real"
DEFAULT_REPOSITORY="ghcr.io/algeon-dev/algeon-server"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "usage: $0 [--dry-run] [--repository <name>] <image>" >&2
    exit 2
}

dry_run=0
repository="$DEFAULT_REPOSITORY"
image=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --repository)
            [[ "$#" -ge 2 && -n "$2" ]] || usage
            repository="$2"
            shift
            ;;
        -*) usage ;;
        *)
            [[ -z "$image" ]] || usage
            image="$1"
            ;;
    esac
    shift
done
[[ -n "$image" ]] || usage

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required to publish the image." >&2; exit 1; }

labels_json=$(docker image inspect --format '{{json .Config.Labels}}' "$image") || {
    echo "ERROR: local image $image not found; build it with docker/build-image.sh first." >&2
    exit 1
}
label() {
    python3 -c 'import json, sys; print((json.loads(sys.argv[1]) or {}).get(sys.argv[2], ""))' "$labels_json" "$1"
}

version=$(label org.opencontainers.image.version)
revision=$(label org.opencontainers.image.revision)
dirty=$(label dev.algeon.worktree-dirty)
features=$(label dev.algeon.features)
architectures=$(label dev.algeon.cuda-architectures)
base_image=$(label org.opencontainers.image.base.name)
cuda_runtime_image=$(label dev.algeon.cuda-runtime-image)
shipped_features=$(python3 "$REPO_ROOT/scripts/build/cpu_packaging_policy.py" shipped-features)

problems=()
[[ "$dirty" == "false" ]] \
    || problems+=("dev.algeon.worktree-dirty is '$dirty'; build from a clean checkout of the release tag")
if [[ -z "$version" ]]; then
    problems+=("the image has no org.opencontainers.image.version label")
elif ! tag_commit=$(git -C "$REPO_ROOT" rev-parse --verify --quiet "refs/tags/v$version^{commit}"); then
    problems+=("tag v$version does not exist in $REPO_ROOT")
elif [[ "$revision" != "$tag_commit" ]]; then
    problems+=("org.opencontainers.image.revision is '$revision', but tag v$version is $tag_commit")
fi
[[ "$features" == "$shipped_features" ]] \
    || problems+=("dev.algeon.features is '$features', but the shipped feature list is '$shipped_features'")
[[ "$architectures" == "$RELEASE_CUDA_ARCHITECTURES" ]] \
    || problems+=("dev.algeon.cuda-architectures is '$architectures', but a release needs '$RELEASE_CUDA_ARCHITECTURES'")
[[ "$base_image" == *@sha256:* ]] \
    || problems+=("org.opencontainers.image.base.name '$base_image' is not pinned by digest; set RUNTIME_BASE_IMAGE=<image>@sha256:<digest>")
[[ "$cuda_runtime_image" == *@sha256:* ]] \
    || problems+=("dev.algeon.cuda-runtime-image '$cuda_runtime_image' is not pinned by digest; set CUDA_RUNTIME_IMAGE=<image>@sha256:<digest>")

if (( ${#problems[@]} > 0 )); then
    echo "ERROR: refusing to publish $image:" >&2
    printf '  - %s\n' "${problems[@]}" >&2
    exit 1
fi

# Prints the highest vX.Y.Z tag; tags with a pre-release suffix are not releases.
highest_release=$(git -C "$REPO_ROOT" tag --list 'v*' | python3 -c '
import re, sys
releases = [tuple(map(int, m.groups())) for m in (re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", line.strip()) for line in sys.stdin) if m]
print(".".join(map(str, max(releases))) if releases else "")
')

targets=("$repository:$version")
echo "==> $image passed the release checks (v$version, $revision)"
if [[ "$version" == "$highest_release" ]]; then
    targets+=("$repository:latest")
else
    echo "==> v$version is not the highest release tag (v${highest_release:-none}); $repository:latest stays where it is"
fi
if [[ $dry_run -eq 1 ]]; then
    printf '==> would push %s\n' "${targets[@]}"
    exit 0
fi

for target in "${targets[@]}"; do
    docker tag "$image" "$target"
done
for target in "${targets[@]}"; do
    docker push "$target"
done
digest=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$repository:$version" \
    | awk -v prefix="$repository@" 'index($0, prefix) == 1 { print; exit }')
[[ -n "$digest" ]] || { echo "ERROR: docker reports no $repository digest after the push." >&2; exit 1; }
echo "==> pushed ${targets[*]} as $digest"
