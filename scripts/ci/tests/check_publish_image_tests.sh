#!/usr/bin/env bash
# Refusal and push paths of docker/publish-image.sh, driven through fake
# `docker` and `git` commands. No registry or image is touched.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PUBLISH="${ROOT}/docker/publish-image.sh"
RELEASE_ARCHITECTURES='80-real;86-real;90a-real;100f-real;120a-real'
TAG_COMMIT=1111111111111111111111111111111111111111
SHIPPED=$(python3 "${ROOT}/scripts/build/cpu_packaging_policy.py" shipped-features)

scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
mkdir -p "${scratch}/bin"
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

cat >"${scratch}/bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${FAKE_DOCKER_LOG}"
case "$1" in
  image)
    if [[ "$*" == *".Config.Labels"* ]]; then
      [[ -f "${FAKE_LABELS}" ]] || { echo "Error: No such image" >&2; exit 1; }
      cat "${FAKE_LABELS}"
    else
      echo "registry.example/other@sha256:0000"
      echo "${FAKE_REPOSITORY}@sha256:abcd"
    fi
    ;;
  tag | push) ;;
  *) exit 64 ;;
esac
SH
cat >"${scratch}/bin/git" <<'SH'
#!/usr/bin/env bash
if [[ "$*" == *"tag --list v*"* ]]; then
  printf '%s\n' ${FAKE_TAGS}
  exit 0
fi
if [[ "$*" == *"rev-parse --verify --quiet refs/tags/v0.1.2^{commit}"* ]]; then
  echo "${FAKE_TAG_COMMIT}"
  exit 0
fi
exit 1
SH
chmod +x "${scratch}/bin/docker" "${scratch}/bin/git"

# write_labels <json overrides> : release-ready labels with the given fields replaced.
write_labels() {
  python3 - "${scratch}/labels.json" "$1" "${TAG_COMMIT}" "${SHIPPED}" "${RELEASE_ARCHITECTURES}" <<'PY'
import json
import sys

path, overrides, commit, features, architectures = sys.argv[1:6]
labels = {
    "org.opencontainers.image.version": "0.1.2",
    "org.opencontainers.image.revision": commit,
    "org.opencontainers.image.base.name": "ubuntu:24.04@sha256:" + "1" * 64,
    "dev.algeon.cuda-runtime-image": "nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:" + "2" * 64,
    "dev.algeon.worktree-dirty": "false",
    "dev.algeon.features": features,
    "dev.algeon.cuda-architectures": architectures,
}
labels.update(json.loads(overrides))
open(path, "w").write(json.dumps(labels))
PY
}

# publish <expected status> <arguments...> : run the script, keep its output and docker log.
publish() {
  local expected=$1 status=0
  shift
  : >"${scratch}/docker.log"
  FAKE_DOCKER_LOG="${scratch}/docker.log" FAKE_LABELS="${scratch}/labels.json" \
    FAKE_REPOSITORY="${FAKE_REPOSITORY:-ghcr.io/algeon-dev/algeon-server}" FAKE_TAG_COMMIT="${TAG_COMMIT}" \
    FAKE_TAGS="${FAKE_TAGS:-v0.1.1 v0.1.2 v0.2.0-rc.1}" \
    PATH="${scratch}/bin:${PATH}" bash "${PUBLISH}" "$@" >"${scratch}/output" 2>&1 || status=$?
  [[ "${status}" -eq "${expected}" ]] || fail "publish $* exited ${status}, expected ${expected}: $(cat "${scratch}/output")"
}

pushed() {
  grep -Eq '^(tag|push) ' "${scratch}/docker.log"
}

# refuse <label overrides> <expected diagnostic>
refuse() {
  write_labels "$1"
  publish 1 local/algeon-server:candidate
  grep -Fq -- "$2" "${scratch}/output" || fail "missing refusal '$2' for $1: $(cat "${scratch}/output")"
  ! pushed || fail "a refused image was tagged or pushed for $1"
}

refuse '{"dev.algeon.worktree-dirty": "true"}' "dev.algeon.worktree-dirty is 'true'"
refuse '{"org.opencontainers.image.version": "0.1.3"}' "tag v0.1.3 does not exist"
refuse '{"org.opencontainers.image.revision": "2222222222222222222222222222222222222222"}' \
  "but tag v0.1.2 is ${TAG_COMMIT}"
refuse '{"dev.algeon.features": "cugraph,cuml,cuvs,nvml"}' "but the shipped feature list is '${SHIPPED}'"
refuse '{"dev.algeon.cuda-architectures": "NATIVE"}' "but a release needs '${RELEASE_ARCHITECTURES}'"
refuse '{"org.opencontainers.image.base.name": "ubuntu:24.04"}' "is not pinned by digest; set RUNTIME_BASE_IMAGE"
refuse '{"dev.algeon.cuda-runtime-image": "nvidia/cuda:13.2.1-runtime-ubuntu24.04"}' \
  "is not pinned by digest; set CUDA_RUNTIME_IMAGE"

rm -f "${scratch}/labels.json"
publish 1 local/algeon-server:missing
grep -Fq 'local image local/algeon-server:missing not found' "${scratch}/output" \
  || fail "missing diagnostic for an absent image: $(cat "${scratch}/output")"

publish 2 --repository
publish 2 one two

write_labels '{}'
publish 0 --dry-run local/algeon-server:candidate
for target in ghcr.io/algeon-dev/algeon-server:0.1.2 ghcr.io/algeon-dev/algeon-server:latest; do
  grep -Fxq "==> would push ${target}" "${scratch}/output" || fail "dry run did not plan ${target}"
done
! pushed || fail "a dry run tagged or pushed"

FAKE_REPOSITORY=registry.example/team/algeon publish 0 --repository registry.example/team/algeon local/algeon-server:candidate
for command in \
  'tag local/algeon-server:candidate registry.example/team/algeon:0.1.2' \
  'tag local/algeon-server:candidate registry.example/team/algeon:latest' \
  'push registry.example/team/algeon:0.1.2' \
  'push registry.example/team/algeon:latest'; do
  grep -Fxq -- "${command}" "${scratch}/docker.log" || fail "publish did not run docker ${command}"
done
grep -Fq '==> pushed registry.example/team/algeon:0.1.2 registry.example/team/algeon:latest as registry.example/team/algeon@sha256:abcd' \
  "${scratch}/output" || fail "publish did not report the pushed digest: $(cat "${scratch}/output")"

# v0.1.10 is a higher release than v0.1.2 by number though not by string, so
# publishing 0.1.2 must leave latest alone; the pre-release v0.2.0-rc.1 above
# never counts as a release.
FAKE_TAGS='v0.1.2 v0.1.10 v0.2.0-rc.1' publish 0 --dry-run local/algeon-server:candidate
grep -Fxq '==> would push ghcr.io/algeon-dev/algeon-server:0.1.2' "${scratch}/output" \
  || fail "dry run of an older release did not plan its version tag"
! grep -Fq 'would push ghcr.io/algeon-dev/algeon-server:latest' "${scratch}/output" || fail "dry run of an older release planned latest"
grep -Fq 'v0.1.2 is not the highest release tag (v0.1.10)' "${scratch}/output" \
  || fail "dry run did not explain why latest stays: $(cat "${scratch}/output")"
FAKE_TAGS='v0.1.2 v0.1.10' publish 0 local/algeon-server:candidate
grep -Fxq 'push ghcr.io/algeon-dev/algeon-server:0.1.2' "${scratch}/docker.log" || fail "older release was not pushed"
! grep -Fq ':latest' "${scratch}/docker.log" || fail "publishing an older release moved latest"

if ((failures > 0)); then
  echo "check_publish_image tests: ${failures} failure(s)" >&2
  exit 1
fi
echo "check_publish_image tests passed"
