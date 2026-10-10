#!/usr/bin/env bash
# Policy test for scripts/ci/check_docker_scripts.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CHECK_SCRIPT="${ROOT}/scripts/ci/check_docker_scripts.sh"

scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

if ! output=$(bash "${CHECK_SCRIPT}" 2>&1); then
  fail "the checked-in docker/ files were rejected: ${output}"
fi

copy_docker_dir() {
  rm -rf "${scratch}/docker"
  mkdir -p "${scratch}/docker"
  cp "${ROOT}/docker/Dockerfile" "${ROOT}"/docker/*.sh "${scratch}/docker/"
}

# Break the last line of a multi-line RUN body that also contains a `#`
# comment line, so the diagnostic must name the instruction's first line.
copy_docker_dir
run_line=$(awk '/^RUN set -eu; mkdir -p \/cuda-collect/ { print NR; exit }' "${scratch}/docker/Dockerfile")
[[ -n "${run_line}" ]] || fail "fixture anchor 'RUN set -eu; mkdir -p /cuda-collect' not found in docker/Dockerfile"
sed -i 's|^    ls -la /cuda-collect$|    ls -la "/cuda-collect|' "${scratch}/docker/Dockerfile"
if output=$(bash "${CHECK_SCRIPT}" "${scratch}/docker" 2>&1); then
  fail "an unterminated quote in a RUN body passed"
elif [[ ${output} != *"Dockerfile:${run_line}: RUN instruction fails sh -n"* ]]; then
  fail "missing Dockerfile:${run_line} diagnostic: ${output}"
fi

copy_docker_dir
printf 'if true; then\n' >>"${scratch}/docker/build-image.sh"
if output=$(bash "${CHECK_SCRIPT}" "${scratch}/docker" 2>&1); then
  fail "a docker/*.sh syntax error passed"
elif [[ ${output} != *"build-image.sh fails bash -n"* ]]; then
  fail "missing build-image.sh diagnostic: ${output}"
fi

if ((failures > 0)); then
  echo "check_docker_scripts tests: ${failures} failure(s)" >&2
  exit 1
fi
echo "check_docker_scripts tests passed"
