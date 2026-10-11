#!/usr/bin/env bash
# Policy check: the Docker packaging scripts and every Dockerfile RUN body parse.
# A full `docker build` needs a staged payload and a Docker daemon, so the gate
# checks syntax only: `bash -n` on docker/*.sh and `sh -n` on each RUN body,
# because the image's /bin/sh executes RUN in shell form.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DOCKER_DIR=${1:-${ROOT}/docker}
DOCKERFILE="${DOCKER_DIR}/Dockerfile"

status=0
for script in "${DOCKER_DIR}"/*.sh; do
  bash -n "${script}" || { echo "ERROR: ${script#"${ROOT}/"} fails bash -n" >&2; status=1; }
done

# Join instructions the way BuildKit does: a trailing backslash continues the
# line and is removed, and lines whose first non-blank character is `#` are
# comments even inside a continued instruction. Each RUN body is emitted as
# "<start line>\t<body>\0".
extract_run_bodies() {
  awk '
    function finish() {
      if (buffer ~ /^[ \t]*[Rr][Uu][Nn][ \t]/) {
        sub(/^[ \t]*[Rr][Uu][Nn][ \t]+/, "", buffer)
        printf "%d\t%s%c", start, buffer, 0
      }
      continued = 0
      buffer = ""
    }
    /^[ \t]*#/ { next }
    /^[ \t]*$/ { next }
    {
      line = $0
      if (!continued) { start = NR; buffer = "" }
      if (line ~ /\\[ \t]*$/) {
        sub(/\\[ \t]*$/, "", line)
        buffer = buffer line
        continued = 1
      } else {
        buffer = buffer line
        finish()
      }
    }
    END { if (continued) finish() }
  ' "$1"
}

run_count=0
while IFS=$'\t' read -r -d '' start body; do
  run_count=$((run_count + 1))
  if ! output=$(printf '%s\n' "${body}" | sh -n 2>&1); then
    echo "ERROR: ${DOCKERFILE#"${ROOT}/"}:${start}: RUN instruction fails sh -n: ${output}" >&2
    status=1
  fi
done < <(extract_run_bodies "${DOCKERFILE}")

if ((run_count == 0)); then
  echo "ERROR: ${DOCKERFILE#"${ROOT}/"} has no RUN instruction to check" >&2
  status=1
fi

if ((status == 0)); then
  echo "Docker script policy: docker/*.sh and ${run_count} Dockerfile RUN bodies parse"
fi
exit "${status}"
