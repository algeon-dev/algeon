#!/usr/bin/env bash
# Recoverable target/ maintenance for algeon-datafusion.
#
# This is an explicit developer/CI maintenance helper. It never runs as a
# hidden side effect of tests. Cleanup forces a cold next build.
#
# Usage:
#   bash scripts/dev/clean_target_artifacts.sh            # report only
#   bash scripts/dev/clean_target_artifacts.sh --report    # same
#   bash scripts/dev/clean_target_artifacts.sh --mold-only # remove orphaned .mold-* temps
#   bash scripts/dev/clean_target_artifacts.sh --clean-cargo-target
#       # cargo clean (full target wipe; next build is cold)
#   bash scripts/dev/clean_target_artifacts.sh --yes --mold-only
#
# Optional:
#   TARGET_DIR=/path/to/target  (default: $REPO/target)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${TARGET_DIR:-${ROOT}/target}"
MODE="report"
ASSUME_YES=0
MODE_OPTION=""

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report) MODE="report" ;;
    --mold-only|--clean-cargo-target)
      if [[ -n "${MODE_OPTION}" && "${MODE_OPTION}" != "$1" ]]; then
        echo "ERROR: ${MODE_OPTION} conflicts with $1." >&2
        usage >&2
        exit 2
      fi
      MODE_OPTION=$1
      [[ "$1" == "--mold-only" ]] && MODE="mold" || MODE="cargo-clean"
      ;;
    --yes|-y) ASSUME_YES=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

canonicalize_target_dir() {
  local resolved
  resolved=$(realpath -e -- "${TARGET_DIR}") || {
    echo "ERROR: cannot resolve target directory: ${TARGET_DIR}" >&2
    exit 2
  }
  TARGET_DIR=${resolved}
}

validate_destructive_target() {
  # Never allow a workspace or one of its ancestors to become a recursive
  # cleanup root. Custom Cargo target directories are allowed, but they must
  # carry both markers Cargo creates for an artifact root.
  if [[ "${TARGET_DIR}" == "/" || "${TARGET_DIR}" == "${ROOT}" || "${ROOT}/" == "${TARGET_DIR}/"* ]]; then
    echo "ERROR: refusing broad cleanup target that contains the workspace: ${TARGET_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${TARGET_DIR}/.rustc_info.json" || ! -f "${TARGET_DIR}/CACHEDIR.TAG" ]]; then
    echo "ERROR: refusing destructive cleanup without Cargo target markers in ${TARGET_DIR}." >&2
    exit 2
  fi
}

human_size() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null | awk '{print $1}'
  else
    echo "missing"
  fi
}

count_files() {
  local path="$1"
  local pattern="${2:-}"
  if [[ ! -e "${path}" ]]; then
    echo 0
    return
  fi
  if [[ -n "${pattern}" ]]; then
    find "${path}" -name "${pattern}" 2>/dev/null | wc -l | tr -d ' '
  else
    find "${path}" -type f 2>/dev/null | wc -l | tr -d ' '
  fi
}

active_build_pids() {
  # cargo / rustc / mold processes that might own target/.
  ps -eo pid=,args= | awk '
    /[c]argo / || /[r]ustc / || /[m]old / { print $1 " " substr($0, index($0,$2)) }
  ' || true
}

confirm() {
  local prompt="$1"
  if [[ "${ASSUME_YES}" -eq 1 ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    echo "ERROR: refusing destructive action without --yes in non-interactive mode." >&2
    exit 2
  fi
  read -r -p "${prompt} [y/N] " answer
  [[ "${answer}" == "y" || "${answer}" == "Y" ]]
}

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "target directory: ${TARGET_DIR}"
  echo "No target directory present; nothing to report."
  exit 0
fi
canonicalize_target_dir
if [[ "${MODE}" != "report" ]]; then
  validate_destructive_target
fi
echo "target directory: ${TARGET_DIR}"

echo
echo "== Active build processes (must be empty before cleanup) =="
ACTIVE="$(active_build_pids)"
if [[ -n "${ACTIVE}" ]]; then
  printf '%s\n' "${ACTIVE}"
else
  echo "(none)"
fi

echo
echo "== Size by subdir / profile =="
printf '  %-28s %s\n' "target/" "$(human_size "${TARGET_DIR}")"
for sub in debug release tmp check-all fingerprint-matrix; do
  printf '  %-28s %s\n' "target/${sub}" "$(human_size "${TARGET_DIR}/${sub}")"
done
for sub in deps incremental build; do
  printf '  %-28s %s  (files=%s)\n' \
    "target/debug/${sub}" \
    "$(human_size "${TARGET_DIR}/debug/${sub}")" \
    "$(count_files "${TARGET_DIR}/debug/${sub}")"
done
for sub in deps incremental build; do
  if [[ -d "${TARGET_DIR}/release/${sub}" ]]; then
    printf '  %-28s %s  (files=%s)\n' \
      "target/release/${sub}" \
      "$(human_size "${TARGET_DIR}/release/${sub}")" \
      "$(count_files "${TARGET_DIR}/release/${sub}")"
  fi
done

MOLD_COUNT="$(count_files "${TARGET_DIR}" '.mold-*')"
MOLD_SIZE="0"
if [[ "${MOLD_COUNT}" -gt 0 ]]; then
  MOLD_SIZE="$(find "${TARGET_DIR}" -name '.mold-*' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1f GiB", s/1024/1024/1024}')"
fi
echo
echo "== Orphaned mold temporary files =="
echo "  count=${MOLD_COUNT}  approx_size=${MOLD_SIZE}"
if [[ "${MOLD_COUNT}" -gt 0 ]]; then
  # sed consumes the full stream; `head` would close the pipe early and make
  # find exit with SIGPIPE under `set -o pipefail` once more than 20 files exist.
  find "${TARGET_DIR}" -name '.mold-*' -printf '  %p (%k KiB)\n' 2>/dev/null | sed -n '1,20p'
  if [[ "${MOLD_COUNT}" -gt 20 ]]; then
    echo "  ... ($((MOLD_COUNT - 20)) more)"
  fi
fi

echo
echo "Advisory: budget active all-feature surface after a clean rebuild, not the"
echo "stale multi-hash total above. sccache (or equivalent) encrypts clean CI /"
echo "branch-switch compile caching but does not remove multi-binary link cost."

case "${MODE}" in
  report)
    echo
    echo "Report only. Destructive modes:"
    echo "  --mold-only             remove orphaned .mold-* temps"
    echo "  --clean-cargo-target    cargo clean (next build is cold)"
    echo "Add --yes for non-interactive confirmation."
    exit 0
    ;;
  mold|cargo-clean)
    if [[ -n "${ACTIVE}" ]]; then
      echo "ERROR: refuse cleanup while cargo/rustc/mold processes are active." >&2
      exit 1
    fi
    ;;
esac

if [[ "${MODE}" == "mold" ]]; then
  if [[ "${MOLD_COUNT}" -eq 0 ]]; then
    echo "No .mold-* files to remove."
    exit 0
  fi
  confirm "Remove ${MOLD_COUNT} orphaned .mold-* files under ${TARGET_DIR}?" || {
    echo "Aborted."
    exit 1
  }
  find "${TARGET_DIR}" -name '.mold-*' -type f -print -delete
  echo "Removed orphaned mold temps."
  exit 0
fi

if [[ "${MODE}" == "cargo-clean" ]]; then
  confirm "Run cargo clean for ${TARGET_DIR}? The next build will be cold." || {
    echo "Aborted."
    exit 1
  }
  cargo clean --manifest-path "${ROOT}/Cargo.toml" --target-dir "${TARGET_DIR}"
  echo "cargo clean completed."
  exit 0
fi
