#!/usr/bin/env bash
# Live Cargo fingerprint matrix for cudf-sys.
#
# Confirms that ambient PATH changes leave cudf-sys fresh while genuine
# compiler, discovery, runtime, and shim-input changes dirty the correct unit.
# Emits Cargo fingerprint reasons (redact PATH values before publishing logs).
#
# Usage:
#   bash scripts/build/check_fingerprint_matrix.sh
#
# Requirements:
#   - CUDF_INSTALL_DIR pointing at a usable libcudf install prefix
#   - CC/CXX set (basenames allowed; absolute preferred)
#   - cargo + a configured CUDA/host toolchain that can build cudf-sys

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -z "${CUDF_INSTALL_DIR:-}" ]]; then
  echo "ERROR: CUDF_INSTALL_DIR must be set to the active libcudf install prefix." >&2
  exit 2
fi
if [[ ! -d "${CUDF_INSTALL_DIR}" ]]; then
  echo "ERROR: CUDF_INSTALL_DIR does not exist: ${CUDF_INSTALL_DIR}" >&2
  exit 2
fi

CC="${CC:-gcc-14}"
CXX="${CXX:-g++-14}"
export CC CXX CUDF_INSTALL_DIR
unset CUDF_CPP_BUILD_DIR || true

LOG_DIR="${CUD_RS_FINGERPRINT_LOG_DIR:-${ROOT}/target/fingerprint-matrix}"
mkdir -p "${LOG_DIR}"
# Build a single lightweight contract harness so the matrix is not a full
# cudf-sys test-link fanout. Fingerprint reasons still cover RunCustomBuild.
CARGO_FINGERPRINT_LOG='cargo::core::compiler::fingerprint=info'
DIRTY_TIMEOUT_SECS="${CUD_RS_FINGERPRINT_DIRTY_TIMEOUT_SECS:-90}"

UNSELECTED_BUILD_CACHE=""
UNSELECTED_BUILD_MTIME_REF=""
SHIM_SOURCE=""
SHIM_MTIME_REF=""
LIBCUDF_PATH=""
LIBCUDF_MTIME_REF=""

restore_probe_mtimes() {
  local status=$?
  set +e
  if [[ -n "${UNSELECTED_BUILD_MTIME_REF}" && -f "${UNSELECTED_BUILD_MTIME_REF}" && -n "${UNSELECTED_BUILD_CACHE}" && -e "${UNSELECTED_BUILD_CACHE}" ]]; then
    touch -r "${UNSELECTED_BUILD_MTIME_REF}" "${UNSELECTED_BUILD_CACHE}"
  fi
  if [[ -n "${SHIM_MTIME_REF}" && -f "${SHIM_MTIME_REF}" && -n "${SHIM_SOURCE}" && -e "${SHIM_SOURCE}" ]]; then
    touch -r "${SHIM_MTIME_REF}" "${SHIM_SOURCE}"
  fi
  if [[ -n "${LIBCUDF_MTIME_REF}" && -f "${LIBCUDF_MTIME_REF}" && -n "${LIBCUDF_PATH}" && -e "${LIBCUDF_PATH}" ]]; then
    touch -r "${LIBCUDF_MTIME_REF}" "${LIBCUDF_PATH}"
  fi
  rm -f "${UNSELECTED_BUILD_MTIME_REF}" "${SHIM_MTIME_REF}" "${LIBCUDF_MTIME_REF}"
  exit "${status}"
}

# Every live probe below changes only mtimes, but interruption or an assertion
# failure must still restore the source tree and selected runtime library.
trap restore_probe_mtimes EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

redact() {
  sed -E 's/(PATH=)[^ ]*/\1<redacted>/g'
}

run_no_run() {
  local label="$1"
  shift
  local allow_timeout=0
  if [[ "${1:-}" == "--allow-timeout" ]]; then
    allow_timeout=1
    shift
  fi
  local log="${LOG_DIR}/${label}.log"
  echo "==> ${label}"

  local status=0
  if [[ "${allow_timeout}" -eq 1 ]]; then
    set +e
    env CARGO_LOG="${CARGO_FINGERPRINT_LOG}" "$@" \
      timeout --signal=INT --kill-after=15s "${DIRTY_TIMEOUT_SECS}" \
      cargo test -p cudf-sys --all-features --test fingerprint_matrix_contract_cpu --no-run \
      >"${log}.raw" 2>&1
    status=$?
    set -e
  else
    set +e
    env CARGO_LOG="${CARGO_FINGERPRINT_LOG}" "$@" \
      cargo test -p cudf-sys --all-features --test fingerprint_matrix_contract_cpu --no-run \
      >"${log}.raw" 2>&1
    status=$?
    set -e
  fi
  redact <"${log}.raw" >"${log}"

  if [[ "${allow_timeout}" -eq 1 && "${status}" -eq 124 ]]; then
    echo "note: ${label}: cargo timed out after ${DIRTY_TIMEOUT_SECS}s (acceptable when dirtiness already recorded)"
  elif [[ "${status}" -ne 0 ]]; then
    if [[ "${allow_timeout}" -eq 1 ]] \
      && rg -q 'fingerprint dirty|dirty: EnvVarChanged|Compiling cudf-sys|Running custom build command for cudf-sys' "${log}"; then
      echo "note: ${label}: cargo exited ${status} after dirtiness was already observed"
    else
      echo "ERROR: ${label}: cargo exited ${status}" >&2
      tail -n 40 "${log}" >&2 || true
      exit "${status}"
    fi
  fi
  rg -n 'Finished|Compiling cudf-sys|Running custom build|fingerprint dirty|dirty:' "${log}" || true
}

assert_no_cudf_sys_compile() {
  local label="$1"
  local log="${LOG_DIR}/${label}.log"
  if rg -q 'Compiling cudf-sys|Running custom build command for cudf-sys|fingerprint dirty' "${log}"; then
    echo "ERROR: ${label}: expected cudf-sys to stay fresh; saw a rebuild:" >&2
    rg -n 'Compiling cudf-sys|Running custom build|dirty:' "${log}" >&2 || true
    exit 1
  fi
  if ! rg -q 'Finished' "${log}"; then
    echo "ERROR: ${label}: missing Finished line" >&2
    exit 1
  fi
  echo "OK: ${label}: cudf-sys remained fresh"
}

assert_cudf_sys_rebuild() {
  local label="$1"
  local log="${LOG_DIR}/${label}.log"
  if ! rg -q 'fingerprint dirty|dirty: EnvVarChanged|Compiling cudf-sys|Running custom build command for cudf-sys' "${log}"; then
    echo "ERROR: ${label}: expected cudf-sys to rebuild; no dirty/compiling evidence found." >&2
    exit 1
  fi
  echo "OK: ${label}: cudf-sys became dirty (see ${log})"
  rg -n 'fingerprint dirty|dirty:|Compiling cudf-sys|Running custom build' "${log}" || true
}

echo "Baseline warm build under install-prefix discovery..."
run_no_run "01-baseline-warm" env CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
run_no_run "02-baseline-repeat" env CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
assert_no_cudf_sys_compile "02-baseline-repeat"

echo "PATH-only change must stay fresh..."
run_no_run "03-path-append" env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}" \
  PATH="/tmp/cudf-irrelevant-bin:${PATH}"
assert_no_cudf_sys_compile "03-path-append"

echo "Simulated venv PATH prefix must stay fresh..."
run_no_run "04-venv-path" env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}" \
  PATH="${ROOT}/.venv/bin:${PATH}"
assert_no_cudf_sys_compile "04-venv-path"

echo "Touch unselected canonical build cache while install-only discovery is selected..."
UNSELECTED_BUILD_CACHE="${ROOT}/target/native/cudf-build/CMakeCache.txt"
if [[ ! -f "${UNSELECTED_BUILD_CACHE}" ]]; then
  echo "SKIP: 05-unselected-build-cache (no ${UNSELECTED_BUILD_CACHE})"
else
  UNSELECTED_BUILD_MTIME_REF="$(mktemp)"
  touch -r "${UNSELECTED_BUILD_CACHE}" "${UNSELECTED_BUILD_MTIME_REF}"
  touch "${UNSELECTED_BUILD_CACHE}"
  run_no_run "05-unselected-build-cache" env \
    CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
  assert_no_cudf_sys_compile "05-unselected-build-cache"
  touch -r "${UNSELECTED_BUILD_MTIME_REF}" "${UNSELECTED_BUILD_CACHE}"
  rm -f "${UNSELECTED_BUILD_MTIME_REF}"
  UNSELECTED_BUILD_MTIME_REF=""
fi

echo "Changing CC must dirty the native build..."
DIRTY_CC="${CC%/}/"
run_no_run "06-change-cc" --allow-timeout env \
  CC="${DIRTY_CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
assert_cudf_sys_rebuild "06-change-cc"

echo "Restoring CC..."
run_no_run "07-restore-cc" --allow-timeout env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"

echo "Changing CUDF_INSTALL_DIR must dirty the native build..."
PROBE_INSTALL="${LOG_DIR}/alternate-install"
mkdir -p "${PROBE_INSTALL}/lib"
if [[ -d "${CUDF_INSTALL_DIR}/lib" ]]; then
  cp -a "${CUDF_INSTALL_DIR}/lib/libcudf.so" "${PROBE_INSTALL}/lib/" 2>/dev/null || true
  cp -a "${CUDF_INSTALL_DIR}/lib/librmm.so" "${PROBE_INSTALL}/lib/" 2>/dev/null || true
fi
run_no_run "08-change-install-dir" --allow-timeout env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${PROBE_INSTALL}"
assert_cudf_sys_rebuild "08-change-install-dir"

run_no_run "09-restore-install-dir" --allow-timeout env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"

echo "Touching a cudf-sys shim source must dirty the native build..."
SHIM_SOURCE="${ROOT}/components/cudf/algeon-cudf-sys/native/src/error.cpp"
if [[ ! -f "${SHIM_SOURCE}" ]]; then
  echo "ERROR: missing shim source ${SHIM_SOURCE}" >&2
  exit 2
fi
SHIM_MTIME_REF="$(mktemp)"
touch -r "${SHIM_SOURCE}" "${SHIM_MTIME_REF}"
touch "${SHIM_SOURCE}"
run_no_run "10-touch-shim-source" --allow-timeout env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
assert_cudf_sys_rebuild "10-touch-shim-source"
touch -r "${SHIM_MTIME_REF}" "${SHIM_SOURCE}"
rm -f "${SHIM_MTIME_REF}"
SHIM_MTIME_REF=""

if [[ -f "${CUDF_INSTALL_DIR}/lib/libcudf.so" ]]; then
  echo "Touching selected libcudf.so must dirty consumers..."
  LIBCUDF_PATH="${CUDF_INSTALL_DIR}/lib/libcudf.so"
  LIBCUDF_MTIME_REF="$(mktemp)"
  touch -r "${LIBCUDF_PATH}" "${LIBCUDF_MTIME_REF}"
  touch "${LIBCUDF_PATH}"
  run_no_run "11-touch-libcudf" --allow-timeout env \
    CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"
  assert_cudf_sys_rebuild "11-touch-libcudf"
  touch -r "${LIBCUDF_MTIME_REF}" "${LIBCUDF_PATH}"
  rm -f "${LIBCUDF_MTIME_REF}"
  LIBCUDF_MTIME_REF=""
else
  echo "SKIP: 11-touch-libcudf (no ${CUDF_INSTALL_DIR}/lib/libcudf.so)"
fi

run_no_run "12-final-restore" --allow-timeout env \
  CC="${CC}" CXX="${CXX}" CUDF_INSTALL_DIR="${CUDF_INSTALL_DIR}"

echo
echo "Fingerprint matrix passed. Redacted logs: ${LOG_DIR}"
echo "Static contracts: cargo test -p cudf-sys --all-features --test fingerprint_matrix_contract_cpu --test build_link_contract_cpu"
