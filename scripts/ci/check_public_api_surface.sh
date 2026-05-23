#!/usr/bin/env bash
# Fail closed if any tracked crate's public API surface drifts from its
# checked-in snapshot without an explicit, reviewed snapshot update.
#
# This replaces three `trybuild`-based compile_fail/pass fixture suites
# (tests/datafusion_public_api_boundary.rs,
# crates/nexus-query-engine/tests/public_api_boundary.rs,
# crates/nexus-bench/tests/bench_public_api_boundary.rs), which built a
# complete second copy of the dependency graph per binary (trybuild 1.0.118
# hardcodes its own CARGO_TARGET_DIR; see its src/cargo.rs). `cargo public-api`
# instead runs `cargo doc --output-format=json` in the workspace's ordinary
# target directory, reusing already-built dependency metadata.
#
# This check is strictly stronger than the fixtures it replaces in one
# respect: it fails on *any* unreviewed public API change, including
# unintended additions, not just the specific forbidden items the old
# hand-written cases enumerated. Reviewing and committing a snapshot update
# (this script's `write` mode) is the deliberate approval step for an
# intentional API change.
#
# `-sss` (--omit blanket-impls,auto-trait-impls,auto-derived-impls) keeps the
# snapshot to item-level signatures. None of the migrated fixture cases
# depended on blanket/auto-trait/derived-impl output: every one of them
# reduced to a plain rustc E0599/E0432/E0603/E0624/E0560/E0609 (missing or
# private item, method, field, or module) when reproduced without trybuild.
#
# `cargo public-api` has no `--toolchain` flag by design (it only supports
# `cargo +toolchain public-api`); left to itself it resolves whatever `nightly`
# currently means to rustup (see cargo-public-api's resolve_toolchain), a
# floating channel, same as this repo's `cargo +nightly fmt`. rustdoc JSON's
# content (not just its stable format-version number) can shift across
# nightlies. Without recording that identity, a routine nightly bump presents
# as an unreviewable wall of "removed"/"added" API lines that looks like a
# real regression. TOOLCHAIN.txt below pins the exact rustc build and
# cargo-public-api version the snapshots were generated with, and `check`
# compares identity *before* diffing content, so a toolchain drift is reported
# as exactly that, distinctly from an actual API change.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

# Overridable so the fault-injection self-test
# (scripts/ci/tests/check_public_api_surface_tests.sh) can point this at a
# throwaway directory instead of mutating the real checked-in snapshots.
SNAPSHOT_DIR=${PUBLIC_API_SNAPSHOT_DIR:-.config/policies/public-api}
TOOLCHAIN_FILE="${SNAPSHOT_DIR}/TOOLCHAIN.txt"

# crate name -> snapshot file basename
CRATES=(
  "datafusion-nexus"
  "nexus-query-engine"
  "nexus-bench"
  "nexus-server"
)

current_toolchain_identity() {
  local rustc_output tool_version
  if ! rustc_output=$(rustc +nightly -Vv 2>&1); then
    echo "ERROR: 'rustc +nightly -Vv' failed; is the nightly toolchain installed?" >&2
    echo "  Install it with: rustup toolchain install nightly" >&2
    return 1
  fi
  if ! tool_version=$(cargo public-api --version 2>&1); then
    echo "ERROR: 'cargo public-api --version' failed." >&2
    return 1
  fi
  printf 'cargo-public-api: %s\nnightly rustc:\n%s\n' "${tool_version}" "${rustc_output}"
}

usage() {
  cat <<'USAGE'
Usage: bash scripts/ci/check_public_api_surface.sh {check|write} [cargo-public-api feature args...]

  check   Compare each tracked crate's public API against its checked-in
          snapshot under .config/policies/public-api/. Fails closed on any
          drift (addition, removal, or signature change) or a missing
          snapshot.
  write   Regenerate every snapshot from the current working tree. Run this,
          review the resulting diff, and commit it as the deliberate approval
          step for an intentional public API change.

Any additional arguments (for example --all-features) are passed through to
`cargo public-api` unchanged, so this check always inspects the same feature
unification as the rest of the gate.
USAGE
}

[[ $# -ge 1 ]] || {
  usage >&2
  exit 2
}
mode=$1
shift
case "${mode}" in
  check | write) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: unknown mode '${mode}'." >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v cargo-public-api >/dev/null 2>&1; then
  echo "ERROR: cargo-public-api is not installed. Install it with:" >&2
  echo "  cargo install cargo-public-api --locked" >&2
  exit 2
fi

status=0
work_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-public-api.XXXXXX")
trap 'rm -rf "${work_dir}"' EXIT

current_identity=$(current_toolchain_identity) || exit 2

if [[ "${mode}" == write ]]; then
  mkdir -p "${SNAPSHOT_DIR}"
  printf '%s\n' "${current_identity}" >"${TOOLCHAIN_FILE}"
  echo "WROTE: ${TOOLCHAIN_FILE}"
else
  if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
    echo "ERROR: missing ${TOOLCHAIN_FILE}." >&2
    echo "  Run: bash scripts/ci/check_public_api_surface.sh write $*" >&2
    exit 1
  fi
  recorded_identity=$(<"${TOOLCHAIN_FILE}")
  if [[ "${recorded_identity}" != "${current_identity}" ]]; then
    echo "ERROR: public API snapshots were generated with a different toolchain than the one active now." >&2
    echo "  This is a toolchain mismatch, not necessarily a public API change: rustdoc JSON content is" >&2
    echo "  not guaranteed stable across nightly builds, so a raw item diff here would be unreliable." >&2
    echo "  Recorded in ${TOOLCHAIN_FILE}:" >&2
    sed 's/^/    /' <<<"${recorded_identity}" >&2
    echo "  Currently active:" >&2
    sed 's/^/    /' <<<"${current_identity}" >&2
    echo "  If the current toolchain is intentional (for example after a nightly bump), confirm no" >&2
    echo "  real public API changed, then regenerate with:" >&2
    echo "    bash scripts/ci/check_public_api_surface.sh write $*" >&2
    echo "  Otherwise, install the recorded nightly build shown above and rerun this check." >&2
    exit 1
  fi
fi

for crate in "${CRATES[@]}"; do
  snapshot="${SNAPSHOT_DIR}/${crate}.txt"
  actual="${work_dir}/${crate}.txt"
  err="${work_dir}/${crate}.err"
  if ! cargo public-api -p "${crate}" "$@" -sss --color never >"${actual}" 2>"${err}"; then
    echo "ERROR: cargo public-api failed for '${crate}':" >&2
    sed 's/^/  /' "${err}" >&2
    status=1
    continue
  fi

  if [[ "${mode}" == write ]]; then
    mkdir -p "${SNAPSHOT_DIR}"
    cp "${actual}" "${snapshot}"
    echo "WROTE: ${snapshot} ($(wc -l <"${actual}") lines)"
    continue
  fi

  if [[ ! -f "${snapshot}" ]]; then
    echo "ERROR: missing public API snapshot '${snapshot}' for crate '${crate}'." >&2
    echo "  Run: bash scripts/ci/check_public_api_surface.sh write $*" >&2
    status=1
    continue
  fi

  if diff -u "${snapshot}" "${actual}" >"${work_dir}/${crate}.diff"; then
    echo "PASS: ${crate} public API matches ${snapshot}"
  else
    echo "ERROR: public API of '${crate}' drifted from the checked-in snapshot '${snapshot}':" >&2
    cat "${work_dir}/${crate}.diff" >&2
    echo "  If this change is intentional, review the diff above, then run:" >&2
    echo "    bash scripts/ci/check_public_api_surface.sh write $*" >&2
    echo "  and commit the updated snapshot." >&2
    status=1
  fi
done

if [[ "${PUBLIC_API_SKIP_FACADE_POLICY:-0}" != "1" ]]; then
  ledger="${SNAPSHOT_DIR}/datafusion-nexus-engine-types.tsv"
  if ! python3 scripts/ci/check_public_api_facade.py "${SNAPSHOT_DIR}/datafusion-nexus.txt" "${ledger}"; then
    status=1
  fi
  if ! bash scripts/ci/check_public_api_facade_consumer.sh; then
    status=1
  fi
fi

exit "${status}"
