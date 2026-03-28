#!/usr/bin/env bash
# Run the full local check surface in one shot: the test suite AND the rustdoc
# lints (the "cargo doc" warnings).
#
# Why this exists: `cargo test` never builds documentation. Its only rustdoc
# pass is the doctest runner (`rustdoc --test`), which compiles ```fenced```
# examples but does NOT run rustdoc's lint/link-resolution passes. So the
# rustdoc lints configured in the workspace manifest --
#   [workspace.lints.rustdoc] broken_intra_doc_links = "deny", bare_urls = "deny"
# -- are only evaluated by `cargo doc`, never by `cargo test`. This script runs
# both passes so a single command surfaces test failures and doc-lint failures
# together. Like `--no-fail-fast`, it does not abort after the first failing
# pass; it runs everything and reports a combined status at the end.
#
# Usage:
#   bash scripts/check_all.sh                      # nextest + doctest + doc, --all-features
#   FEATURES="server,iceberg" bash scripts/check_all.sh
#   TEST_RUNNER=cargo bash scripts/check_all.sh    # force the legacy cargo test runner
#   GPU_LOCK=/tmp/cudf-gpu.lock bash scripts/check_all.sh
#   GPU_PARALLEL=1 GPU_TEST_THREADS=4 bash scripts/check_all.sh
#   STRICT=1 bash scripts/check_all.sh             # promote ALL rustdoc warnings to errors
#
# The configured deny-lints already fail `cargo doc` on their own. Set STRICT=1
# to additionally turn every other rustdoc warning into a hard error.
set -uo pipefail

FEATURES="${FEATURES:-}"
if [[ -n "${FEATURES}" ]]; then
  FEATURE_ARGS=(--features "${FEATURES}")
else
  FEATURE_ARGS=(--all-features)
fi

DOC_RUSTDOCFLAGS="${RUSTDOCFLAGS:-}"
if [[ "${STRICT:-0}" == "1" ]]; then
  DOC_RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS:+${DOC_RUSTDOCFLAGS} }-D warnings"
fi

status=0

TEST_RUNNER="${TEST_RUNNER:-nextest}"
GPU_LOCK="${GPU_LOCK:-/tmp/cudf-gpu.lock}"
GPU_PARALLEL="${GPU_PARALLEL:-0}"
GPU_TEST_THREADS="${GPU_TEST_THREADS:-4}"
gpu_parallel_enabled=0
case "${GPU_PARALLEL}" in
  1 | true | TRUE | yes | YES | on | ON)
    gpu_parallel_enabled=1
    ;;
esac

if [[ "${TEST_RUNNER}" == "nextest" ]] && cargo nextest --version >/dev/null 2>&1; then
  echo "==> cargo nextest run ${FEATURE_ARGS[*]} -E 'not group(gpu)' --no-fail-fast"
  cargo nextest run "${FEATURE_ARGS[@]}" -E 'not group(gpu)' --no-fail-fast || status=1

  echo
  if [[ "${gpu_parallel_enabled}" -eq 1 ]]; then
    if ! [[ "${GPU_TEST_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: GPU_TEST_THREADS must be a positive integer, got '${GPU_TEST_THREADS}'."
      exit 2
    fi
    echo "==> flock ${GPU_LOCK} cargo nextest run -P gpu-parallel ${FEATURE_ARGS[*]} -E 'group(gpu-parallel)' --no-fail-fast --test-threads ${GPU_TEST_THREADS}"
    flock "${GPU_LOCK}" cargo nextest run -P gpu-parallel "${FEATURE_ARGS[@]}" -E 'group(gpu-parallel)' --no-fail-fast --test-threads "${GPU_TEST_THREADS}" || status=1
  else
    echo "==> flock ${GPU_LOCK} cargo nextest run ${FEATURE_ARGS[*]} -E 'group(gpu)' --no-fail-fast"
    flock "${GPU_LOCK}" cargo nextest run "${FEATURE_ARGS[@]}" -E 'group(gpu)' --no-fail-fast || status=1
  fi

  echo
  echo "==> cargo test ${FEATURE_ARGS[*]} --doc --no-fail-fast  (doctests)"
  cargo test "${FEATURE_ARGS[@]}" --doc --no-fail-fast || status=1
else
  if [[ "${TEST_RUNNER}" == "nextest" ]]; then
    echo "WARN: cargo-nextest is not installed; falling back to cargo test."
  fi
  echo "==> flock ${GPU_LOCK} cargo test ${FEATURE_ARGS[*]} --no-fail-fast -- --test-threads=1"
  RUST_TEST_THREADS=1 flock "${GPU_LOCK}" cargo test "${FEATURE_ARGS[@]}" --no-fail-fast -- --test-threads=1 || status=1
fi

echo
echo "==> cargo doc ${FEATURE_ARGS[*]} --no-deps --keep-going  (rustdoc lints)"
RUSTDOCFLAGS="${DOC_RUSTDOCFLAGS}" cargo doc "${FEATURE_ARGS[@]}" --no-deps --keep-going || status=1

echo
if [[ "${status}" -ne 0 ]]; then
  echo "FAIL: one or more checks reported problems (see above)."
else
  echo "OK: tests passed and rustdoc lints are clean."
fi
exit "${status}"
