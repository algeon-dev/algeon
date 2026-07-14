#!/usr/bin/env bash
# Advisory and growth-ratchet checks for Rust source layout.
#
# New production modules over 800 lines are review signals; 1,200 lines is a
# soft ceiling that needs an explicit responsibility rationale. Test files use
# a 1,500-line review signal. These thresholds are advisory. By contrast,
# every current oversized source file is recorded in source_layout_baseline.tsv
# and may only stay the same size or shrink.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BASELINE_FILE=${SOURCE_LAYOUT_BASELINE_FILE:-scripts/source_layout_baseline.tsv}
PRODUCTION_REVIEW_LINES=800
PRODUCTION_SOFT_CEILING_LINES=1200
TEST_REVIEW_LINES=1500

# Generated/static catalogs are intentionally kept as single files. They do
# not participate in review signals or the blocking growth ratchet.
is_generated_static() {
  case "$1" in
    crates/datafusion-nexus-bench/src/workload/tpcds_queries.rs) return 0 ;;
    *) return 1 ;;
  esac
}

declare -A baseline_lines=()
declare -A baseline_kind=()
blocking_count=0
baseline_line_number=0
while read -r kind lines path; do
  baseline_line_number=$((baseline_line_number + 1))
  [[ -z "$kind" || "$kind" == \#* ]] && continue
  if [[ ! "$kind" =~ ^(production|test)$ || ! "$lines" =~ ^[0-9]+$ || -z "$path" ]]; then
    echo "ERROR: invalid source-layout baseline entry at line $baseline_line_number: $kind $lines $path" >&2
    exit 2
  fi
  case "$path" in
    src/*.rs | tests/*.rs | crates/*.rs) ;;
    *)
      echo "ERROR: invalid source-layout baseline path at line $baseline_line_number: $path" >&2
      exit 2
      ;;
  esac
  if [[ -v "baseline_lines[$path]" ]]; then
    echo "ERROR: duplicate source-layout baseline entry at line $baseline_line_number: $path" >&2
    exit 2
  fi
  if is_generated_static "$path"; then
    echo "ERROR: generated/static allowlist entry must not also appear in the baseline: $path" >&2
    exit 2
  fi
  baseline_lines["$path"]="$lines"
  baseline_kind["$path"]="$kind"
done < "$BASELINE_FILE"

advisory_count=0

for path in "${!baseline_lines[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "BLOCKING: baseline path no longer exists: $path. Remove or rename its entry in $BASELINE_FILE." >&2
    blocking_count=$((blocking_count + 1))
  fi
done

while IFS= read -r -d '' file; do
  path="$file"
  is_generated_static "$path" && continue

  lines=$(wc -l < "$file" | tr -d '[:space:]')
  kind=production
  review_lines=$PRODUCTION_REVIEW_LINES
  if [[ "/$path/" == */tests/* || "$path" == */tests.rs ]]; then
    kind=test
    review_lines=$TEST_REVIEW_LINES
  fi

  if [[ -v "baseline_lines[$path]" ]]; then
    if [[ "${baseline_kind[$path]}" != "$kind" ]]; then
      echo "BLOCKING: baseline kind mismatch for $path: expected ${baseline_kind[$path]}, found $kind." >&2
      blocking_count=$((blocking_count + 1))
    elif (( lines > baseline_lines[$path] )); then
      echo "BLOCKING: $path grew from baseline ${baseline_lines[$path]} to $lines lines." >&2
      blocking_count=$((blocking_count + 1))
    elif (( lines < baseline_lines[$path] )); then
      echo "RATCHET: $path shrank from ${baseline_lines[$path]} to $lines; lower its baseline in $BASELINE_FILE."
    fi
    continue
  fi

  if (( lines > review_lines )); then
    if [[ "$kind" == production && "$lines" -gt "$PRODUCTION_SOFT_CEILING_LINES" ]]; then
      echo "ADVISORY: new production module $path is $lines lines (review signal $PRODUCTION_REVIEW_LINES, soft ceiling $PRODUCTION_SOFT_CEILING_LINES); document its responsibility seam."
    else
      echo "ADVISORY: new $kind module $path is $lines lines (review signal $review_lines); consider splitting by tested dimension or responsibility."
    fi
    advisory_count=$((advisory_count + 1))
  fi
done < <(find src tests crates -type f -name '*.rs' -print0 | sort -z)

if (( advisory_count > 0 )); then
  echo "SOURCE LAYOUT: $advisory_count advisory signal(s); advisory thresholds are not blocking."
fi
if (( blocking_count > 0 )); then
  echo "SOURCE LAYOUT: $blocking_count baseline growth violation(s)." >&2
  exit 1
fi
echo "SOURCE LAYOUT: baseline growth ratchet passed."
