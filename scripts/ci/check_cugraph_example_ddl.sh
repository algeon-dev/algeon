#!/usr/bin/env bash
# Policy check: every CREATE statement in the cuGraph documentation SQL blocks
# uses CREATE OR REPLACE. Readers run the example pages in one Flight SQL
# session, so a page must rerun cleanly and pages that share a view name must
# overwrite each other instead of failing on "already exists".
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TARGET=${1:-${ROOT}/website/docs/cugraph}

status=0
while IFS= read -r -d '' file; do
  awk -v file="${file#"${ROOT}/"}" '
    /^```sql[[:space:]]*$/ { in_sql = 1; next }
    /^```/ { in_sql = 0; next }
    in_sql {
      line = toupper($0)
      sub(/--.*/, "", line)
      if (line ~ /(^|[^A-Z0-9_])CREATE[[:space:]]/ && line !~ /CREATE[[:space:]]+OR[[:space:]]+REPLACE[[:space:]]/) {
        printf "ERROR: %s:%d: use CREATE OR REPLACE so the example reruns: %s\n", file, NR, $0 > "/dev/stderr"
        failed = 1
      }
    }
    END { exit failed }
  ' "${file}" || status=1
done < <(find "${TARGET}" -name '*.mdx' -print0 | sort -z)

if ((status == 0)); then
  echo "cuGraph example DDL policy: every CREATE uses OR REPLACE"
fi
exit "${status}"
