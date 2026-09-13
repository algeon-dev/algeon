#!/usr/bin/env bash
# Policy test for scripts/ci/check_cugraph_example_ddl.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CHECK_SCRIPT="${ROOT}/scripts/ci/check_cugraph_example_ddl.sh"

scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

write_page() {
  mkdir -p "${scratch}/$1"
  cat >"${scratch}/$1/page.mdx"
}

write_page rerunnable <<'EOF'
Prose may mention `CREATE TABLE ... AS` outside SQL blocks.

```sql
-- create the view once per session
CREATE OR REPLACE VIEW seeds AS SELECT 1 AS vertex;
CREATE OR REPLACE TABLE snapshot AS SELECT * FROM seeds;
```
EOF
if ! bash "${CHECK_SCRIPT}" "${scratch}/rerunnable" >/dev/null 2>&1; then
  fail "CREATE OR REPLACE statements, prose, and comments were rejected"
fi

for statement in "CREATE VIEW seeds AS SELECT 1 AS vertex;" "  create table snapshot as select 1;"; do
  write_page bare <<EOF
\`\`\`sql
${statement}
\`\`\`
EOF
  output=$(bash "${CHECK_SCRIPT}" "${scratch}/bare" 2>&1 || true)
  if bash "${CHECK_SCRIPT}" "${scratch}/bare" >/dev/null 2>&1; then
    fail "bare DDL passed: ${statement}"
  elif [[ ${output} != *"page.mdx:2: use CREATE OR REPLACE"* ]]; then
    fail "missing file:line diagnostic for '${statement}': ${output}"
  fi
done

if ((failures > 0)); then
  echo "check_cugraph_example_ddl tests: ${failures} failure(s)" >&2
  exit 1
fi
echo "check_cugraph_example_ddl tests passed"
