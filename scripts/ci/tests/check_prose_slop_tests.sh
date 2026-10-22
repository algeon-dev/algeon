#!/usr/bin/env bash
# Policy test for scripts/ci/check_prose_slop.sh.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
CHECK_SCRIPT="${ROOT}/scripts/ci/check_prose_slop.sh"

if [[ ! -x "${CHECK_SCRIPT}" ]]; then
  echo "FAIL: check_prose_slop script not found or not executable at ${CHECK_SCRIPT}" >&2
  exit 1
fi

scratch=$(mktemp -d)
trap 'rm -rf "${scratch}"' EXIT
failures=0

fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

# 1. Clean file passes.
cat <<'EOF' >"${scratch}/clean.md"
# Query Optimization

The planner lowers logical plans into physical candidates.
Each operator adheres to memory bounds and device ledger accounting.
Errors are typed and mapped at system boundaries.
EOF

if ! "${CHECK_SCRIPT}" "${scratch}/clean.md" >/dev/null 2>&1; then
  fail "clean file unexpectedly failed check_prose_slop"
fi

# 1b. The default scan reads website/docs/ only; the gitignored docs/ and
# note/ trees are never scanned. The script scans the checkout that contains
# it, so the scratch checkout gets its own copy.
mkdir -p "${scratch}/checkout/website/docs" "${scratch}/checkout/scripts/ci" \
  "${scratch}/checkout/docs" "${scratch}/checkout/note"
cp "${scratch}/clean.md" "${scratch}/checkout/website/docs/clean.md"
printf 'A private draft \u2014 never scanned.\n' >"${scratch}/checkout/docs/draft.md"
printf 'A private note \u2014 never scanned.\n' >"${scratch}/checkout/note/draft.md"
cp "${CHECK_SCRIPT}" "${scratch}/checkout/scripts/ci/check_prose_slop.sh"
checkout_script="${scratch}/checkout/scripts/ci/check_prose_slop.sh"
if ! (cd "${scratch}/checkout" && "${checkout_script}" >/dev/null 2>&1); then
  fail "default scan read docs/ or note/"
fi
printf 'A published page \u2014 scanned.\n' >"${scratch}/checkout/website/docs/page.md"
if (cd "${scratch}/checkout" && "${checkout_script}" >/dev/null 2>&1); then
  fail "default scan missed a violation under website/docs/"
fi
if (cd "${scratch}/checkout" && "${checkout_script}" missing.md >/dev/null 2>&1); then
  fail "an explicitly named missing path unexpectedly passed"
fi

# 2. Em-dash is detected.
cat <<'EOF' >"${scratch}/em_dash.md"
This is an architecture document — it explains memory limits.
EOF

if "${CHECK_SCRIPT}" "${scratch}/em_dash.md" >/dev/null 2>&1; then
  fail "file containing em-dash unexpectedly passed"
fi
err_out=$("${CHECK_SCRIPT}" "${scratch}/em_dash.md" 2>&1 || true)
if ! echo "${err_out}" | grep -q "em-dash"; then
  fail "em-dash error message missing: ${err_out}"
fi

# 3. Each banned phrase is detected.
banned_phrases=(
  "delve"
  "seamlessly"
  "pivotal"
  "testament to"
  "robust and scalable"
  "in today's data landscape"
  "revolutionize"
  "harness the power of"
)

for phrase in "${banned_phrases[@]}"; do
  file="${scratch}/test_${phrase// /_}.md"
  printf '# Test\nWe must %s our system.\n' "${phrase}" >"${file}"
  if "${CHECK_SCRIPT}" "${file}" >/dev/null 2>&1; then
    fail "banned phrase '${phrase}' was not detected"
  fi
  out=$("${CHECK_SCRIPT}" "${file}" 2>&1 || true)
  if ! echo "${out}" | grep -qi "${phrase}"; then
    fail "banned phrase '${phrase}' not mentioned in error output: ${out}"
  fi
done

# 4. Fenced code blocks and inline code are ignored.
cat <<'EOF' >"${scratch}/code_ignored.md"
# Documentation of Ban

The following items are banned:
- The em-dash character `—` is prohibited.
- Do not use `delve` or `seamlessly` in prose.

```rust
// In code comments or code strings, delve or — might appear:
let x = "delve";
// test — comment
```
EOF

if ! "${CHECK_SCRIPT}" "${scratch}/code_ignored.md" >/dev/null 2>&1; then
  err_msg=$("${CHECK_SCRIPT}" "${scratch}/code_ignored.md" 2>&1 || true)
  fail "inline code or code block falsely triggered violation: ${err_msg}"
fi

# 5. Exclude pattern works.
cat <<'EOF' >"${scratch}/excluded_slop.md"
This file delves into revolutionary systems.
EOF

if ! "${CHECK_SCRIPT}" --exclude "excluded" "${scratch}/excluded_slop.md" >/dev/null 2>&1; then
  fail "--exclude option failed to skip matching file"
fi

# 6. Valid YAML front matter passes.
cat <<'EOF' >"${scratch}/valid_frontmatter.md"
---
sidebar_position: 1
description: "A valid description: with a quoted colon"
---
# Content
Document content without slop.
EOF

if ! "${CHECK_SCRIPT}" "${scratch}/valid_frontmatter.md" >/dev/null 2>&1; then
  fail "valid front matter unexpectedly failed check"
fi

# 7. Invalid YAML front matter (unquoted colon) is detected.
cat <<'EOF' >"${scratch}/invalid_frontmatter.md"
---
sidebar_position: 1
description: An invalid description: with unquoted colon
---
# Content
Document content.
EOF

if "${CHECK_SCRIPT}" "${scratch}/invalid_frontmatter.md" >/dev/null 2>&1; then
  fail "invalid front matter was not detected"
fi
out=$("${CHECK_SCRIPT}" "${scratch}/invalid_frontmatter.md" 2>&1 || true)
if ! echo "${out}" | grep -q "invalid front matter YAML"; then
  fail "invalid front matter error message missing: ${out}"
fi

# 8. Unclosed front matter delimiter is detected.
cat <<'EOF' >"${scratch}/unclosed_frontmatter.md"
---
sidebar_position: 1
# Content
Document content.
EOF

if "${CHECK_SCRIPT}" "${scratch}/unclosed_frontmatter.md" >/dev/null 2>&1; then
  fail "unclosed front matter was not detected"
fi
out=$("${CHECK_SCRIPT}" "${scratch}/unclosed_frontmatter.md" 2>&1 || true)
if ! echo "${out}" | grep -q "unclosed front matter delimiter"; then
  fail "unclosed front matter error message missing: ${out}"
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "check_prose_slop_tests: ${failures} check(s) failed" >&2
  exit 1
fi

echo "check_prose_slop_tests: all checks passed"
