#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-pass-as-skip.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/tests"
printf 'fn test() { let Ok(runtime) = cudf::Runtime::new() else { return; }; }\n' >"${tmp_dir}/tests/bad.rs"

if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/bad.out" 2>&1; then
  echo "FAIL: optional CUDA runtime return unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'optional_gpu_runtime' "${tmp_dir}/bad.out"

printf 'fn test() { handle_cugraph_python_oracle_unavailable("missing"); return None; }\n' >"${tmp_dir}/tests/oracle.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/oracle.out" 2>&1; then
  echo "FAIL: optional Python oracle return unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'optional_python_oracle' "${tmp_dir}/oracle.out"
rm "${tmp_dir}/tests/oracle.rs"

printf 'fn test() { let _ = sf1_fixture_root_or_skip(); }\n' >"${tmp_dir}/tests/sf1.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/sf1.out" 2>&1; then
  echo "FAIL: optional SF1 fixture unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'optional_sf1_fixture' "${tmp_dir}/sf1.out"
rm "${tmp_dir}/tests/sf1.rs"

printf 'fn test() { eprintln!("skipping cugraph crate drift test"); }\n' >"${tmp_dir}/tests/skip.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/skip.out" 2>&1; then
  echo "FAIL: sibling source skip unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'sibling_source_skip' "${tmp_dir}/skip.out"
rm "${tmp_dir}/tests/skip.rs"

printf 'const GRAPH_RS: &str = "../../../cugraph-nexus/cugraph-nexus/src/graph.rs";\n' >"${tmp_dir}/tests/scrape.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/scrape.out" 2>&1; then
  echo "FAIL: sibling source scrape unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'sibling_source_scrape' "${tmp_dir}/scrape.out"
rm "${tmp_dir}/tests/scrape.rs"

cat >"${tmp_dir}/allow.toml" <<'TOML'
[[allow]]
path = "/tmp/placeholder"
rule = "optional_gpu_runtime"
owner = "policy self-test"
reason = "verifies allowlist schema"
TOML
if python3 scripts/ci/pass_as_skip_policy.py --allowlist "${tmp_dir}/allow.toml" --root "${tmp_dir}/tests" >"${tmp_dir}/allow.out" 2>&1; then
  echo "FAIL: mismatched allowlist path unexpectedly passed" >&2
  exit 1
fi

printf 'fn test() { let _runtime = cudf::Runtime::new().expect("required"); }\n' >"${tmp_dir}/tests/good.rs"
rm "${tmp_dir}/tests/bad.rs"
python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests"
echo "PASS: pass-as-skip policy self-tests"
