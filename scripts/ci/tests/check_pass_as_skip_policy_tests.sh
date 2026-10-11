#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/algeon-datafusion-pass-as-skip.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/tests"
printf 'fn test() { let Ok(runtime) = cudf::Runtime::new() else { return; }; }\n' >"${tmp_dir}/tests/bad.rs"

if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/bad.out" 2>&1; then
  echo "FAIL: optional CUDA runtime return unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'optional_gpu_runtime' "${tmp_dir}/bad.out"
rm "${tmp_dir}/tests/bad.rs"

printf 'fn test() { if cudf::DeviceId::count().unwrap_or(0) == 0 { return; } }\n' >"${tmp_dir}/tests/device_count.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/device_count.out" 2>&1; then
  echo "FAIL: DeviceId count guard unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'device_count_skip' "${tmp_dir}/device_count.out"
rm "${tmp_dir}/tests/device_count.rs"

printf 'fn test() { if cudf::DeviceId::count().unwrap_or(0) < 2 { return; } }\n' >"${tmp_dir}/tests/device_count_less_than.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/device_count_less_than.out" 2>&1; then
  echo "FAIL: DeviceId count < 2 guard unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'device_count_skip' "${tmp_dir}/device_count_less_than.out"
rm "${tmp_dir}/tests/device_count_less_than.rs"

for count_expression in 'cudf::DeviceId::count().expect("CUDA device count")' 'DeviceId::count().unwrap()' 'cudf::DeviceId::count().unwrap_or(0)'; do
  printf 'fn test() { let device_count = %s; if device_count < 2 { return; } }\n' "${count_expression}" \
    >"${tmp_dir}/tests/bound_device_count.rs"
  if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/bound_device_count.out" 2>&1; then
    echo "FAIL: bound DeviceId count guard unexpectedly passed" >&2
    exit 1
  fi
  grep -Fq 'bound_device_count_skip' "${tmp_dir}/bound_device_count.out"
done
rm "${tmp_dir}/tests/bound_device_count.rs"

printf 'fn test() { handle_cugraph_python_oracle_unavailable("missing"); return None; }\n' >"${tmp_dir}/tests/oracle.rs"
if python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests" >"${tmp_dir}/oracle.out" 2>&1; then
  echo "FAIL: optional Python oracle return unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'optional_python_oracle' "${tmp_dir}/oracle.out"
rm "${tmp_dir}/tests/oracle.rs"

printf 'fn test() { let _runtime = cudf::Runtime::new().expect("required"); }\n' >"${tmp_dir}/tests/good.rs"
cat >"${tmp_dir}/tests/allowed_returns.rs" <<'EOF'
fn optional_input(input: Option<usize>) { let Some(rows) = input else { return; }; }
fn empty_input() { let device_count = rows.len(); if device_count < 2 { return; } }
fn different_count() { let device_count = cudf::DeviceId::count().expect("required"); if row_count < 2 { return; } }
fn required_devices() { let device_count = cudf::DeviceId::count().expect("required"); assert!(device_count >= 2); }
EOF
python3 scripts/ci/pass_as_skip_policy.py --root "${tmp_dir}/tests"

# Exercise default-root discovery in an isolated checkout, without --root.
mkdir -p "${tmp_dir}/checkout/scripts/ci" "${tmp_dir}/checkout/components/example/tests"
cp scripts/ci/pass_as_skip_policy.py "${tmp_dir}/checkout/scripts/ci/"
printf 'fn test() { if cudf::DeviceId::count().unwrap_or(0) < 2 { return; } }\n' \
  >"${tmp_dir}/checkout/components/example/tests/bad.rs"
if python3 "${tmp_dir}/checkout/scripts/ci/pass_as_skip_policy.py" >"${tmp_dir}/component.out" 2>&1; then
  echo "FAIL: component pass-as-skip escaped default-root discovery" >&2
  exit 1
fi
grep -Fq 'components/example/tests/bad.rs: device_count_skip' "${tmp_dir}/component.out"

python3 scripts/ci/pass_as_skip_policy.py
echo "PASS: pass-as-skip policy self-tests"
