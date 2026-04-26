#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-public-terminology.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
fixture="${tmp_dir}/fixture"
mkdir -p \
  "${fixture}/crates/datafusion-nexus-examples/src" \
  "${fixture}/crates/datafusion-nexus-tools/src/bin" \
  "${fixture}/crates/datafusion-nexus-tools/src/e2e" \
  "${fixture}/tests" \
  "${fixture}/docs/plan" \
  "${fixture}/docs/report" \
  "${fixture}/website/docs" \
  "${fixture}/bin"

cat >"${fixture}/bin/npm" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${NPM_LOG}"
[[ "$1" == "--prefix" && "$3" == "run" && "$4" == "build" ]] || exit 64
mkdir -p "$2/build"
printf '%s\n' "${RENDERED_WEBSITE_TEXT:-current terminology}" >"$2/build/index.html"
SH
chmod +x "${fixture}/bin/npm"

printf '# Current public documentation\n' >"${fixture}/README.md"
printf '[package]\nname = "datafusion-nexus-tools"\n' >"${fixture}/crates/datafusion-nexus-tools/Cargo.toml"
printf 'fn main() { println!("current tool help"); }\n' >"${fixture}/crates/datafusion-nexus-tools/src/bin/tools.rs"
printf 'fn main() { println!("current example"); }\n' >"${fixture}/crates/datafusion-nexus-examples/src/main.rs"
printf 'const LEGACY: &str = "cpu-fallback";\n' >"${fixture}/crates/datafusion-nexus-tools/src/e2e/tests.rs"
printf '#[cfg(test)]\nmod tests {\n    const LEGACY: &str = "GpuFallbackPolicy";\n}\n' \
  >>"${fixture}/crates/datafusion-nexus-examples/src/main.rs"
printf '{"name":"website"}\n' >"${fixture}/website/package.json"
printf '# Current website source\n' >"${fixture}/website/docs/intro.md"

# These are intentionally outside the public checker surface. They prove the
# scope keeps ignored planning/report material and non-public Rust tests out.
printf 'GpuFallbackPolicy\n' >"${fixture}/docs/plan/legacy.md"
printf 'cpu-fallback\n' >"${fixture}/docs/report/legacy.md"
printf 'let legacy = "cpu_fallback";\n' >"${fixture}/tests/non_public.rs"
printf 'overall=cpu_only\nNATIVE_EXECUTION_TSV_HEADER\ncpu_fallback_after_native_start=false\n// gpuonly artifact\n' \
  >>"${fixture}/website/docs/intro.md"

run_check() {
  local name=$1
  shift
  local output="${tmp_dir}/${name}.out"
  local status=0
  PATH="${fixture}/bin:${PATH}" NPM_LOG="${tmp_dir}/${name}.npm.log" "$@" >"${output}" 2>&1 || status=$?
  printf '%s\n' "${status}" >"${tmp_dir}/${name}.status"
}

assert_status() {
  local name=$1 expected=$2 actual
  actual=$(<"${tmp_dir}/${name}.status")
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "FAIL: ${name}: expected ${expected}, got ${actual}" >&2
    sed 's/^/  /' "${tmp_dir}/${name}.out" >&2
    exit 1
  fi
}

assert_contains() {
  local file=$1 expected=$2
  if ! grep -Fq -- "${expected}" "${file}"; then
    echo "FAIL: ${file} does not contain: ${expected}" >&2
    sed 's/^/  /' "${file}" >&2
    exit 1
  fi
}

check=(bash scripts/ci/check_public_terminology.sh --root "${fixture}")

run_check pass "${check[@]}"
assert_status pass 0
assert_contains "${tmp_dir}/pass.out" "OK: maintained public docs"
assert_contains "${tmp_dir}/pass.npm.log" "--prefix ${fixture}/website run build"

reject_root_term() {
  local name=$1 term=$2 rule=$3
  printf '%s\n' "${term}" >>"${fixture}/README.md"
  run_check "${name}" "${check[@]}"
  assert_status "${name}" 1
  assert_contains "${tmp_dir}/${name}.out" "README.md:2: ${rule}"
  sed -i '$d' "${fixture}/README.md"
}

reject_root_term root_policy GpuFallbackPolicy GpuFallbackPolicy
reject_root_term gpu_preferred GpuPreferred 'GpuPreferred policy wording'
reject_root_term gpu_only GpuOnly 'GpuOnly policy wording'
reject_root_term partial_native PartialNative PartialNative
reject_root_term iceberg_cpu iceberg-cpu iceberg-cpu
reject_root_term iceberg_provider IcebergProviderKind IcebergProviderKind

printf 'cpu-fallback-contract-validate = []\n' >>"${fixture}/crates/datafusion-nexus-tools/Cargo.toml"
run_check tool_selector "${check[@]}"
assert_status tool_selector 1
assert_contains "${tmp_dir}/tool_selector.out" "legacy CPU fallback contract tool or environment"
sed -i '$d' "${fixture}/crates/datafusion-nexus-tools/Cargo.toml"

printf 'NEXUS_SERVER_CPU_FALLBACK_WAIT_TIMEOUT_MS\n' >>"${fixture}/website/docs/intro.md"
run_check website_source "${check[@]}"
assert_status website_source 1
assert_contains "${tmp_dir}/website_source.out" "website/docs/intro.md:6: NEXUS_SERVER_CPU_FALLBACK_* environment variables"
sed -i '$d' "${fixture}/website/docs/intro.md"

printf 'fallback.max_concurrent_statements = 2\n' >>"${fixture}/website/docs/intro.md"
run_check legacy_toml "${check[@]}"
assert_status legacy_toml 1
assert_contains "${tmp_dir}/legacy_toml.out" "website/docs/intro.md:6: legacy fallback TOML settings"
sed -i '$d' "${fixture}/website/docs/intro.md"

run_check rendered env RENDERED_WEBSITE_TEXT='cpu_fallback' "${check[@]}"
assert_status rendered 1
assert_contains "${tmp_dir}/rendered.out" "website/build/index.html:1: cpu_fallback or cpu-fallback runtime scenario alias"

echo "PASS: public terminology checker rejects removed public terms and preserves allowed machine/test artifacts"
