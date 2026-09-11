#!/usr/bin/env bash
# Isolated safety tests for the explicit target-artifact maintenance helper.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SCRIPT="${ROOT}/scripts/dev/clean_target_artifacts.sh"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/algeon-datafusion-clean-target-tests.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT

mkdir -p "${tmp_dir}/bin" "${tmp_dir}/cargo-target/debug/deps"
touch "${tmp_dir}/cargo-target/.rustc_info.json" "${tmp_dir}/cargo-target/CACHEDIR.TAG"

cat >"${tmp_dir}/bin/ps" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$*" >"${FAKE_CARGO_LOG}"
exit 0
SH
chmod +x "${tmp_dir}/bin/ps" "${tmp_dir}/bin/cargo"

run_script() {
  PATH="${tmp_dir}/bin:${PATH}" \
    FAKE_CARGO_LOG="${tmp_dir}/cargo.log" \
    TARGET_DIR="$1" \
    bash "${SCRIPT}" "${@:2}"
}

run_script "${tmp_dir}/cargo-target" --report >"${tmp_dir}/report.out"
rg -q "target directory: ${tmp_dir}/cargo-target" "${tmp_dir}/report.out"
echo "PASS: report accepts a marked custom Cargo target"

for index in $(seq 1 21); do
  touch "${tmp_dir}/cargo-target/debug/deps/.mold-probe-${index}"
done
run_script "${tmp_dir}/cargo-target" --report >"${tmp_dir}/many-mold.out"
rg -Fq "... (1 more)" "${tmp_dir}/many-mold.out"
echo "PASS: report handles more than 20 mold files under pipefail"

run_script "${tmp_dir}/cargo-target" --mold-only --yes >"${tmp_dir}/mold.out"
[[ -z "$(find "${tmp_dir}/cargo-target" -name '.mold-*' -print -quit)" ]]
echo "PASS: mold-only cleanup stays inside the selected Cargo target"

: >"${tmp_dir}/cargo.log"
run_script "${tmp_dir}/cargo-target" --clean-cargo-target --yes >"${tmp_dir}/cargo-clean.out"
expected="clean --manifest-path ${ROOT}/Cargo.toml --target-dir ${tmp_dir}/cargo-target"
[[ "$(<"${tmp_dir}/cargo.log")" == "${expected}" ]]
echo "PASS: cargo clean receives the exact selected target directory"

mkdir -p "${tmp_dir}/unmarked"
if run_script "${tmp_dir}/unmarked" --mold-only --yes >"${tmp_dir}/unmarked.out" 2>&1; then
  echo "FAIL: destructive cleanup accepted an unmarked directory" >&2
  exit 1
fi
rg -q "without Cargo target markers" "${tmp_dir}/unmarked.out"
echo "PASS: destructive cleanup rejects unmarked directories"

if TARGET_DIR="${ROOT}" bash "${SCRIPT}" --mold-only --yes >"${tmp_dir}/broad.out" 2>&1; then
  echo "FAIL: destructive cleanup accepted the workspace root" >&2
  exit 1
fi
rg -q "refusing broad cleanup target" "${tmp_dir}/broad.out"
echo "PASS: destructive cleanup rejects workspace ancestors"

if run_script "${tmp_dir}/cargo-target" --mold-only --clean-cargo-target --yes \
  >"${tmp_dir}/conflict.out" 2>&1; then
  echo "FAIL: conflicting destructive modes were accepted" >&2
  exit 1
fi
rg -q "conflicts with" "${tmp_dir}/conflict.out"
echo "PASS: conflicting destructive modes fail closed"

bash "${SCRIPT}" --help >"${tmp_dir}/help.out"
rg -q "Usage:" "${tmp_dir}/help.out"
echo "PASS: help exits successfully without inspecting artifacts"

echo "OK: target-artifact cleanup safety tests passed."
