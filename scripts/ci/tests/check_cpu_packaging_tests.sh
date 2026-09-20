#!/usr/bin/env bash
# Isolated CPU packaging/provenance tests. Uses fake cargo/rustc and fixtures;
# it does not compile the release server or build a Docker image.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"

HELPER="${ROOT}/scripts/build/cpu_packaging_policy.py"
POLICY="${ROOT}/.config/policies/cpu-packaging.toml"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/algeon-datafusion-cpu-policy.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/home"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

if rg -n 'ensure_cpu_baseline|is_x86_feature_detected!' crates/algeon-server/src/bin/algeon_server.rs; then
  fail "server still contains an in-process CPU feature check"
fi
echo "PASS: server has no in-process CPU feature detection"

python3 - "${POLICY}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as policy_file:
    policy = tomllib.load(policy_file)
assert policy["clean_environment"]["explicit_inputs"] == [
    "ALGEON_NATIVE_DIR",
]
cpu = policy["cpu"]
assert cpu["baseline"] == "x86-64-v4"
assert cpu["rustflags"] == ["-C", "target-cpu=x86-64-v4"]
assert cpu["artifact_dir"] == "x86-64-v4"
assert "x86-64-v4" in cpu["deployment_requirement"]
PY
echo "PASS: CPU packaging has one fixed x86-64-v4 contract"

shipped=$(python3 "${HELPER}" shipped-features)
[[ "${shipped}" == "$(python3 "${HELPER}" normalize-features --features "${shipped}")" ]] \
  || fail "server.features is not in canonical form: ${shipped}"
if grep -Eq 'FEATURES:-[A-Za-z]' docker/stage-docker-payload.sh; then
  fail "docker staging hard-codes a default feature list instead of reading server.features"
fi
grep -Fq 'cpu_packaging_policy.py shipped-features' scripts/check_all.sh \
  || fail "the shipped-clippy phase does not read server.features"
cp "${POLICY}" "${tmp_dir}/unsorted-features-policy.toml"
python3 - "${tmp_dir}/unsorted-features-policy.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
assert '\nfeatures = ["cugraph", "cuml"' in text
path.write_text(text.replace('\nfeatures = ["cugraph", "cuml"', '\nfeatures = ["cuml", "cugraph"', 1))
PY
status=0
python3 "${HELPER}" --policy "${tmp_dir}/unsorted-features-policy.toml" shipped-features \
  >"${tmp_dir}/unsorted-features.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "an unsorted server.features list was accepted"
echo "PASS: packaging and the shipped clippy lane read one canonical feature list"

cat >"${tmp_dir}/bin/cargo" <<'SH'
#!/usr/bin/bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "cargo 1.95.0 (fake 2026-01-01)"
  exit 0
fi
for forbidden in RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS CFLAGS CXXFLAGS; do
  [[ ! -v "${forbidden}" ]] || { echo "dirty build environment: ${forbidden}" >&2; exit 88; }
done
[[ "${CC:-}" == "/usr/bin/gcc-14" ]]
[[ "${CXX:-}" == "/usr/bin/g++-14" ]]
[[ "${CUDAHOSTCXX:-}" == "/usr/bin/g++-14" ]]
[[ "${AR:-}" == "/usr/bin/ar" ]]
[[ "${CMAKE:-}" == "/usr/bin/cmake" ]]
[[ "${CMAKE_GENERATOR:-}" == "Ninja" ]]
[[ -n "${CARGO_HOME:-}" && ! -e "${CARGO_HOME}/config.toml" && ! -e "${CARGO_HOME}/config" ]]

target_dir=""
target=""
profile=""
previous=""
for argument in "$@"; do
  case "${previous}" in
    --target-dir) target_dir="${argument}" ;;
    --target) target="${argument}" ;;
    --profile) profile="${argument}" ;;
  esac
  previous="${argument}"
done
[[ -n "${target_dir}" && -n "${target}" && -n "${profile}" ]]
mkdir -p "${target_dir}/${target}/${profile}"
printf 'fake canonical server\n' >"${target_dir}/${target}/${profile}/algeon_server"
SH

cat >"${tmp_dir}/bin/rustc" <<'SH'
#!/usr/bin/bash
set -euo pipefail
case "${1:-}" in
  --version)
    echo "rustc 1.95.0 (fake 2026-01-01)"
    ;;
  -Vv)
    cat <<'VV'
rustc 1.95.0 (fake 2026-01-01)
binary: rustc
commit-hash: fake0001
commit-date: 2026-01-01
host: x86_64-unknown-linux-gnu
release: 1.95.0
LLVM version: 22.0.0
VV
    ;;
  *) exit 90 ;;
esac
SH
chmod +x "${tmp_dir}/bin/cargo" "${tmp_dir}/bin/rustc"

target_root="${tmp_dir}/target"
env -i PATH="${tmp_dir}/bin:${PATH}" HOME="${tmp_dir}/home" \
  CC='gcc -march=native' CXX='g++ -march=x86-64-v4' \
  CUDAHOSTCXX='g++ -march=native' AR='ar --plugin=/tmp/unsafe' \
  CMAKE='/tmp/ambient-cmake' CMAKE_GENERATOR='Unix Makefiles' \
  RUSTFLAGS='-C target-cpu=native' \
  CARGO_BUILD_RUSTFLAGS='-C target-cpu=x86-64-v3' \
  python3 "${HELPER}" build \
    --features 'nvml' \
    --target-root "${target_root}" \
    --manifest "${ROOT}/crates/algeon-server/Cargo.toml" \
    --explicit-env "ALGEON_NATIVE_DIR=${tmp_dir}/native" \
    >"${tmp_dir}/build.output"

binary="${target_root}/x86-64-v4/x86_64-unknown-linux-gnu/release/algeon_server"
build_record="${binary}.build-record.json"
[[ -f "${binary}" && -f "${build_record}" ]] || fail "fake canonical build did not create artifact/record"

python3 - "${build_record}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as record_file:
    record = json.load(record_file)
cpu = record["cpu"]
command = record["cargo_command"]
assert record["record_kind"] == "canonical-build-record"
assert record["features"] == ["nvml"]
assert cpu["baseline"] == "x86-64-v4"
assert cpu["target_triple"] == "x86_64-unknown-linux-gnu"
assert cpu["effective_rustflags"] == ["-C", "target-cpu=x86-64-v4"]
assert record["rustc_version"].startswith("rustc 1.95.0")
assert "LLVM version: 22.0.0" in record["rustc_vv"]
assert record["cargo_version"].startswith("cargo 1.95.0")
assert record["fixed_build_environment"] == {
    "AR": "/usr/bin/ar",
    "CC": "/usr/bin/gcc-14",
    "CMAKE": "/usr/bin/cmake",
    "CMAKE_GENERATOR": "Ninja",
    "CUDAHOSTCXX": "/usr/bin/g++-14",
    "CXX": "/usr/bin/g++-14",
}
assert set(record["fixed_tool_versions"]) == {"ar", "cc", "cmake", "cxx", "ninja"}
assert all(record["fixed_tool_versions"].values())
assert "--target" in command and "x86_64-unknown-linux-gnu" in command
assert "target.x86_64-unknown-linux-gnu.rustflags=[\"-C\",\"target-cpu=x86-64-v4\"]" in command
assert "--no-default-features" in command and "--locked" in command
PY
echo "PASS: ambient compiler selectors are replaced and build record captures fixed tool inputs"
echo "PASS: canonical command and build record reflect exact policy inputs"

env -i PATH="${tmp_dir}/bin:${PATH}" HOME="${tmp_dir}/home" \
  python3 "${HELPER}" verify-build \
    --features 'nvml' \
    --binary "${binary}" --record "${build_record}" \
    >"${tmp_dir}/reuse.output"
echo "PASS: verified canonical artifact can be reused"

cp "${binary}" "${tmp_dir}/binary.backup"
printf 'tamper\n' >>"${binary}"
status=0
python3 "${HELPER}" verify-build \
  --features 'nvml' \
  --binary "${binary}" --record "${build_record}" \
  >"${tmp_dir}/tampered-binary.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "tampered canonical artifact was accepted"
grep -Fq 'artifact hash mismatch' "${tmp_dir}/tampered-binary.output" || fail "missing artifact tamper diagnostic"
cp "${tmp_dir}/binary.backup" "${binary}"
echo "PASS: artifact tamper prevents SKIP_CARGO_BUILD reuse"

cp "${POLICY}" "${tmp_dir}/prose-policy.toml"
python3 - "${tmp_dir}/prose-policy.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace('deployment_requirement = "The host CPU', 'deployment_requirement = "CHANGED: The host CPU', 1))
PY
python3 "${HELPER}" --policy "${tmp_dir}/prose-policy.toml" verify-build \
  --features 'nvml' \
  --binary "${binary}" --record "${build_record}" \
  >"${tmp_dir}/prose-policy.output"
echo "PASS: prose change in policy does not invalidate artifact reuse"

cp "${POLICY}" "${tmp_dir}/tampered-policy.toml"
python3 - "${tmp_dir}/tampered-policy.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace('baseline = "x86-64-v4"', 'baseline = "x86-64-v3"', 1).replace('target-cpu=x86-64-v4', 'target-cpu=x86-64-v3', 1))
PY
status=0
python3 "${HELPER}" --policy "${tmp_dir}/tampered-policy.toml" verify-build \
  --features 'nvml' \
  --binary "${binary}" --record "${build_record}" \
  >"${tmp_dir}/tampered-policy.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "changed policy was accepted for artifact reuse"
grep -Fq 'policy_fingerprint' "${tmp_dir}/tampered-policy.output" || fail "missing policy tamper diagnostic"
echo "PASS: artifact-affecting policy tamper prevents artifact reuse"

staged_binary="${tmp_dir}/dist/bin/algeon_server"
mkdir -p "$(dirname "${staged_binary}")"
cp "${binary}" "${staged_binary}"
printf 'post-stage-patchelf-fixture\n' >>"${staged_binary}"
stage_manifest="${tmp_dir}/dist/payload-manifest.json"
python3 "${HELPER}" create-manifest \
  --dist-dir "${tmp_dir}/dist" \
  --build-record "${build_record}" \
  --output "${stage_manifest}" \
  >"${tmp_dir}/manifest.output"

eval "$(python3 "${HELPER}" verify-manifest --manifest "${stage_manifest}" --dist-dir "${tmp_dir}/dist")"
[[ "${CPU_BASELINE}" == "x86-64-v4" ]]
[[ "${CPU_IMAGE_ARCH}" == "amd64" ]]
[[ "${CPU_TARGET_TRIPLE}" == "x86_64-unknown-linux-gnu" ]]
[[ -n "${CPU_BINARY_SHA256}" ]]
[[ -n "${CPU_MANIFEST_SHA256}" ]]
echo "PASS: payload manifest created and verified"

cp "${staged_binary}" "${tmp_dir}/staged_binary.backup"
printf 'tamper-staged\n' >>"${staged_binary}"
status=0
python3 "${HELPER}" verify-manifest --manifest "${stage_manifest}" --dist-dir "${tmp_dir}/dist" \
  >"${tmp_dir}/tampered-staged.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "tampered staged binary was accepted"
grep -Fq 'hash mismatch' "${tmp_dir}/tampered-staged.output" || fail "missing staged file tamper diagnostic"
cp "${tmp_dir}/staged_binary.backup" "${staged_binary}"
echo "PASS: tampered staged file fails manifest verification"

mkdir -p "${tmp_dir}/dist/lib"
printf 'untracked\n' >"${tmp_dir}/dist/lib/untracked.so"
status=0
python3 "${HELPER}" verify-manifest --manifest "${stage_manifest}" --dist-dir "${tmp_dir}/dist" \
  >"${tmp_dir}/untracked.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "untracked staged file was accepted"
grep -Fq 'inventory mismatch' "${tmp_dir}/untracked.output" || fail "missing inventory mismatch diagnostic"
rm -f "${tmp_dir}/dist/lib/untracked.so"
echo "PASS: unexpected staged file fails manifest verification"

for needle in \
  'dev.algeon.cpu-baseline="${CPU_BASELINE}"' \
  'dev.algeon.cpu-arch="${CPU_ARCH}"' \
  'dev.algeon.cpu-target="${CPU_TARGET}"' \
  'dev.algeon.manifest-sha256="${CPU_MANIFEST_SHA256}"' \
  'CPU_MANIFEST_SHA256' \
  'sha256sum -c -'; do
  grep -Fq "${needle}" docker/Dockerfile || fail "Dockerfile missing assertion: ${needle}"
done
grep -Fq 'verify-build' docker/stage-docker-payload.sh || fail "SKIP_CARGO_BUILD path does not verify record"
if rg -n 'CPU_PACKAGE_PROFILE|tuned-local-v3|cpu_profile_kind|cpu_profile' \
  docker .config/policies/cpu-packaging.toml website/docs/build/docker.md; then
  fail "Docker packaging still contains a profile branch"
fi

status=0
CPU_PACKAGE_TARGET_ROOT="${tmp_dir}/missing-target" FEATURES=nvml SKIP_CARGO_BUILD=1 \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/skip-without-provenance.output" 2>&1 || status=$?
[[ "${status}" -ne 0 ]] || fail "SKIP_CARGO_BUILD reused an artifact without canonical build record"
grep -Fq 'build-record' "${tmp_dir}/skip-without-provenance.output" \
  || fail "missing build record reuse diagnostic"

# Exercise the complete SKIP/stage/image-entry path with the fake canonical artifact,
# fake native libraries, and an isolated dist directory.
mkdir -p "${tmp_dir}/cudf/lib" "${tmp_dir}/native" \
  "${target_root}/x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-sys-fake/out/lib"
ln -s ../cudf "${tmp_dir}/native/cudf-install"
for library in libcudf.so librmm.so librapids_logger.so; do
  printf 'fake %s\n' "${library}" >"${tmp_dir}/cudf/lib/${library}"
done
printf 'fake cudf bridge\n' \
  >"${target_root}/x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-sys-fake/out/lib/libcudf_rust.so"
printf 'fake cudf RMM interop bridge\n' \
  >"${target_root}/x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-sys-fake/out/lib/libcudf_rust_rmm_interop.so"
cat >"${tmp_dir}/bin/patchelf" <<'SH'
#!/usr/bin/bash
set -euo pipefail
case "${1:-}" in
  --print-rpath) exit 0 ;;
  --set-rpath) exit 0 ;;
  *) exit 2 ;;
esac
SH
cat >"${tmp_dir}/bin/readelf" <<'SH'
#!/usr/bin/bash
set -euo pipefail
case "${1:-}" in
  -h) echo 'ELF Header:' ;;
  -d)
    echo ' 0x0000000000000001 (NEEDED) Shared library: [librmm.so]'
    [[ "${FAKE_LOGGER_NEEDED:-0}" != 1 ]] \
      || echo ' 0x0000000000000001 (NEEDED) Shared library: [librapids_logger.so]'
    [[ -z "${FAKE_READELF_NEEDED:-}" ]] \
      || echo " 0x0000000000000001 (NEEDED) Shared library: [${FAKE_READELF_NEEDED}]"
    ;;
  *) exit 2 ;;
esac
SH
cat >"${tmp_dir}/bin/ldd" <<'SH'
#!/usr/bin/bash
set -euo pipefail
echo "librmm.so => ${CPU_PACKAGE_DIST_DIR}/lib/librmm.so (0x0001)"
if [[ "$1" == */cugraph-install/lib/libcugraph.so && -n "${FAKE_GRAPH_CUVS_REALPATH:-}" ]]; then
  echo "libcuvs.so => ${FAKE_GRAPH_CUVS_REALPATH} (0x0002)"
elif [[ "${FAKE_CLOSURE_HAS_CUVS:-0}" == 1 ]]; then
  echo "libcuvs.so => ${CPU_PACKAGE_DIST_DIR}/lib/libcuvs.so (0x0002)"
fi
if [[ "${FAKE_CLOSURE_HAS_LOGGER:-0}" == 1 ]]; then
  logger_realpath="${FAKE_LOGGER_REALPATH:-${CPU_PACKAGE_DIST_DIR}/lib/librapids_logger.so}"
  if [[ "$1" == */lib/libcuvs.so && -n "${FAKE_SECOND_LOGGER_REALPATH:-}" ]]; then
    logger_realpath="${FAKE_SECOND_LOGGER_REALPATH}"
  fi
  echo "librapids_logger.so => ${logger_realpath} (0x0003)"
fi
[[ "${FAKE_LDD_AUXILIARY_NOT_FOUND:-0}" != 1 ]] || echo 'visibility=hidden => not found'
[[ "${FAKE_LDD_NOT_FOUND:-0}" != 1 ]] || echo 'libmissing.so => not found'
SH
chmod +x "${tmp_dir}/bin/patchelf" "${tmp_dir}/bin/readelf" "${tmp_dir}/bin/ldd"
cat >"${tmp_dir}/bin/docker" <<'SH'
#!/usr/bin/bash
set -euo pipefail
printf '%s\n' "$@" >"${FAKE_DOCKER_ARGS}"
SH
chmod +x "${tmp_dir}/bin/docker"

CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist" \
FEATURES=nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_LDD_AUXILIARY_NOT_FOUND=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/full-skip-stage.output"
[[ -f "${tmp_dir}/docker-dist/payload-manifest.json" ]] \
  || fail "full fake staging path omitted payload manifest"
[[ ! -e "${tmp_dir}/docker-dist/lib/librapids_logger.so" ]] \
  || fail "static/absorbed logger package staged an unneeded shared logger"
grep -Fq '==> SUCCESS: staging and validation passed.' "${tmp_dir}/full-skip-stage.output" \
  || fail "staging did not report success"
grep -Fq '==> NEXT: bash docker/build-image.sh' "${tmp_dir}/full-skip-stage.output" \
  || fail "staging did not point to the image build entry point"
if grep -Fq 'Canonical Docker command' "${tmp_dir}/full-skip-stage.output" || \
   grep -Fq -- '--build-arg' "${tmp_dir}/full-skip-stage.output"; then
  fail "staging still exposes the internal Docker build command"
fi
echo "PASS: staging reports a concise image-build next step"

FAKE_DOCKER_ARGS="${tmp_dir}/docker.args" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist" \
IMAGE_REPOSITORY=algeon-datafusion-test \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/build-image.sh >"${tmp_dir}/build-image.output"
grep -Fq '==> Building image: algeon-datafusion-test:x86-64-v4' "${tmp_dir}/build-image.output" \
  || fail "image builder omitted its build status"
grep -Fq '==> SUCCESS: Docker image built: algeon-datafusion-test:x86-64-v4' "${tmp_dir}/build-image.output" \
  || fail "image builder omitted its success status"
manifest_sha=$(sha256sum "${tmp_dir}/docker-dist/payload-manifest.json" | awk '{print $1}')
binary_sha=$(sha256sum "${tmp_dir}/docker-dist/bin/algeon_server" | awk '{print $1}')
for needle in \
  'build' \
  '--build-arg' \
  'CPU_BASELINE=x86-64-v4' \
  'CPU_ARCH=amd64' \
  'CPU_TARGET=x86_64-unknown-linux-gnu' \
  "CPU_BINARY_SHA256=${binary_sha}" \
  "CPU_MANIFEST_SHA256=${manifest_sha}" \
  '-t' \
  'algeon-datafusion-test:x86-64-v4'; do
  grep -Fxq -- "${needle}" "${tmp_dir}/docker.args" \
    || fail "image builder omitted Docker input: ${needle}"
done
echo "PASS: image builder verifies manifest and supplies Docker build inputs"

CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-shared-logger" \
FEATURES=nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_LOGGER_NEEDED=1 \
FAKE_CLOSURE_HAS_LOGGER=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/shared-logger-stage.output"
[[ -f "${tmp_dir}/docker-dist-shared-logger/lib/librapids_logger.so" ]] \
  || fail "shared logger DT_NEEDED edge did not stage the canonical logger"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-static-external-logger" \
FEATURES=nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_CLOSURE_HAS_LOGGER=1 \
FAKE_LOGGER_REALPATH="${tmp_dir}/cudf/lib/librapids_logger.so" \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/static-external-logger.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "static/absorbed logger package accepted an external shared logger"
grep -Fq 'without one canonical staged identity' "${tmp_dir}/static-external-logger.output" \
  || fail "external shared logger diagnostic missing"
echo "PASS: static and shared rapids_logger packaging contracts stay distinct"

mkdir -p "${tmp_dir}/cuvs/lib" "${tmp_dir}/cugraph/lib"
ln -s ../cuvs "${tmp_dir}/native/cuvs-install"
ln -s ../cugraph "${tmp_dir}/native/cugraph-install"
printf 'fake canonical cuvs\n' >"${tmp_dir}/cuvs/lib/libcuvs.so"
printf 'fake canonical cuvs C API\n' >"${tmp_dir}/cuvs/lib/libcuvs_c.so"
for library in libcugraph.so libcugraph_c.so libcugraph_common.so libcugraph_mg.so; do
  printf 'fake %s\n' "${library}" >"${tmp_dir}/cugraph/lib/${library}"
done

build_fake_variant() {
  local features=$1
  env -i PATH="${tmp_dir}/bin:${PATH}" HOME="${tmp_dir}/home" \
    python3 "${HELPER}" build \
      --features "${features}" \
      --target-root "${target_root}" \
      --manifest "${ROOT}/crates/algeon-server/Cargo.toml" \
      --explicit-env "ALGEON_NATIVE_DIR=${tmp_dir}/native" \
      >"${tmp_dir}/variant-build.output"
}

build_fake_variant cuvs
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-cuvs" \
FEATURES=cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_CLOSURE_HAS_CUVS=1 PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/cuvs-stage.output"
[[ -f "${tmp_dir}/docker-dist-cuvs/lib/libcuvs.so" ]] || fail "cuVS-only package omitted canonical libcuvs.so"
[[ -f "${tmp_dir}/docker-dist-cuvs/lib/libcuvs_c.so" ]] || fail "cuVS-only package omitted libcuvs_c.so"

build_fake_variant cugraph
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-cugraph" \
FEATURES=cugraph SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/cugraph-stage.output"
[[ -f "${tmp_dir}/docker-dist-cugraph/lib/libcuvs.so" ]] || fail "cuGraph-only package omitted canonical libcuvs.so"
[[ ! -e "${tmp_dir}/docker-dist-cugraph/lib/libcuvs_c.so" ]] || fail "cuGraph-only package unexpectedly staged libcuvs_c.so"

build_fake_variant cuvs,cugraph
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-combined" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/combined-stage.output"
[[ -f "${tmp_dir}/docker-dist-combined/lib/libcuvs.so" ]] || fail "combined package omitted canonical libcuvs.so"
[[ -f "${tmp_dir}/docker-dist-combined/lib/libcuvs_c.so" ]] || fail "combined package omitted libcuvs_c.so"
echo "PASS: cuVS-only, cuGraph-only, and combined packages stage the declared canonical cuVS libraries"

printf 'fake divergent logger\n' >"${tmp_dir}/divergent-librapids_logger.so"
status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-divergent-logger" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
FAKE_LOGGER_NEEDED=1 FAKE_CLOSURE_HAS_LOGGER=1 \
FAKE_SECOND_LOGGER_REALPATH="${tmp_dir}/divergent-librapids_logger.so" \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/divergent-logger.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "combined package accepted divergent librapids_logger.so identities"
grep -Fq 'multiple librapids_logger.so identities' "${tmp_dir}/divergent-logger.output" \
  || fail "divergent librapids_logger.so diagnostic missing"

printf 'fake divergent cuvs\n' >"${tmp_dir}/divergent-libcuvs.so"
status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-divergent" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/divergent-libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/divergent-cuvs.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "combined package accepted divergent libcuvs.so identity"
grep -Fq 'different libcuvs.so identities' "${tmp_dir}/divergent-cuvs.output" \
  || fail "divergent libcuvs.so diagnostic missing"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-not-found" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 FAKE_LDD_NOT_FOUND=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/loader-not-found.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "loader closure accepted a not-found dependency"
grep -Fq 'unresolved loader dependency' "${tmp_dir}/loader-not-found.output" \
  || fail "loader not-found diagnostic missing"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-raft" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
ALGEON_NATIVE_DIR="${tmp_dir}/native" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
FAKE_READELF_NEEDED=libraft.so PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/loader-raft.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "loader closure accepted a shared libraft.so"
grep -Fq 'unexpected shared RAFT dependency' "${tmp_dir}/loader-raft.output" \
  || fail "shared RAFT diagnostic missing"
echo "PASS: packaging rejects divergent cuVS/logger identities and incomplete/unexpected loader closure entries"

status=0
bash docker/stage-docker-payload.sh --config 'target.x86_64-unknown-linux-gnu.rustflags=[]' \
  >"${tmp_dir}/cli-override.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "stage-docker-payload.sh accepted an undeclared Cargo CLI override"
grep -Fq 'accepts no positional or Cargo arguments' "${tmp_dir}/cli-override.output" \
  || fail "missing CLI override diagnostic"
echo "PASS: Dockerfile/build script bind image labels to provenance"

echo "OK: CPU packaging policy tests passed without a release or Docker build."
