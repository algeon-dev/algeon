#!/usr/bin/env bash
# Isolated CPU packaging/provenance tests. Uses fake cargo/rustc and fixtures;
# it does not compile the release server or build a Docker image.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "${ROOT}"

HELPER="${ROOT}/scripts/build/cpu_packaging_policy.py"
POLICY="${ROOT}/.config/policies/cpu-packaging.toml"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/datafusion-nexus-cpu-policy.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
mkdir -p "${tmp_dir}/bin" "${tmp_dir}/home"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

run_rejected_env() {
  local name=$1
  local assignment=$2
  local output="${tmp_dir}/${name}.output"
  local status=0
  env -i PATH="${PATH}" HOME="${tmp_dir}/home" "${assignment}" \
    python3 "${HELPER}" --policy "${POLICY}" check-env --profile portable \
    >"${output}" 2>&1 || status=$?
  [[ "${status}" -eq 2 ]] || fail "${name}: expected exit 2, got ${status}"
  grep -Fq "ambient codegen/config override variables are forbidden" "${output}" \
    || fail "${name}: missing rejection diagnostic"
  echo "PASS: reject ${name}"
}

# Presence of a codegen override is rejected wholesale. The fixtures include
# every unsafe spelling called out by the review, without teaching the policy
# a brittle denylist of CPU names/features.
run_rejected_env rustflags_v3 'RUSTFLAGS=-C target-cpu=x86-64-v3'
run_rejected_env rustflags_v4 'RUSTFLAGS=-C target-cpu=x86-64-v4'
run_rejected_env named_microarchitecture 'RUSTFLAGS=-C target-cpu=haswell'
run_rejected_env positive_avx 'RUSTFLAGS=-C target-feature=+avx2'
run_rejected_env positive_aes 'RUSTFLAGS=-C target-feature=+aes'
run_rejected_env cargo_build_rustflags 'CARGO_BUILD_RUSTFLAGS=-C target-cpu=native'
run_rejected_env target_specific_cargo 'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS=-C target-cpu=x86-64-v3'
run_rejected_env target_specific_cflags 'CFLAGS_x86_64_unknown_linux_gnu=-march=x86-64-v3'
run_rejected_env target_specific_cxxflags 'x86_64_unknown_linux_gnu_CXXFLAGS=-march=native'
run_rejected_env cargo_build_target 'CARGO_BUILD_TARGET=x86_64-unknown-linux-musl'
run_rejected_env encoded_rustflags "CARGO_ENCODED_RUSTFLAGS=-C"$'\x1f'"target-feature=+aes"

if rg -n 'ensure_cpu_baseline|is_x86_feature_detected!' crates/nexus-server/src/bin/datafusion_nexus_server.rs; then
  fail "server still contains an in-process CPU feature check"
fi
echo "PASS: server has no in-process CPU feature detection"

python3 - "${POLICY}" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as policy_file:
    policy = tomllib.load(policy_file)
assert policy["clean_environment"]["explicit_inputs"] == [
    "CUDF_INSTALL_DIR",
    "CUVS_INSTALL_DIR",
    "CUVS_ROOT",
    "CUGRAPH_CMAKE_BUILD_DIR",
]
portable = policy["profiles"]["portable"]
tuned = policy["profiles"]["tuned-local-v3"]
assert portable["declared_baseline"] == "x86-64-v4"
assert portable["rustflags"] == ["-C", "target-cpu=x86-64-v4"]
assert tuned["declared_baseline"] == "x86-64-v3"
assert tuned["artifact_dir"] != portable["artifact_dir"]
assert tuned["image_tag_suffix"] != portable["image_tag_suffix"]
assert "x86-64-v3" in tuned["deployment_requirement"]
PY
echo "PASS: portable and tuned profiles are separate declarative contracts"

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
printf 'fake canonical server\n' >"${target_dir}/${target}/${profile}/datafusion_nexus_server"
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
  python3 "${HELPER}" --policy "${POLICY}" build \
    --profile portable \
    --features 'nvml,iceberg' \
    --target-root "${target_root}" \
    --manifest "${ROOT}/crates/nexus-server/Cargo.toml" \
    --explicit-env "CUDF_INSTALL_DIR=${tmp_dir}/cudf" \
    --explicit-env "CUVS_INSTALL_DIR=${tmp_dir}/cuvs" \
    --explicit-env "CUVS_ROOT=${tmp_dir}/cuvs" \
    --explicit-env "CUGRAPH_CMAKE_BUILD_DIR=${tmp_dir}/cugraph" \
    >"${tmp_dir}/build.output"

binary="${target_root}/portable-x86-64-v4/x86_64-unknown-linux-gnu/release/datafusion_nexus_server"
build_provenance="${binary}.cpu-provenance.json"
[[ -f "${binary}" && -f "${build_provenance}" ]] || fail "fake canonical build did not create artifact/provenance"

python3 - "${build_provenance}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as provenance_file:
    provenance = json.load(provenance_file)
profile = provenance["cpu_profile"]
command = provenance["cargo_command"]
assert provenance["provenance_kind"] == "canonical-build-artifact"
assert provenance["features"] == ["iceberg", "nvml"]
assert profile["declared_baseline"] == "x86-64-v4"
assert profile["target_triple"] == "x86_64-unknown-linux-gnu"
assert profile["effective_rustflags"] == ["-C", "target-cpu=x86-64-v4"]
assert provenance["rustc_version"].startswith("rustc 1.95.0")
assert "LLVM version: 22.0.0" in provenance["rustc_vv"]
assert provenance["cargo_version"].startswith("cargo 1.95.0")
assert provenance["fixed_build_environment"] == {
    "AR": "/usr/bin/ar",
    "CC": "/usr/bin/gcc-14",
    "CMAKE": "/usr/bin/cmake",
    "CMAKE_GENERATOR": "Ninja",
    "CUDAHOSTCXX": "/usr/bin/g++-14",
    "CXX": "/usr/bin/g++-14",
}
assert set(provenance["fixed_tool_versions"]) == {"ar", "cc", "cmake", "cxx", "ninja"}
assert all(provenance["fixed_tool_versions"].values())
assert "--target" in command and "x86_64-unknown-linux-gnu" in command
assert "target.x86_64-unknown-linux-gnu.rustflags=[\"-C\",\"target-cpu=x86-64-v4\"]" in command
assert "--no-default-features" in command and "--locked" in command
PY
echo "PASS: ambient compiler selectors are replaced and provenance records fixed tool inputs"
echo "PASS: canonical command and build provenance record exact policy/tool inputs"

env -i PATH="${tmp_dir}/bin:${PATH}" HOME="${tmp_dir}/home" \
  python3 "${HELPER}" --policy "${POLICY}" verify-build \
    --profile portable --features 'iceberg,nvml' \
    --binary "${binary}" --provenance "${build_provenance}" \
    >"${tmp_dir}/reuse.output"
echo "PASS: verified canonical artifact can be reused"

cp "${binary}" "${tmp_dir}/binary.backup"
printf 'tamper\n' >>"${binary}"
status=0
python3 "${HELPER}" --policy "${POLICY}" verify-build \
  --profile portable --features 'iceberg,nvml' \
  --binary "${binary}" --provenance "${build_provenance}" \
  >"${tmp_dir}/tampered-binary.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "tampered canonical artifact was accepted"
grep -Fq 'artifact hash mismatch' "${tmp_dir}/tampered-binary.output" || fail "missing artifact tamper diagnostic"
cp "${tmp_dir}/binary.backup" "${binary}"
echo "PASS: artifact tamper prevents SKIP_CARGO_BUILD reuse"

cp "${POLICY}" "${tmp_dir}/tampered-policy.toml"
python3 - "${tmp_dir}/tampered-policy.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace('deployment_requirement = "The host CPU', 'deployment_requirement = "CHANGED: The host CPU', 1))
PY
status=0
python3 "${HELPER}" --policy "${tmp_dir}/tampered-policy.toml" verify-build \
  --profile portable --features 'iceberg,nvml' \
  --binary "${binary}" --provenance "${build_provenance}" \
  >"${tmp_dir}/tampered-policy.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "changed policy was accepted for artifact reuse"
grep -Fq 'policy_sha256' "${tmp_dir}/tampered-policy.output" || fail "missing policy tamper diagnostic"
echo "PASS: policy tamper prevents artifact reuse"

staged_binary="${tmp_dir}/dist/bin/datafusion_nexus_server"
mkdir -p "$(dirname "${staged_binary}")"
cp "${binary}" "${staged_binary}"
printf 'post-stage-patchelf-fixture\n' >>"${staged_binary}"
stage_provenance="${tmp_dir}/dist/cpu-provenance.json"
stage_labels="${tmp_dir}/dist/cpu-provenance.labels"
python3 "${HELPER}" --policy "${POLICY}" stage \
  --profile portable --features 'iceberg,nvml' \
  --source-binary "${binary}" --staged-binary "${staged_binary}" \
  --build-provenance "${build_provenance}" \
  --output "${stage_provenance}" --labels-output "${stage_labels}" \
  >"${tmp_dir}/stage.output"

binary_sha=$(awk -F= '$1 == "binary_sha256" { print $2 }' "${stage_labels}")
provenance_sha=$(sha256sum "${stage_provenance}" | awk '{print $1}')
python3 "${HELPER}" verify-image-inputs \
  --provenance "${stage_provenance}" --labels "${stage_labels}" \
  --baseline x86-64-v4 --arch amd64 --profile portable --profile-kind redistributable \
  --target x86_64-unknown-linux-gnu \
  --binary-sha256 "${binary_sha}" --provenance-sha256 "${provenance_sha}" \
  >"${tmp_dir}/image-inputs.output"
echo "PASS: staged provenance and Docker label inputs agree"

status=0
python3 "${HELPER}" verify-image-inputs \
  --provenance "${stage_provenance}" --labels "${stage_labels}" \
  --baseline x86-64-v3 --arch amd64 --profile portable --profile-kind redistributable \
  --target x86_64-unknown-linux-gnu \
  --binary-sha256 "${binary_sha}" --provenance-sha256 "${provenance_sha}" \
  >"${tmp_dir}/tampered-label-input.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "mismatched Docker baseline build arg was accepted"
grep -Fq 'Docker build input mismatch' "${tmp_dir}/tampered-label-input.output" || fail "missing Docker tamper diagnostic"
echo "PASS: mismatched Docker label/build input fails"

status=0
python3 "${HELPER}" verify-image-inputs \
  --provenance "${stage_provenance}" --labels "${stage_labels}" \
  --baseline x86-64-v4 --arch amd64 --profile portable --profile-kind tuned-local \
  --target x86_64-unknown-linux-gnu --binary-sha256 "${binary_sha}" \
  --provenance-sha256 "${provenance_sha}" \
  >"${tmp_dir}/tampered-kind-input.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "mismatched Docker profile kind build arg was accepted"
grep -Fq 'cpu_profile_kind' "${tmp_dir}/tampered-kind-input.output" || fail "missing profile-kind tamper diagnostic"
echo "PASS: mismatched Docker profile kind fails"

for needle in \
  'org.datafusion-nexus.cpu-baseline="${CPU_BASELINE}"' \
  'org.datafusion-nexus.cpu-arch="${CPU_ARCH}"' \
  'org.datafusion-nexus.cpu-profile="${CPU_PROFILE}"' \
  'org.datafusion-nexus.cpu-profile-kind="${CPU_PROFILE_KIND}"' \
  'org.datafusion-nexus.cpu-target="${CPU_TARGET}"' \
  'cpu_baseline=${CPU_BASELINE}' \
  'CPU_PROVENANCE_SHA256' \
  'sha256sum -c -'; do
  grep -Fq "${needle}" docker/Dockerfile || fail "Dockerfile missing provenance assertion: ${needle}"
done
grep -Fq 'verify-build' docker/stage-docker-payload.sh || fail "SKIP_CARGO_BUILD path does not verify provenance"
grep -Fq 'CPU_PACKAGE_PROFILE="${CPU_PACKAGE_PROFILE:-portable}"' docker/stage-docker-payload.sh \
  || fail "staging script does not select a packaging profile"
grep -Fq 'CPU_PACKAGE_PROFILE=tuned-local-v3 bash docker/stage-docker-payload.sh' website/docs/build/docker.md \
  || fail "tuned profile is not exposed in the Docker build guide"
status=0
env RUSTFLAGS='-C target-cpu=x86-64-v3' bash docker/stage-docker-payload.sh \
  >"${tmp_dir}/build-script-dirty-env.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "canonical stage-docker-payload.sh did not reject ambient RUSTFLAGS"
grep -Fq 'ambient codegen/config override variables are forbidden' "${tmp_dir}/build-script-dirty-env.output" \
  || fail "canonical build path omitted the clean-environment diagnostic"

status=0
CPU_PACKAGE_TARGET_ROOT="${tmp_dir}/missing-target" FEATURES=iceberg SKIP_CARGO_BUILD=1 \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/skip-without-provenance.output" 2>&1 || status=$?
[[ "${status}" -eq 2 ]] || fail "SKIP_CARGO_BUILD reused an artifact without canonical provenance"
grep -Fq 'cannot load provenance' "${tmp_dir}/skip-without-provenance.output" \
  || fail "missing provenance reuse diagnostic"

# Exercise the complete SKIP/stage/image-entry path with the fake canonical artifact,
# fake native libraries, and an isolated dist directory.
mkdir -p "${tmp_dir}/cudf/lib" \
  "${target_root}/portable-x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-nexus-sys-fake/out/lib"
for library in libcudf.so librmm.so librapids_logger.so; do
  printf 'fake %s\n' "${library}" >"${tmp_dir}/cudf/lib/${library}"
done
printf 'fake cudf bridge\n' \
  >"${target_root}/portable-x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-nexus-sys-fake/out/lib/libcudf_rust.so"
printf 'fake cudf RMM interop bridge\n' \
  >"${target_root}/portable-x86-64-v4/x86_64-unknown-linux-gnu/release/build/cudf-nexus-sys-fake/out/lib/libcudf_rust_rmm_interop.so"
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
if [[ "$1" == */cugraph/libcugraph.so && -n "${FAKE_GRAPH_CUVS_REALPATH:-}" ]]; then
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
FEATURES=iceberg,nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_LDD_AUXILIARY_NOT_FOUND=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/full-skip-stage.output"
[[ -f "${tmp_dir}/docker-dist/cpu-provenance.json" ]] \
  || fail "full fake staging path omitted provenance"
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
IMAGE_REPOSITORY=datafusion-nexus-test \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/build-image.sh >"${tmp_dir}/build-image.output"
grep -Fq '==> Building image: datafusion-nexus-test:x86-64-v4' "${tmp_dir}/build-image.output" \
  || fail "image builder omitted its build status"
grep -Fq '==> SUCCESS: Docker image built: datafusion-nexus-test:x86-64-v4' "${tmp_dir}/build-image.output" \
  || fail "image builder omitted its success status"
binary_sha=$(awk -F= '$1 == "binary_sha256" { print $2 }' "${tmp_dir}/docker-dist/cpu-provenance.labels")
provenance_sha=$(sha256sum "${tmp_dir}/docker-dist/cpu-provenance.json" | awk '{print $1}')
for needle in \
  'build' \
  '--build-arg' \
  'CPU_BASELINE=x86-64-v4' \
  'CPU_PROFILE=portable' \
  "CPU_BINARY_SHA256=${binary_sha}" \
  "CPU_PROVENANCE_SHA256=${provenance_sha}" \
  '-t' \
  'datafusion-nexus-test:x86-64-v4'; do
  grep -Fxq -- "${needle}" "${tmp_dir}/docker.args" \
    || fail "image builder omitted Docker input: ${needle}"
done
echo "PASS: image builder verifies provenance and hides Docker build inputs"

CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-shared-logger" \
FEATURES=iceberg,nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_LOGGER_NEEDED=1 \
FAKE_CLOSURE_HAS_LOGGER=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/shared-logger-stage.output"
[[ -f "${tmp_dir}/docker-dist-shared-logger/lib/librapids_logger.so" ]] \
  || fail "shared logger DT_NEEDED edge did not stage the canonical logger"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-static-external-logger" \
FEATURES=iceberg,nvml \
SKIP_CARGO_BUILD=1 \
NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_CLOSURE_HAS_LOGGER=1 \
FAKE_LOGGER_REALPATH="${tmp_dir}/cudf/lib/librapids_logger.so" \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/static-external-logger.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "static/absorbed logger package accepted an external shared logger"
grep -Fq 'without one canonical staged identity' "${tmp_dir}/static-external-logger.output" \
  || fail "external shared logger diagnostic missing"
echo "PASS: static and shared rapids_logger packaging contracts stay distinct"

mkdir -p "${tmp_dir}/cuvs/lib" "${tmp_dir}/cugraph"
printf 'fake canonical cuvs\n' >"${tmp_dir}/cuvs/lib/libcuvs.so"
printf 'fake canonical cuvs C API\n' >"${tmp_dir}/cuvs/lib/libcuvs_c.so"
for library in libcugraph.so libcugraph_c.so libcugraph_common.so libcugraph_mg.so; do
  printf 'fake %s\n' "${library}" >"${tmp_dir}/cugraph/${library}"
done

build_fake_variant() {
  local features=$1
  env -i PATH="${tmp_dir}/bin:${PATH}" HOME="${tmp_dir}/home" \
    python3 "${HELPER}" --policy "${POLICY}" build \
      --profile portable \
      --features "${features}" \
      --target-root "${target_root}" \
      --manifest "${ROOT}/crates/nexus-server/Cargo.toml" \
      --explicit-env "CUDF_INSTALL_DIR=${tmp_dir}/cudf" \
      --explicit-env "CUVS_INSTALL_DIR=${tmp_dir}/cuvs" \
      --explicit-env "CUVS_ROOT=${tmp_dir}/cuvs" \
      --explicit-env "CUGRAPH_CMAKE_BUILD_DIR=${tmp_dir}/cugraph" \
      >"${tmp_dir}/variant-build.output"
}

build_fake_variant cuvs
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-cuvs" \
FEATURES=cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
FAKE_CLOSURE_HAS_CUVS=1 PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/cuvs-stage.output"
[[ -f "${tmp_dir}/docker-dist-cuvs/lib/libcuvs.so" ]] || fail "cuVS-only package omitted canonical libcuvs.so"
[[ -f "${tmp_dir}/docker-dist-cuvs/lib/libcuvs_c.so" ]] || fail "cuVS-only package omitted libcuvs_c.so"

build_fake_variant cugraph
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-cugraph" \
FEATURES=cugraph SKIP_CARGO_BUILD=1 NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/cugraph-stage.output"
[[ -f "${tmp_dir}/docker-dist-cugraph/lib/libcuvs.so" ]] || fail "cuGraph-only package omitted canonical libcuvs.so"
[[ ! -e "${tmp_dir}/docker-dist-cugraph/lib/libcuvs_c.so" ]] || fail "cuGraph-only package unexpectedly staged libcuvs_c.so"

build_fake_variant cugraph,cuvs
CPU_PACKAGE_TARGET_ROOT="${target_root}" \
CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-combined" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
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
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
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
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/divergent-libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/divergent-cuvs.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "combined package accepted divergent libcuvs.so identity"
grep -Fq 'different libcuvs.so identities' "${tmp_dir}/divergent-cuvs.output" \
  || fail "divergent libcuvs.so diagnostic missing"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-not-found" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
FAKE_GRAPH_CUVS_REALPATH="${tmp_dir}/cuvs/lib/libcuvs.so" FAKE_CLOSURE_HAS_CUVS=1 FAKE_LDD_NOT_FOUND=1 \
PATH="${tmp_dir}/bin:${PATH}" \
  bash docker/stage-docker-payload.sh >"${tmp_dir}/loader-not-found.output" 2>&1 || status=$?
[[ "${status}" -eq 1 ]] || fail "loader closure accepted a not-found dependency"
grep -Fq 'unresolved loader dependency' "${tmp_dir}/loader-not-found.output" \
  || fail "loader not-found diagnostic missing"

status=0
CPU_PACKAGE_TARGET_ROOT="${target_root}" CPU_PACKAGE_DIST_DIR="${tmp_dir}/docker-dist-raft" \
FEATURES=cugraph,cuvs SKIP_CARGO_BUILD=1 NO_STRIP=1 \
CUDF_INSTALL_DIR="${tmp_dir}/cudf" CUVS_INSTALL_DIR="${tmp_dir}/cuvs" \
CUGRAPH_CMAKE_BUILD_DIR="${tmp_dir}/cugraph" \
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
