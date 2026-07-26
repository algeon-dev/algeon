#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
package_root="$(mktemp -d)"
trap 'rm -rf -- "${package_root}"' EXIT

for package in cuvs-nexus-sys cuvs-nexus; do
    list_file="${package_root}/${package}.list"
    cargo package \
        --manifest-path "${repo_root}/Cargo.toml" \
        -p "${package}" \
        --allow-dirty \
        --no-verify \
        --list >"${list_file}"
    grep -Fxq 'LICENSE' "${list_file}" || {
        echo "error: ${package} archive omits the workspace LICENSE" >&2
        exit 1
    }

    target_dir="${package_root}/${package}-target"
    package_command=(cargo package
        --manifest-path "${repo_root}/Cargo.toml" \
        -p "${package}" \
        --allow-dirty \
        --no-verify \
        --target-dir "${target_dir}")
    if [[ "${package}" == "cuvs-nexus" ]]; then
        package_command+=(
            --config "patch.crates-io.cudf-nexus.path='${repo_root}/../cudf-nexus/cudf-nexus'"
            --config "patch.crates-io.rapids-interop.path='${repo_root}/../cudf-nexus/rapids-interop'"
            --config "patch.crates-io.cuvs-nexus-sys.path='${repo_root}/cuvs-nexus-sys'"
        )
    fi
    "${package_command[@]}" >/dev/null
    archive="$(find "${target_dir}/package" -maxdepth 1 -type f -name "${package}-*.crate" -print -quit)"
    [[ -n "${archive}" ]] || {
        echo "error: cargo package did not create an archive for ${package}" >&2
        exit 1
    }
    extract_dir="${package_root}/${package}-extract"
    mkdir -p "${extract_dir}"
    tar -xzf "${archive}" -C "${extract_dir}"
    extracted="$(find "${extract_dir}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    [[ -f "${extracted}/LICENSE" ]] || {
        echo "error: extracted ${package} archive omits LICENSE" >&2
        exit 1
    }
done

sys_extract="$(find "${package_root}/cuvs-nexus-sys-extract" -mindepth 1 -maxdepth 1 -type d -print -quit)"
if rg -n 'include_str!\([^)]*cuvs-nexus/src' "${sys_extract}"; then
    echo "error: packaged sys tests depend on safe-crate source outside their archive" >&2
    exit 1
fi

echo "crate package/extract contract OK"
