#!/usr/bin/env bash

ALGEON_CUDA_ARCHITECTURE_CONTRACT_RELATIVE_PATH="share/algeon-native/cuda-architectures.env"

algeon_cuda_architectures() {
    if [[ -n "${CUDF_CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
        echo "error: CUDF_CMAKE_CUDA_ARCHITECTURES is no longer a supported public selector." >&2
        echo "Set CMAKE_CUDA_ARCHITECTURES once for the complete cuDF -> cuVS -> cuGraph build." >&2
        return 1
    fi

    case "${CMAKE_CUDA_ARCHITECTURES:-NATIVE}" in
        native | NATIVE) printf '%s\n' "NATIVE" ;;
        *) printf '%s\n' "${CMAKE_CUDA_ARCHITECTURES}" ;;
    esac
}

algeon_cuda_contract_path() {
    printf '%s/%s\n' "$1" "${ALGEON_CUDA_ARCHITECTURE_CONTRACT_RELATIVE_PATH}"
}

# Real cubin architecture token grammar, shared with build_support.rs's
# parse_real_cuda_architectures: digits plus an optional single lowercase
# suffix letter (`120a` architecture-specific, `100f` family-specific).
ALGEON_CUDA_REAL_ARCHITECTURE_PATTERN='[0-9]+[a-z]?'

algeon_cuda_real_architectures() {
    local library="$1"
    local cuda_toolkit_root="${CUDAToolkit_ROOT:-${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}}"
    local cuobjdump="${cuda_toolkit_root}/bin/cuobjdump"

    if [[ ! -x "${cuobjdump}" ]]; then
        cuobjdump="$(command -v cuobjdump || true)"
    fi
    if [[ -z "${cuobjdump}" ]]; then
        echo "error: cuobjdump was not found under '${cuda_toolkit_root}/bin' or on PATH." >&2
        return 1
    fi
    if [[ ! -f "${library}" ]]; then
        echo "error: CUDA architecture inspection target does not exist: ${library}" >&2
        return 1
    fi

    local architectures
    # Large primary-library cubin groups are build targets; small groups can be embedded dependency stubs.
    architectures="$("${cuobjdump}" --list-elf "${library}" \
        | sed -n -E "s/.*[.]sm_(${ALGEON_CUDA_REAL_ARCHITECTURE_PATTERN})[.]cubin.*/\\1/p" \
        | LC_ALL=C sort \
        | uniq -c \
        | awk '{ count[$2] = $1; if ($1 > max) max = $1 } END { for (arch in count) if (count[arch] * 2 >= max) print arch }' \
        | LC_ALL=C sort \
        | paste -sd ';' -)"
    if [[ -z "${architectures}" ]]; then
        echo "error: '${library}' does not contain a detectable real CUDA cubin architecture." >&2
        return 1
    fi
    printf '%s\n' "${architectures}"
}

algeon_cuda_write_contract() {
    local root="$1"
    local library="$2"
    local requested="$3"
    local contract_path
    local contract_dir
    local contract_tmp
    local real_architectures

    contract_path="$(algeon_cuda_contract_path "${root}")"
    contract_dir="$(dirname -- "${contract_path}")"
    real_architectures="$(algeon_cuda_real_architectures "${library}")"
    mkdir -p -- "${contract_dir}"
    contract_tmp="$(mktemp "${contract_path}.tmp.XXXXXX")"
    printf '%s\n' \
        "requested=${requested}" \
        "real=${real_architectures}" >"${contract_tmp}"
    chmod 0644 -- "${contract_tmp}"
    mv -f -- "${contract_tmp}" "${contract_path}"
    printf '%s\n' "${contract_path}"
}

algeon_cuda_read_contract_value() {
    local contract_path="$1"
    local key="$2"
    local value

    value="$(sed -n "s/^${key}=//p" "${contract_path}")"
    if [[ -z "${value}" || "$(grep -c "^${key}=" "${contract_path}")" != "1" ]]; then
        echo "error: CUDA architecture contract '${contract_path}' has no unique ${key} value." >&2
        return 1
    fi
    printf '%s\n' "${value}"
}

algeon_cuda_require_contract() {
    local component="$1"
    local root="$2"
    local requested="$3"
    local rebuild_command="$4"
    local contract_path
    local installed_requested

    contract_path="$(algeon_cuda_contract_path "${root}")"
    if [[ ! -f "${contract_path}" ]]; then
        echo "error: ${component} is missing its CUDA architecture contract '${contract_path}'." >&2
        echo "Run with CMAKE_CUDA_ARCHITECTURES='${requested}': ${rebuild_command}" >&2
        return 1
    fi
    installed_requested="$(algeon_cuda_read_contract_value "${contract_path}" requested)"
    if [[ "${installed_requested}" != "${requested}" ]]; then
        echo "error: ${component} was built for CMAKE_CUDA_ARCHITECTURES='${installed_requested}'," >&2
        echo "but this build requested '${requested}'." >&2
        echo "Run with CMAKE_CUDA_ARCHITECTURES='${requested}': ${rebuild_command}" >&2
        return 1
    fi
}

algeon_cuda_require_matching_contracts() {
    local left_component="$1"
    local left_root="$2"
    local right_component="$3"
    local right_root="$4"
    local rebuild_command="$5"
    local left_contract
    local right_contract
    local left_real
    local right_real

    left_contract="$(algeon_cuda_contract_path "${left_root}")"
    right_contract="$(algeon_cuda_contract_path "${right_root}")"
    left_real="$(algeon_cuda_read_contract_value "${left_contract}" real)"
    right_real="$(algeon_cuda_read_contract_value "${right_contract}" real)"
    if [[ "${left_real}" != "${right_real}" ]]; then
        echo "error: ${left_component} contains full-kernel real CUDA cubins '${left_real}'," >&2
        echo "but ${right_component} contains '${right_real}'." >&2
        echo "Rebuild the native stack in dependency order: ${rebuild_command}" >&2
        return 1
    fi
}
