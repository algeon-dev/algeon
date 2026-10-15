#!/usr/bin/env bash
# Usage: validate-loader-closure.sh --bin <server binary> --lib <library dir> --features <comma list>
#
# Checks the readelf NEEDED entries and the ldd closure of the binary and of
# every regular file under --lib, resolving --lib first. Any unresolved edge,
# a second librmm, librapids_logger, libcuvs or libcudart identity, a shared
# RAFT library, a libcudart soname other than libcudart.so.13, or a library
# missing for the selected features fails the check.

set -euo pipefail

usage() {
    echo "ERROR: usage: $0 --bin <path> --lib <dir> --features <comma list>" >&2
    exit 2
}

BIN=""
LIB_DIR=""
FEATURES=""
have_features=0
while [[ "$#" -gt 0 ]]; do
    [[ "$#" -ge 2 ]] || usage
    case "$1" in
        --bin) BIN="$2" ;;
        --lib) LIB_DIR="$2" ;;
        --features) FEATURES="$2"; have_features=1 ;;
        *) usage ;;
    esac
    shift 2
done
[[ -n "$BIN" && -n "$LIB_DIR" && $have_features -eq 1 ]] || usage
[[ -f "$BIN" ]] || { echo "ERROR: server binary $BIN does not exist." >&2; exit 1; }
[[ -d "$LIB_DIR" ]] || { echo "ERROR: library directory $LIB_DIR does not exist." >&2; exit 1; }

command -v readelf >/dev/null 2>&1 || { echo "ERROR: readelf is required for loader validation." >&2; exit 1; }
command -v ldd >/dev/null 2>&1 || { echo "ERROR: ldd is required for loader validation." >&2; exit 1; }

want_cugraph=0
want_cuvs=0
want_cuml=0
[[ ",$FEATURES," == *",cugraph,"* ]] && want_cugraph=1
[[ ",$FEATURES," == *",cuvs,"* ]] && want_cuvs=1
[[ ",$FEATURES," == *",cuml,"* ]] && want_cuml=1
want_cuvs_runtime=0
[[ $want_cugraph -eq 1 || $want_cuvs -eq 1 || $want_cuml -eq 1 ]] && want_cuvs_runtime=1

loader_path="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

resolve_needed_realpath() {
    local elf="$1" soname_prefix="$2" resolved
    resolved=$(LD_LIBRARY_PATH="$loader_path" ldd "$elf" 2>/dev/null | awk -v prefix="$soname_prefix" \
        '$1 ~ ("^" prefix "(\\.|$)") && $2 == "=>" && $3 ~ "^/" { print $3; exit }')
    [[ -n "$resolved" ]] || return 1
    readlink -f "$resolved"
}

declare -A cuvs_realpaths=() rmm_realpaths=() logger_realpaths=() cudart_realpaths=()
while IFS= read -r elf; do
    readelf -h "$elf" >/dev/null 2>&1 || {
        echo "ERROR: packaged runtime artifact is not a readable ELF file: $elf" >&2
        exit 1
    }
    dynamic=$(readelf -d "$elf" 2>&1) || {
        echo "ERROR: cannot inspect ELF dynamic section: $elf" >&2
        echo "$dynamic" >&2
        exit 1
    }
    while IFS= read -r needed; do
        case "$needed" in
            libraft.so*)
                echo "ERROR: unexpected shared RAFT dependency $needed in $elf; RAFT must remain header-only." >&2
                exit 1
                ;;
            libcudart.so*)
                [[ "$needed" == "libcudart.so.13" ]] || {
                    echo "ERROR: unexpected CUDA runtime dependency $needed in $elf; this package targets CUDA 13." >&2
                    exit 1
                }
                ;;
        esac
    done < <(awk -F'[][]' '/\(NEEDED\)/ { print $2 }' <<< "$dynamic")

    ldd_output=$(LD_LIBRARY_PATH="$loader_path" ldd "$elf" 2>&1) || {
        echo "ERROR: ldd failed for packaged ELF $elf" >&2
        echo "$ldd_output" >&2
        exit 1
    }
    # NVIDIA's libnvtx3interop uses DT_AUXILIARY [visibility=hidden]. glibc
    # ldd renders that marker as a `=> not found` row although it is not a
    # DT_NEEDED soname. Every other missing edge fails.
    unresolved=$(awk \
        '$1 != "visibility=hidden" && $2 == "=>" && $3 == "not" && $4 == "found"' \
        <<< "$ldd_output")
    if [[ -n "$unresolved" ]]; then
        echo "ERROR: unresolved loader dependency for packaged ELF $elf" >&2
        echo "$unresolved" >&2
        exit 1
    fi
    while read -r dependency _ resolved _; do
        [[ "$resolved" == /* ]] || continue
        realpath=$(readlink -f "$resolved") || {
            echo "ERROR: cannot canonicalize resolved dependency $resolved from $elf" >&2
            exit 1
        }
        case "$dependency" in
            libcuvs.so*) cuvs_realpaths["$realpath"]=1 ;;
            librmm.so*) rmm_realpaths["$realpath"]=1 ;;
            librapids_logger.so*) logger_realpaths["$realpath"]=1 ;;
            libcudart.so*) cudart_realpaths["$realpath"]=1 ;;
            libraft.so*)
                echo "ERROR: unexpected shared RAFT dependency resolved for $elf: $realpath" >&2
                exit 1
                ;;
        esac
    done <<< "$ldd_output"
done < <(printf '%s\n' "$BIN"; find "$LIB_DIR" -type f -print | sort)

if (( ${#rmm_realpaths[@]} != 1 )); then
    echo "ERROR: loader closure must resolve exactly one librmm.so identity; found ${#rmm_realpaths[@]}." >&2
    printf '  %s\n' "${!rmm_realpaths[@]}" >&2
    exit 1
fi
staged_rmm=$(readlink -f "$LIB_DIR/librmm.so")
[[ -n "${rmm_realpaths[$staged_rmm]+present}" ]] || {
    echo "ERROR: loader closure selected librmm.so outside the packaged canonical identity $staged_rmm." >&2
    exit 1
}

if (( ${#logger_realpaths[@]} > 1 )); then
    echo "ERROR: loader closure resolves multiple librapids_logger.so identities." >&2
    printf '  %s\n' "${!logger_realpaths[@]}" >&2
    exit 1
fi
declare -A staged_logger_realpaths=()
while IFS= read -r dependency; do
    realpath=$(readlink -f "$dependency") || {
        echo "ERROR: cannot canonicalize packaged rapids_logger $dependency" >&2
        exit 1
    }
    staged_logger_realpaths["$realpath"]=1
done < <(find "$LIB_DIR" -maxdepth 1 -name 'librapids_logger.so*' -print)
if (( ${#logger_realpaths[@]} == 1 )); then
    (( ${#staged_logger_realpaths[@]} == 1 )) || {
        echo "ERROR: loader closure resolved a shared librapids_logger.so without one canonical staged identity." >&2
        exit 1
    }
    for realpath in "${!logger_realpaths[@]}"; do
        [[ -n "${staged_logger_realpaths[$realpath]+present}" ]] || {
            echo "ERROR: loader closure selected librapids_logger.so outside the packaged canonical identity." >&2
            echo "  resolved: $realpath" >&2
            printf '  staged:   %s\n' "${!staged_logger_realpaths[@]}" >&2
            exit 1
        }
    done
elif (( ${#staged_logger_realpaths[@]} != 0 )); then
    echo "ERROR: packaged librapids_logger.so is absent from the loader closure; refusing an unneeded or stale shared logger." >&2
    exit 1
fi

if [[ $want_cuvs_runtime -eq 1 ]]; then
    if (( ${#cuvs_realpaths[@]} != 1 )); then
        echo "ERROR: loader closure must resolve exactly one libcuvs.so identity; found ${#cuvs_realpaths[@]}." >&2
        printf '  %s\n' "${!cuvs_realpaths[@]}" >&2
        exit 1
    fi
    staged_cuvs=$(readlink -f "$LIB_DIR/libcuvs.so")
    [[ -n "${cuvs_realpaths[$staged_cuvs]+present}" ]] || {
        echo "ERROR: loader closure selected libcuvs.so outside the packaged canonical identity $staged_cuvs." >&2
        exit 1
    }
elif (( ${#cuvs_realpaths[@]} != 0 )); then
    echo "ERROR: libcuvs.so entered a package without cugraph, cuvs, or cuml enabled." >&2
    exit 1
fi

if [[ $want_cuvs -eq 1 && ! -e "$LIB_DIR/libcuvs_c.so" ]]; then
    echo "ERROR: cuvs feature package is missing libcuvs_c.so." >&2
    exit 1
fi
if [[ $want_cuml -eq 1 ]]; then
    for name in libcuml libcuml_rust libtreelite libnvforest; do
        [[ -e "$LIB_DIR/$name.so" ]] || {
            echo "ERROR: cuml feature package is missing $name.so." >&2
            exit 1
        }
    done
    ml_cuvs=$(resolve_needed_realpath "$LIB_DIR/libcuml.so" libcuvs) || {
        echo "ERROR: packaged libcuml.so does not resolve libcuvs.so." >&2
        exit 1
    }
    [[ "$ml_cuvs" == "$(readlink -f "$LIB_DIR/libcuvs.so")" ]] || {
        echo "ERROR: packaged libcuml.so resolves a different libcuvs.so identity: $ml_cuvs" >&2
        exit 1
    }
fi
if (( ${#cudart_realpaths[@]} > 1 )); then
    echo "ERROR: loader closure resolves multiple CUDA runtime identities." >&2
    printf '  %s\n' "${!cudart_realpaths[@]}" >&2
    exit 1
fi

echo "OK: loader closure of $BIN with $LIB_DIR passed (features: ${FEATURES:-none})."
