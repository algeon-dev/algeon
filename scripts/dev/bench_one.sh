#!/usr/bin/env bash
# One-command single-query TPC-H / TPC-DS bench for humans.
#
# The raw bench TSVs are machine format (100+ columns); this wrapper resolves
# the dataset env, runs one bench binary for the requested queries, keeps the
# TSV clean (stdout only: progress stays on stderr), and prints a compact
# human-readable summary afterwards.
#
# Usage:
#   bash scripts/dev/bench_one.sh tpch q06
#   bash scripts/dev/bench_one.sh tpch q06,q12 --scale 10
#   bash scripts/dev/bench_one.sh tpcds q42 --engine native_gpu
#   bash scripts/dev/bench_one.sh tpcds q42 --engine cpu_datafusion
#   bash scripts/dev/bench_one.sh {tpch|tpcds} --help
#
# Query ids accept the canonical zero-padded form (q06, q42) as well as the
# short form (q6 → q06). "all" runs every query (tpcds passes it through to
# the bench binary; tpch expands to q01..q22).
#
# Flags (a flag wins when given, otherwise an already-exported env value wins,
# otherwise the default below applies):
#   --scale 1|10|100            dataset scale factor (default: 1;
#                               sf1/sf10/sf100 also accepted)
#   --warmups N                 warmup iterations (default: 0)
#   --iterations N              measured iterations (default: 1)
#   --engine E                  tpcds only: native_gpu (default) |
#                               cpu_datafusion | cost_declined_datafusion
#   --width N                   tpch only: review table width (default: 0 =
#                               auto-detect; the wide table prints only when it
#                               fits, e.g. --width 80 forces the compact view)
#   --no-pretty                 tpch only: plain-text review even when rich is
#                               available (default: rich tables on a terminal)
#   --explore                   open the resulting TSV in visidata for a
#                               full-column interactive look (needs a terminal)
#   --parquet-dir PATH          override dataset auto-detection
#   --out PATH                  TSV output path (default: /tmp/bench-one/...)
#   --dev                       use the dev (debug) profile; it compiles faster, but
#                               its timings are meaningless for tuning (default: release)
#   --no-review                 skip the human-readable summary
#   --dry-run                   print the resolved env + command without running
#
# Dataset auto-detection order: --parquet-dir, $ALGEON_PARQUET_DIR,
# /scratch/<tpch|tpcds>/sf<scale>/parquet, then the in-repo fixture root.
# Set ALGEON_BENCH_ONE_NO_ENV=1 to skip sourcing env.sh.

set -euo pipefail

usage() {
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

WORKLOAD="${1:-}"
if [[ "${WORKLOAD}" == "-h" || "${WORKLOAD}" == "--help" || -z "${WORKLOAD}" ]]; then
    usage
    exit 0
fi
if [[ "${WORKLOAD}" != "tpch" && "${WORKLOAD}" != "tpcds" ]]; then
    echo "ERROR: unknown workload '${WORKLOAD}'; expected tpch or tpcds" >&2
    exit 2
fi
shift

QUERIES_RAW=""
SCALE_FLAG=""
WARMUPS_FLAG=""
ITERATIONS_FLAG=""
ENGINE_FLAG=""
WIDTH_FLAG=""
PARQUET_DIR_FLAG=""
OUT_FLAG=""
NO_REVIEW=0
NO_PRETTY=0
EXPLORE=0
DEV_PROFILE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scale) SCALE_FLAG="${2:?missing value for --scale}"; shift 2 ;;
        --scale=*) SCALE_FLAG="${1#*=}"; shift ;;
        --warmups) WARMUPS_FLAG="${2:?missing value for --warmups}"; shift 2 ;;
        --warmups=*) WARMUPS_FLAG="${1#*=}"; shift ;;
        --iterations) ITERATIONS_FLAG="${2:?missing value for --iterations}"; shift 2 ;;
        --iterations=*) ITERATIONS_FLAG="${1#*=}"; shift ;;
        --engine) ENGINE_FLAG="${2:?missing value for --engine}"; shift 2 ;;
        --engine=*) ENGINE_FLAG="${1#*=}"; shift ;;
        --width) WIDTH_FLAG="${2:?missing value for --width}"; shift 2 ;;
        --width=*) WIDTH_FLAG="${1#*=}"; shift ;;
        --no-pretty) NO_PRETTY=1; shift ;;
        --explore) EXPLORE=1; shift ;;
        --dev) DEV_PROFILE=1; shift ;;
        --parquet-dir) PARQUET_DIR_FLAG="${2:?missing value for --parquet-dir}"; shift 2 ;;
        --parquet-dir=*) PARQUET_DIR_FLAG="${1#*=}"; shift ;;
        --out) OUT_FLAG="${2:?missing value for --out}"; shift 2 ;;
        --out=*) OUT_FLAG="${1#*=}"; shift ;;
        --no-review) NO_REVIEW=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; break ;;
        -*) echo "ERROR: unknown flag $1" >&2; exit 2 ;;
        *)
            if [[ -n "${QUERIES_RAW}" ]]; then
                echo "ERROR: unexpected extra argument $1" >&2
                exit 2
            fi
            QUERIES_RAW="$1"
            shift
            ;;
    esac
done

if [[ -z "${QUERIES_RAW}" ]]; then
    echo "ERROR: missing query ids; e.g. bash scripts/dev/bench_one.sh ${WORKLOAD} q06" >&2
    exit 2
fi
if [[ -n "${ENGINE_FLAG}" && "${WORKLOAD}" != "tpcds" ]]; then
    echo "ERROR: --engine is tpcds-only" >&2
    exit 2
fi
if [[ -n "${WIDTH_FLAG}" && "${WORKLOAD}" != "tpch" ]]; then
    echo "ERROR: --width is tpch-only" >&2
    exit 2
fi
if [[ -n "${WIDTH_FLAG}" && ! "${WIDTH_FLAG}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --width must be a non-negative integer; got '${WIDTH_FLAG}'" >&2
    exit 2
fi
if [[ "${NO_PRETTY}" -eq 1 && "${WORKLOAD}" != "tpch" ]]; then
    echo "ERROR: --no-pretty is tpch-only" >&2
    exit 2
fi

# Flag wins, else already-exported env, else default.
pick() {
    if [[ -n "$2" ]]; then
        printf '%s' "$2"
    elif [[ -n "${!1:-}" ]]; then
        printf '%s' "${!1}"
    else
        printf '%s' "$3"
    fi
}

# Accept 1, 10, 100 as well as sf1, sf10, sf100 (any case).
normalize_scale() {
    local raw="$1"
    raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]' | tr -d '[:blank:]')"
    raw="${raw#sf}"
    printf '%s' "${raw}"
}
SCALE="$(normalize_scale "$(pick ALGEON_BENCH_ONE_SCALE "${SCALE_FLAG}" 1)")"
WARMUPS="$(pick ALGEON_BENCH_ONE_WARMUPS "${WARMUPS_FLAG}" 0)"
ITERATIONS="$(pick ALGEON_BENCH_ONE_ITERATIONS "${ITERATIONS_FLAG}" 1)"
ENGINE="$(pick ALGEON_TPCDS_BENCH_ENGINE "${ENGINE_FLAG}" native_gpu)"

if [[ "${SCALE}" != "1" && "${SCALE}" != "10" && "${SCALE}" != "100" ]]; then
    echo "ERROR: --scale must be 1, 10, or 100 (sf1/sf10/sf100 also accepted); got '${SCALE}'" >&2
    exit 2
fi
if [[ "${WORKLOAD}" == "tpcds" ]]; then
    case "${ENGINE}" in
        native_gpu | cpu_datafusion | cost_declined_datafusion) ;;
        *) echo "ERROR: --engine must be native_gpu, cpu_datafusion, or cost_declined_datafusion" >&2; exit 2 ;;
    esac
fi

# Normalize query ids: lowercase, strip blanks, pad q6 -> q06.
# "all" expands to q01..q22 for tpch (the bench binary needs explicit ids);
# tpcds passes "all" through (the binary understands it).
normalize_queries() {
    local workload="$1" raw="$2"
    raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]' | tr -d '[:blank:]')"
    if [[ "${raw}" == "all" ]]; then
        if [[ "${workload}" == "tpch" ]]; then
            seq -f 'q%02g' 1 22 | paste -sd, -
        else
            printf 'all'
        fi
        return
    fi
    local out="" part num
    local IFS=','
    local -a parts
    read -ra parts <<< "${raw}"
    for part in "${parts[@]}"; do
        if [[ "${part}" =~ ^q([0-9]{1,2})$ ]]; then
            num="${BASH_REMATCH[1]}"
            part="$(printf 'q%02d' "$((10#${num}))")"
        fi
        if [[ -n "${part}" ]]; then
            out="${out:+${out},}${part}"
        fi
    done
    if [[ -z "${out}" ]]; then
        echo "ERROR: no query ids parsed from '${QUERIES_RAW}'" >&2
        exit 2
    fi
    printf '%s' "${out}"
}
QUERIES="$(normalize_queries "${WORKLOAD}" "${QUERIES_RAW}")"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${ALGEON_BENCH_ONE_NO_ENV:-0}" != "1" && -f "${REPO_ROOT}/env.sh" ]]; then
    # Provides CUDA_HOME, CC/CXX, and the checkout-local native lib paths.
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/env.sh"
fi

# Dataset auto-detection: explicit flag, exported env, /scratch, fixture.
resolve_parquet_dir() {
    local workload="$1" scale="$2"
    if [[ -n "${PARQUET_DIR_FLAG}" ]]; then
        printf '%s' "${PARQUET_DIR_FLAG}"
        return
    fi
    if [[ -n "${ALGEON_PARQUET_DIR:-}" ]]; then
        printf '%s' "${ALGEON_PARQUET_DIR}"
        return
    fi
    local scratch="/scratch/${workload}/sf${scale}/parquet"
    if [[ -d "${scratch}" ]]; then
        printf '%s' "${scratch}"
        return
    fi
    local fixture=""
    if [[ "${workload}" == "tpch" ]]; then
        if [[ "${scale}" == "100" ]]; then
            fixture="${REPO_ROOT}/fixture/tpc-h/data/sf100_optimized/parquet"
        else
            fixture="${REPO_ROOT}/fixture/tpc-h/data/sf${scale}/parquet"
        fi
    else
        fixture="${REPO_ROOT}/fixture/tpc-ds/data/sf${scale}/parquet"
    fi
    printf '%s' "${fixture}"
}
# Review rendering prefers the project venv python when it can import rich;
# otherwise plain `python3` (the review script falls back to plain text).
REVIEW_PY="python3"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]] \
    && "${REPO_ROOT}/.venv/bin/python" -c "import rich" > /dev/null 2>&1; then
    REVIEW_PY="${REPO_ROOT}/.venv/bin/python"
fi

# Interactive TSV explorer for --explore and the post-review hint.
resolve_vd() {
    if [[ -x "${REPO_ROOT}/.venv/bin/vd" ]]; then
        printf '%s' "${REPO_ROOT}/.venv/bin/vd"
    elif command -v vd > /dev/null 2>&1; then
        command -v vd
    fi
}

PARQUET_DIR="$(resolve_parquet_dir "${WORKLOAD}" "${SCALE}")"
if [[ ! -d "${PARQUET_DIR}" ]]; then
    echo "ERROR: parquet dir not found: ${PARQUET_DIR}" >&2
    echo "       Generate it (fixture/fixture.sh ${WORKLOAD} generate --scale ${SCALE}) or pass --parquet-dir PATH" >&2
    exit 1
fi
if ! compgen -G "${PARQUET_DIR}/*.parquet" > /dev/null; then
    echo "ERROR: no *.parquet files in ${PARQUET_DIR}" >&2
    exit 1
fi
export ALGEON_PARQUET_DIR="${PARQUET_DIR}"

if [[ "${WORKLOAD}" == "tpch" ]]; then
    BIN="native_tpch_benchmark"
    export ALGEON_TPCH_BENCH_QUERY_IDS="${QUERIES}"
    export ALGEON_TPCH_BENCH_DATASET_MODE="sf${SCALE}_external"
    export ALGEON_TPCH_SCALE_FACTOR="${ALGEON_TPCH_SCALE_FACTOR:-sf${SCALE}}"
    export ALGEON_TPCH_GENERATION_COMMAND="${ALGEON_TPCH_GENERATION_COMMAND:-fixture/fixture.sh tpch generate --scale ${SCALE}}"
    export ALGEON_TPCH_BENCH_WARMUPS="${WARMUPS}"
    export ALGEON_TPCH_BENCH_ITERATIONS="${ITERATIONS}"
    NEEDS_LOCK=1
else
    case "${ENGINE}" in
        native_gpu) BIN="tpcds_three_way_bench"; NEEDS_LOCK=1 ;;
        cpu_datafusion | cost_declined_datafusion) BIN="tpcds_three_way_bench"; NEEDS_LOCK=0 ;;
    esac
    export ALGEON_TPCDS_BENCH_ENGINE="${ENGINE}"
    export ALGEON_TPCDS_BENCH_QUERIES="${QUERIES}"
    export ALGEON_TPCDS_BENCH_WARMUPS="${WARMUPS}"
    export ALGEON_TPCDS_BENCH_ITERATIONS="${ITERATIONS}"
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
SAFE_QUERIES="$(printf '%s' "${QUERIES}" | tr ',' '_')"
if [[ "${#SAFE_QUERIES}" -gt 40 ]]; then
    SAFE_QUERIES="multiq"
fi
if [[ -n "${OUT_FLAG}" ]]; then
    TSV_OUT="${OUT_FLAG}"
else
    TSV_OUT="/tmp/bench-one/${WORKLOAD}_sf${SCALE}_${SAFE_QUERIES}_${STAMP}.tsv"
fi
mkdir -p "$(dirname "${TSV_OUT}")"
STDERR_LOG="${TSV_OUT}.stderr.log"

# Benchmarks always default to the release profile: dev-profile timings are
# meaningless for tuning. --dev opts back into a fast-compile debug build.
if [[ "${DEV_PROFILE}" -eq 1 ]]; then
    CMD=(cargo run -p bench --all-features --bin "${BIN}")
else
    CMD=(cargo run --release -p bench --all-features --bin "${BIN}")
fi
if [[ "${NEEDS_LOCK}" -eq 1 ]]; then
    CMD=(flock /tmp/cudf-gpu.lock "${CMD[@]}")
fi

if [[ "${DEV_PROFILE}" -eq 1 ]]; then
    PROFILE="dev (debug; timings NOT meaningful)"
else
    PROFILE="release"
fi
echo "workload:       ${WORKLOAD} (bin: ${BIN})"
echo "profile:        ${PROFILE}"
echo "queries:        ${QUERIES}"
echo "parquet:        ${ALGEON_PARQUET_DIR}"
echo "warmups/iters:  ${WARMUPS}/${ITERATIONS}"
if [[ "${WORKLOAD}" == "tpcds" ]]; then
    echo "engine:         ${ENGINE}"
fi
echo "tsv:            ${TSV_OUT}"
echo "stderr log:     ${STDERR_LOG}"
echo ""

review_command() {
    if [[ "${WORKLOAD}" == "tpch" ]]; then
        local args=()
        if [[ -n "${WIDTH_FLAG}" ]]; then
            args+=(--width "${WIDTH_FLAG}")
        fi
        if [[ "${NO_PRETTY}" -eq 1 ]]; then
            args+=(--pretty never)
        fi
        printf '  %q' "${REVIEW_PY}" "${REPO_ROOT}/crates/bench/tools/tpch_bench_review.py"
        if [[ "${#args[@]}" -gt 0 ]]; then
            printf ' %q' "${args[@]}"
        fi
        printf ' %q\n' "${TSV_OUT}"
    else
        printf '  # compact key columns of this run:\n'
        printf '  grep -v %s %s | cut -f1-6,9,11-13 | column -t\n' "'^#'" "${TSV_OUT}"
        printf '  # CPU-vs-GPU compare needs a second engine TSV:\n'
        printf '  python3 %q --gpu-tsv <gpu.tsv> --cpu-tsv <cpu.tsv>\n' \
            "${REPO_ROOT}/crates/bench/tools/tpcds_cpu_gpu_compare.py"
    fi
}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] environment this run would export:"
    if [[ "${WORKLOAD}" == "tpch" ]]; then
        echo "  ALGEON_PARQUET_DIR=${ALGEON_PARQUET_DIR}"
        echo "  ALGEON_TPCH_BENCH_QUERY_IDS=${QUERIES}"
        echo "  ALGEON_TPCH_BENCH_DATASET_MODE=sf${SCALE}_external"
        echo "  ALGEON_TPCH_SCALE_FACTOR=${ALGEON_TPCH_SCALE_FACTOR}"
        echo "  ALGEON_TPCH_GENERATION_COMMAND=${ALGEON_TPCH_GENERATION_COMMAND}"
        echo "  ALGEON_TPCH_BENCH_WARMUPS=${WARMUPS}"
        echo "  ALGEON_TPCH_BENCH_ITERATIONS=${ITERATIONS}"
    else
        echo "  ALGEON_PARQUET_DIR=${ALGEON_PARQUET_DIR}"
        echo "  ALGEON_TPCDS_BENCH_ENGINE=${ENGINE}"
        echo "  ALGEON_TPCDS_BENCH_QUERIES=${QUERIES}"
        echo "  ALGEON_TPCDS_BENCH_WARMUPS=${WARMUPS}"
        echo "  ALGEON_TPCDS_BENCH_ITERATIONS=${ITERATIONS}"
    fi
    echo "[dry-run] command:"
    printf '  %q' "${CMD[@]}"
    printf ' > %q 2> >(tee %q >&2)\n' "${TSV_OUT}" "${STDERR_LOG}"
    if [[ "${NO_REVIEW}" -eq 0 ]]; then
        echo "[dry-run] review step:"
        review_command
    fi
    if [[ "${EXPLORE}" -eq 1 ]]; then
        echo "[dry-run] explore step:"
        if [[ -n "$(resolve_vd)" ]]; then
            printf '  %q %q\n' "$(resolve_vd)" "${TSV_OUT}"
        else
            echo "  (visidata not found; would print a hint)"
        fi
    fi
    exit 0
fi

# stdout is the TSV (kept machine-clean); stderr stays live on the terminal
# and is also saved to the stderr log.
"${CMD[@]}" > "${TSV_OUT}" 2> >(tee "${STDERR_LOG}" >&2)

echo ""
echo "wrote ${TSV_OUT}"

if [[ "${NO_REVIEW}" -eq 0 ]]; then
    echo ""
    if [[ "${WORKLOAD}" == "tpch" ]]; then
        review_args=()
        if [[ -n "${WIDTH_FLAG}" ]]; then
            review_args+=(--width "${WIDTH_FLAG}")
        fi
        if [[ "${NO_PRETTY}" -eq 1 ]]; then
            review_args+=(--pretty never)
        fi
        "${REVIEW_PY}" "${REPO_ROOT}/crates/bench/tools/tpch_bench_review.py" "${review_args[@]:+${review_args[@]}}" "${TSV_OUT}"
    else
        echo "--- compact summary (key columns) ---"
        grep '^#' "${TSV_OUT}" || true
        if command -v column > /dev/null 2>&1; then
            grep -v '^#' "${TSV_OUT}" | cut -f1-6,9,11-13 | column -t -s $'\t'
        else
            grep -v '^#' "${TSV_OUT}" | cut -f1-6,9,11-13
        fi
        echo ""
        echo "For a CPU-vs-GPU comparison, re-run with --engine cpu_datafusion, then:"
        echo "  python3 crates/bench/tools/tpcds_cpu_gpu_compare.py --gpu-tsv <gpu.tsv> --cpu-tsv <cpu.tsv>"
    fi
fi

VD_BIN="$(resolve_vd)"
if [[ "${EXPLORE}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
    echo ""
    if [[ -z "${VD_BIN}" ]]; then
        echo "visidata not found: activate the project .venv (ships vd) and run vd ${TSV_OUT}" >&2
        exit 1
    elif [[ ! -t 0 ]]; then
        echo "skipping --explore: stdin is not a terminal; inspect the TSV with vd ${TSV_OUT}" >&2
    else
        echo "opening ${TSV_OUT} in visidata (q to quit) ..."
        "${VD_BIN}" "${TSV_OUT}"
    fi
elif [[ -n "${VD_BIN}" ]]; then
    echo "Full-column interactive view: vd ${TSV_OUT}"
fi
