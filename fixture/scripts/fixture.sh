#!/usr/bin/env bash
# Unified public fixture entrypoint for benchmark data and local Iceberg E2E.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  fixture.sh tpch generate [--scale sf10] [--out PATH] [--force]
  fixture.sh tpcds generate [--scale sf10] [--out PATH] [--force]
  fixture.sh clickbench fetch [--layout single|partitioned|both] [--out PATH]
  fixture.sh iceberg rest load --workload tpch|tpcds [--scale sf1]
  fixture.sh iceberg glue-moto load --workload tpch|tpcds [--scale sf1]

This is a thin router. Workload-specific logic lives under
fixture/scripts/workloads/.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

domain="$1"
command="$2"
shift 2

case "${domain}:${command}" in
  tpch:generate)
    exec "${SCRIPT_DIR}/workloads/tpch/generate.sh" "$@"
    ;;
  tpcds:generate)
    exec "${SCRIPT_DIR}/workloads/tpcds/generate.sh" "$@"
    ;;
  clickbench:fetch)
    exec "${SCRIPT_DIR}/workloads/clickbench/fetch.sh" "$@"
    ;;
  iceberg:rest)
    subcommand="${1:-}"
    if [[ "${subcommand}" != "load" ]]; then
      echo "error: unsupported iceberg rest command '${subcommand}' (expected load)" >&2
      usage >&2
      exit 2
    fi
    shift
    exec "${SCRIPT_DIR}/workloads/iceberg/rest_load.sh" "$@"
    ;;
  iceberg:glue-moto)
    subcommand="${1:-}"
    if [[ "${subcommand}" != "load" ]]; then
      echo "error: unsupported iceberg glue-moto command '${subcommand}' (expected load)" >&2
      usage >&2
      exit 2
    fi
    shift
    exec "${SCRIPT_DIR}/workloads/iceberg/glue_moto_load.sh" "$@"
    ;;
  -h:*|--help:*)
    usage
    ;;
  *)
    echo "error: unsupported fixture command '${domain} ${command}'" >&2
    usage >&2
    exit 2
    ;;
esac
