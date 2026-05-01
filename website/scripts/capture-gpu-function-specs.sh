#!/usr/bin/env bash
# Capture all installed GPU function descriptors from the unified catalog on a
# running Flight SQL server. Specs are stored by provider so generated docs
# retain the descriptor's provider/domain grouping without a second inventory.
#
# Prerequisites: arrow_cli and jq on PATH; a server with the desired GPU
# families installed. For example:
#   cargo build --release --features server,cugraph,cuvs --bin datafusion_nexus_server
#   NEXUS_SERVER_CUGRAPH_ENABLED=true \
#   bash scripts/dev/run_server.sh
set -euo pipefail

HOST="${GPU_FUNCTION_SPECS_HOST:-127.0.0.1}"
PORT="${GPU_FUNCTION_SPECS_PORT:-50051}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="$SCRIPT_DIR/../gpu-function-specs"
STAGE_ROOT="$(mktemp -d "$SCRIPT_DIR/../.gpu-function-specs.stage.XXXXXX")"
OLD_ROOT="$OUT_ROOT.old.$$"
LIST_FILE="$(mktemp "$SCRIPT_DIR/../.gpu-function-specs.XXXXXX")"
trap 'rm -rf "$STAGE_ROOT" "$OLD_ROOT" "$LIST_FILE"' EXIT

run_sql() {
  arrow_cli --host "$HOST" -P "$PORT" --timeout 120 --output json -c "$1"
}

validate_provider() {
  [[ "$1" =~ ^[a-z][a-z0-9_]*$ ]]
}

validate_function_name() {
  [[ "$1" =~ ^[a-z][a-z0-9_]*$ ]]
}

echo "Capturing GPU function descriptors from $HOST:$PORT ..."
run_sql "SELECT function_name, provider FROM gpu_list_functions() ORDER BY function_name" > "$LIST_FILE"

count=0
while IFS= read -r row; do
  [[ -n "$row" ]] || continue
  function_name="$(jq -er '.function_name | strings' <<<"$row")"
  provider="$(jq -er '.provider | strings' <<<"$row")"
  validate_function_name "$function_name" || {
    echo "ERROR: invalid function_name from catalog: $function_name" >&2
    exit 1
  }
  validate_provider "$provider" || {
    echo "ERROR: invalid provider from catalog: $provider" >&2
    exit 1
  }
  escaped_name="${function_name//\'/\'\'}"
  descriptor="$(run_sql "SELECT * FROM gpu_describe_function('$escaped_name')")"
  descriptor="$(jq -ce \
    'select(.function_name == $name and .provider == $provider)' \
    --arg name "$function_name" --arg provider "$provider" <<<"$descriptor")"
  mkdir -p "$STAGE_ROOT/$provider"
  # Flight SQL's JSON output omits nullable fields whose value is SQL NULL.
  # The committed descriptor contract retains the key so generators can
  # distinguish an available function from a malformed spec deterministically.
  jq -cS '. + {unavailable_reason: (.unavailable_reason // null)}' <<<"$descriptor" > "$STAGE_ROOT/$provider/$function_name.json"
  count=$((count + 1))
done < "$LIST_FILE"

if [[ "$count" -eq 0 ]]; then
  echo "ERROR: gpu_list_functions returned no functions; refusing to replace $OUT_ROOT" >&2
  exit 1
fi
if [[ "$(find "$STAGE_ROOT" -type f -name '*.json' | wc -l)" -ne "$count" ]]; then
  echo "ERROR: duplicate or invalid catalog descriptors; refusing to replace $OUT_ROOT" >&2
  exit 1
fi

if [[ -d "$OUT_ROOT" ]]; then mv "$OUT_ROOT" "$OLD_ROOT"; fi
mv "$STAGE_ROOT" "$OUT_ROOT"
echo "Wrote $count unified GPU function specs to $OUT_ROOT (atomic swap)"
