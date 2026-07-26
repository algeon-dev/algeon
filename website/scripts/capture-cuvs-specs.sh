#!/usr/bin/env bash
# Capture the cuVS SQL operation registry from a running Flight SQL server.
#
# The generated reference pages read the committed JSON below; site builds do
# not need a server. Capture after a metadata-contract change, then run
# `node website/scripts/gen-cuvs-docs.mjs`.
#
# Prerequisites: `arrow_cli` on PATH and a cuVS-enabled server, for example:
#   cargo build --release --features server,cuvs --bin datafusion_nexus_server
#   NEXUS_SERVER_NATIVE_MEMORY_POLICY=unbounded bash scripts/dev/run_server.sh
set -euo pipefail

HOST="${CUVS_DOCS_HOST:-127.0.0.1}"
PORT="${CUVS_DOCS_PORT:-50051}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../cuvs-specs"
STAGE_DIR="$(mktemp -d "$SCRIPT_DIR/../.cuvs-specs.stage.XXXXXX")"
OLD_DIR="$OUT_DIR.old.$$"
LIST_FILE="$(mktemp "$SCRIPT_DIR/../.cuvs-operation-specs.XXXXXX")"
trap 'rm -rf "$STAGE_DIR" "$OLD_DIR" "$LIST_FILE"' EXIT

run_sql() {
  arrow_cli --host "$HOST" -P "$PORT" --timeout 120 --output json -c "$1"
}

echo "Capturing cuVS SQL metadata from $HOST:$PORT ..."
run_sql 'SELECT * FROM cuvs_list_operations() ORDER BY function_name' > "$LIST_FILE"

count=0
while IFS= read -r row; do
  [[ -n "$row" ]] || continue
  function_name="$(jq -er '.function_name | strings | select(test("^cuvs_[a-z0-9_]+$"))' <<<"$row")"
  jq -e \
    'has("operation") and has("relation_roles_json") and has("relation_argument_syntax") and has("output_contract") and has("options_schema_json")' \
    >/dev/null <<<"$row"
  jq -cS . <<<"$row" > "$STAGE_DIR/$function_name.json"
  count=$((count + 1))
done < "$LIST_FILE"

if [[ "$count" -eq 0 ]]; then
  echo "ERROR: cuvs_list_operations returned no operations; refusing to replace $OUT_DIR" >&2
  exit 1
fi
if [[ "$(find "$STAGE_DIR" -maxdepth 1 -name '*.json' -type f | wc -l)" -ne "$count" ]]; then
  echo "ERROR: duplicate or invalid cuVS operation names in metadata; refusing to replace $OUT_DIR" >&2
  exit 1
fi

if [[ -d "$OUT_DIR" ]]; then mv "$OUT_DIR" "$OLD_DIR"; fi
mv "$STAGE_DIR" "$OUT_DIR"
echo "Wrote $count cuVS operation specs to $OUT_DIR (atomic swap)"
