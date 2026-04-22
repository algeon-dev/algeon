#!/usr/bin/env bash
# Capture verbose cuGraph UDF specs from a running cuGraph-enabled Flight SQL
# server into website/cugraph-specs/. Run once when the metadata contract
# changes, then re-run gen-cugraph-docs.mjs. Build never needs the server.
#
# Prereqs: a running cuGraph-enabled server and arrow_cli on PATH.
#   cargo build --release --features server-cugraph --bin datafusion_nexus_server
#   NEXUS_SERVER_CUGRAPH_ENABLED=true \
#   NEXUS_SERVER_BIND=127.0.0.1:50051 bash scripts/dev/run_server.sh
set -euo pipefail

HOST="${CUGRAPH_DOCS_HOST:-127.0.0.1}"
PORT="${CUGRAPH_DOCS_PORT:-50051}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../cugraph-specs"
# Capture into a sibling staging dir (same filesystem as OUT_DIR) so the final
# swap is a single atomic rename — readers never see an empty/partial spec set,
# and a removed algorithm cannot leave a stale JSON behind.
STAGE_DIR="$(mktemp -d "$SCRIPT_DIR/../.cugraph-specs.stage.XXXXXX")"
OLD_DIR="$OUT_DIR.old.$$"
trap 'rm -rf "$STAGE_DIR" "$OLD_DIR"' EXIT

run_sql() { printf '%s\n' "$1" | arrow_cli --host "$HOST" -P "$PORT" --timeout 120 --output "$2"; }

echo "Listing algorithms from $HOST:$PORT ..."
mapfile -t FNS < <(run_sql "SELECT function_name FROM cugraph_list_algorithms();" tsv | sed '/^$/d')
echo "Found ${#FNS[@]} functions."

for fn in "${FNS[@]}"; do
  echo "  describe $fn"
  run_sql "SELECT * FROM cugraph_describe_algorithm('$fn', 'verbose');" json > "$STAGE_DIR/$fn.json"
done

# Atomic swap: rename the populated staging dir into place. If OUT_DIR exists,
# move it aside first, then drop it (cleaned by the trap). Both renames are
# single atomic operations on the same filesystem.
if [ -d "$OUT_DIR" ]; then mv "$OUT_DIR" "$OLD_DIR"; fi
mv "$STAGE_DIR" "$OUT_DIR"
echo "Wrote ${#FNS[@]} specs to $OUT_DIR (atomic swap)"
