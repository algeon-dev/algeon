#!/usr/bin/env bash
# Holistic AD/Entra BloodHound graph: GPU ForceAtlas2 layout + sigma.js render.
#
#   scripts/holistic/run.sh            # compute layout, then serve the viz
#   scripts/holistic/run.sh layout     # GPU layout only -> out/graph.json
#   scripts/holistic/run.sh serve 8000 # serve the existing render on :8000
#   scripts/holistic/run.sh shot       # headless-Chrome screenshot -> out/preview.png
#
# All HOLISTIC_* env vars from holistic_layout.py are honored, e.g.:
#   HOLISTIC_RENDER_COORDS=holistic HOLISTIC_NODE_BUDGET=20000 scripts/holistic/run.sh layout
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PORT="${2:-8000}"

# venv with cudf/cugraph (override with HOLISTIC_PYTHON)
PY="${HOLISTIC_PYTHON:-$REPO/.venv/bin/python}"
CHROME="${HOLISTIC_CHROME:-$REPO/.venv/chrome-linux64/chrome}"
GPU_LOCK="${GPU_LOCK:-/tmp/cudf-gpu.lock}"

run_layout() {
  [ -x "$PY" ] || { echo "no python at $PY (set HOLISTIC_PYTHON)"; exit 1; }
  echo "[run] GPU ForceAtlas2 layout (serialized on $GPU_LOCK)…"
  # GPU is a single shared resource — always serialize (see AGENTS.md).
  flock "$GPU_LOCK" "$PY" "$HERE/holistic_layout.py"
}

serve() {
  [ -f "$HERE/out/graph.json" ] || { echo "no out/graph.json — run '$0 layout' first"; exit 1; }
  echo "[run] serving $HERE on http://localhost:$PORT/  (Ctrl-C to stop)"
  cd "$HERE" && exec /usr/bin/python3 -m http.server "$PORT"
}

shot() {
  [ -x "$CHROME" ] || { echo "no chrome at $CHROME (set HOLISTIC_CHROME)"; exit 1; }
  local out="$HERE/out/preview.png"
  cd "$HERE" && /usr/bin/python3 -m http.server "$PORT" >/tmp/holistic_http.$$ 2>&1 &
  local srv=$!; sleep 1.5
  "$CHROME" --headless=new --no-sandbox --hide-scrollbars \
    --use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist \
    --window-size=1680,1050 --virtual-time-budget=20000 \
    --screenshot="$out" "http://localhost:$PORT/" 2>/dev/null || true
  kill "$srv" 2>/dev/null || true
  echo "[run] wrote $out"
}

case "${1:-all}" in
  layout) run_layout ;;
  serve)  serve ;;
  shot)   shot ;;
  all)    run_layout; serve ;;
  *) echo "usage: $0 {all|layout|serve [port]|shot [port]}"; exit 2 ;;
esac
