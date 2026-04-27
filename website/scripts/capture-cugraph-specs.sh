#!/usr/bin/env bash
# Compatibility entry point for capturing the whole unified GPU catalog.
# Generated cuGraph docs read website/gpu-function-specs/cugraph/ after this
# script completes. See capture-gpu-function-specs.sh for prerequisites.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/capture-gpu-function-specs.sh" "$@"
