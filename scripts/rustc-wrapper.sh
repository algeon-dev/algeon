#!/usr/bin/env bash
set -euo pipefail

rustc="$1"
shift

for arg in "$@"; do
    case "$arg" in
        /home/ubuntu/code/datafusion/*)
            exec "$rustc" -A deprecated "$@"
            ;;
    esac
done

exec "$rustc" "$@"
