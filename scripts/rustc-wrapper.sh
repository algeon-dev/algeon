#!/usr/bin/env bash
set -euo pipefail

rustc="$1"
shift

for arg in "$@"; do
    case "$arg" in
        */datafusion/datafusion/*)
            exec "$rustc" "$@" -A deprecated
            ;;
        @*)
            response_file="${arg#@}"
            if [[ -f "$response_file" ]] && grep -q '/datafusion/datafusion/' "$response_file"; then
                exec "$rustc" "$@" -A deprecated
            fi
            ;;
    esac
done

exec "$rustc" "$@"
