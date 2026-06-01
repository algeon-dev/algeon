#!/usr/bin/env bash
# Fails if staged tracked content, staged paths, or an optional commit message
# contain the restricted external-engine term. The needle is constructed from
# bytes so this guard does not reintroduce the term it blocks.

set -euo pipefail

restricted="$(printf '\x73\x69\x72\x69\x75\x73')"

usage() {
    cat >&2 <<'USAGE'
usage:
  scripts/guard_restricted_term.sh [--staged] [--commit-msg <path>] [--history]

With no arguments, checks the staged tree. --history runs retained-ref checks.
USAGE
}

check_staged() {
    local status=0

    if git diff --cached --name-only --diff-filter=ACMRT | rg -i "$restricted" >&2; then
        echo "restricted term found in staged path" >&2
        status=1
    fi

    while IFS= read -r path; do
        [[ -n "$path" ]] || continue
        if git show ":$path" | rg -i "$restricted" >&2; then
            echo "restricted term found in staged content: $path" >&2
            status=1
        fi
    done < <(git diff --cached --name-only --diff-filter=ACMRT)

    return "$status"
}

check_commit_msg() {
    local msg_path="$1"
    if rg -i "$restricted" "$msg_path" >&2; then
        echo "restricted term found in commit message: $msg_path" >&2
        return 1
    fi
}

check_history() {
    local status=0
    if git grep -I -n -i "$restricted" -- . >&2; then
        echo "restricted term found in current tracked tree" >&2
        status=1
    fi
    if git ls-files | rg -i "$restricted" >&2; then
        echo "restricted term found in tracked path" >&2
        status=1
    fi
    if git log --all --format=%B | rg -i "$restricted" >&2; then
        echo "restricted term found in retained commit message" >&2
        status=1
    fi
    if git log --all --name-only --format= | rg -i "$restricted" >&2; then
        echo "restricted term found in retained historical path" >&2
        status=1
    fi
    if git rev-list --all --objects | rg -i "$restricted" >&2; then
        echo "restricted term found in retained object/path list" >&2
        status=1
    fi
    return "$status"
}

if [[ $# -eq 0 ]]; then
    check_staged
    exit $?
fi

ran=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --staged)
            check_staged
            ran=1
            shift
            ;;
        --commit-msg)
            [[ $# -ge 2 ]] || {
                usage
                exit 2
            }
            check_commit_msg "$2"
            ran=1
            shift 2
            ;;
        --history)
            check_history
            ran=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

[[ "$ran" -eq 1 ]]
