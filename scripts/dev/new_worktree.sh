#!/usr/bin/env bash
# Create an isolated algeon-datafusion worktree, including native sources and
# build/install directories.
#
# Usage:
#   scripts/dev/new_worktree.sh <name> [base-ref]
#
#   <name>     short tag used for the branch and worktree dir
#   [base-ref] branch/sha to fork from (default: main)
#
# Result:
#   - worktree at ../algeon-wt-<name>
#   - new branch wt/<name> tracking the base ref
#   - recursive native-source submodules at their pinned revisions
#   - isolated CARGO_TARGET_DIR and native artifacts under the worktree
#   - .envrc.sh inside the worktree that exports the right env vars

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <name> [base-ref]" >&2
    exit 2
fi

name="$1"
base_ref="${2:-main}"

if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "error: name must match [a-zA-Z0-9_-]+" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
parent_dir="$(dirname "$repo_root")"

worktree_path="$parent_dir/algeon-wt-$name"
branch="wt/$name"
target_dir="$worktree_path/target"

if [[ -e "$worktree_path" ]]; then
    echo "error: $worktree_path already exists" >&2
    exit 1
fi

echo "==> Creating worktree at $worktree_path (branch $branch from $base_ref)"
git -C "$repo_root" worktree add -b "$branch" "$worktree_path" "$base_ref"
git -C "$worktree_path" submodule update --init --recursive

mkdir -p "$target_dir"

envrc="$worktree_path/.envrc.sh"
cat >"$envrc" <<EOF
# Source this file when working in this worktree:
#   source .envrc.sh
#
CARGO_TARGET_DIR="$target_dir"
export CARGO_TARGET_DIR

# Native selectors default to this worktree's target/native/ prefixes; clearing
# them keeps the main checkout's libraries out and the build fingerprint stable.
unset ALGEON_NATIVE_DIR CUDF_CPP_BUILD_DIR CUDF_INSTALL_DIR INSTALL_PREFIX \\
  CUVS_INSTALL_DIR CUVS_ROOT CUGRAPH_CMAKE_BUILD_DIR CUGRAPH_INSTALL_DIR

# Tag the shell so you can see which worktree you're in.
PS1_WT_TAG="(wt:$name) "
case "\${PS1:-}" in
    *"\$PS1_WT_TAG"*) ;;
    *) PS1="\$PS1_WT_TAG\${PS1:-}" ;;
esac
EOF

chmod +x "$envrc"

cat <<EOF

==> Done.

Worktree:        $worktree_path
Branch:          $branch  (forked from $base_ref)
CARGO_TARGET_DIR: $target_dir
Native artifacts: $target_dir/native

Next steps:
  cd $worktree_path
  source .envrc.sh
  cargo build --all-features

The component crates, native gitlinks, Cargo target, and mutable native
build/install directories are all isolated inside this worktree.

GPU is a single resource. Serialize benchmark/execution runs across
worktrees with e.g. \`flock /tmp/cudf-gpu.lock cargo run --release --all-features ...\`.
EOF
