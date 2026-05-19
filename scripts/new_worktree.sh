#!/usr/bin/env bash
# Create an isolated git worktree for parallel agent development on
# datafusion-cudf. Shares sibling path dependencies (cudf-rs, datafusion,
# external-engine, ...) with the primary checkout; isolates only the Cargo target dir
# so parallel builds don't clobber each other.
#
# Usage:
#   scripts/new_worktree.sh <name> [base-ref]
#
#   <name>     short tag used for the branch and worktree dir
#   [base-ref] branch/sha to fork from (default: main)
#
# Result:
#   - worktree at ../datafusion-cudf-wt-<name>
#   - new branch wt/<name> tracking the base ref
#   - isolated CARGO_TARGET_DIR under ~/code/cargo-target/datafusion-cudf-wt-<name>
#   - .venv symlink to the shared primary-checkout Python environment
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
repo_root="$(cd "$script_dir/.." && pwd)"
parent_dir="$(dirname "$repo_root")"

worktree_path="$parent_dir/datafusion-cudf-wt-$name"
branch="wt/$name"
target_dir="$HOME/code/cargo-target/datafusion-cudf-wt-$name"
shared_venv="$parent_dir/datafusion-cudf/.venv"

if [[ -e "$worktree_path" ]]; then
    echo "error: $worktree_path already exists" >&2
    exit 1
fi

echo "==> Creating worktree at $worktree_path (branch $branch from $base_ref)"
git -C "$repo_root" worktree add -b "$branch" "$worktree_path" "$base_ref"

mkdir -p "$target_dir"
if [[ -d "$shared_venv" ]]; then
    ln -s "$shared_venv" "$worktree_path/.venv"
else
    echo "warning: shared Python venv $shared_venv does not exist; skipping worktree .venv symlink" >&2
fi

envrc="$worktree_path/.envrc.sh"
cat >"$envrc" <<EOF
# Source this file when working in this worktree:
#   source .envrc.sh
#
# Provides:
#   - isolated CARGO_TARGET_DIR so parallel worktrees don't fight over target/
#   - PATH pointing at the shared Python .venv when present
#   - LD_LIBRARY_PATH pointing at the shared cudf-rs install + this worktree's
#     own cudf-sys build output (in case cudf-sys is rebuilt here).
CARGO_TARGET_DIR="$target_dir"
export CARGO_TARGET_DIR

if [[ -d "$worktree_path/.venv/bin" && ":\${PATH:-}:" != *":$worktree_path/.venv/bin:"* ]]; then
    PATH="$worktree_path/.venv/bin\${PATH:+:\$PATH}"
    export PATH
fi

# Shared libcudf install (from the primary cudf-rs checkout). Every worktree
# links against the same libcudf.so unless you explicitly override cudf-rs via
# a .cargo/config.toml [paths] entry in this worktree.
_CUDF_INSTALL_LIB="$HOME/code/cudf-rs/cudf-install/lib"
_WT_CUDF_SYS_OUT="\$CARGO_TARGET_DIR/release/build"

if [[ ":\${LD_LIBRARY_PATH:-}:" != *":\$_CUDF_INSTALL_LIB:"* ]]; then
    LD_LIBRARY_PATH="\$_CUDF_INSTALL_LIB\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi

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
Python venv:      $worktree_path/.venv -> $shared_venv

Next steps:
  cd $worktree_path
  source .envrc.sh
  cargo build

Sibling path-deps shared with primary checkout (read-only by convention):
  $parent_dir/cudf-rs
  $parent_dir/datafusion
  $parent_dir/external-engine
  $parent_dir/iceberg-rust
  $parent_dir/cuCascade

If you need to modify a sibling dep for this worktree only, add a
[paths] override to $worktree_path/.cargo/config.toml — do NOT edit
the sibling repo in place, every other worktree shares it.

GPU is a single resource. Serialize benchmark/execution runs across
worktrees with e.g. \`flock /tmp/cudf-gpu.lock cargo run --release ...\`.
EOF
