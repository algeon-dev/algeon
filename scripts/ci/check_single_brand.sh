#!/usr/bin/env bash
# Enforce one brand across every tracked path and blob. Gitlinks are checked by
# path here; their independently owned content is validated in their own repos.
set -euo pipefail

ROOT=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
legacy_brand=$(printf '%s%s' 'nex' 'us')
status=0

while IFS= read -r -d '' entry; do
  metadata=${entry%%$'\t'*}
  path=${entry#*$'\t'}
  mode=${metadata%% *}

  if [[ ${path,,} == *"${legacy_brand}"* ]]; then
    echo "ERROR: tracked path contains the legacy brand: ${path}" >&2
    status=1
  fi

  [[ ${mode} == 160000 ]] && continue

  if git -C "${ROOT}" show ":${path}" | LC_ALL=C grep -a -i -q -m1 -- "${legacy_brand}"; then
    echo "ERROR: tracked content contains the legacy brand: ${path}" >&2
    status=1
  fi

  if [[ ! -e ${ROOT}/${path} && ! -L ${ROOT}/${path} ]]; then
    # Pending deletions have no working-tree content; their indexed blob was
    # still checked above, including unstaged deletions.
    continue
  elif [[ -L ${ROOT}/${path} ]]; then
    target=$(readlink -- "${ROOT}/${path}")
    if [[ ${target,,} == *"${legacy_brand}"* ]]; then
      echo "ERROR: tracked symlink target contains the legacy brand: ${path}" >&2
      status=1
    fi
  elif LC_ALL=C grep -a -i -q -m1 -- "${legacy_brand}" "${ROOT}/${path}"; then
    echo "ERROR: worktree content contains the legacy brand: ${path}" >&2
    status=1
  fi
done < <(git -C "${ROOT}" ls-files -s -z)

if ((status == 0)); then
  echo "single-brand policy: tracked paths and content use Algeon"
fi
exit "${status}"
