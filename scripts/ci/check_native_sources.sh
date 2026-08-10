#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly expected_paths=(
  components/cudf/cudf
  components/cugraph/cugraph
  components/cuvs/cuvs
)

case "${1:-}" in
  '') selected_paths=("${expected_paths[@]}") ;;
  cudf | cugraph | cuvs) selected_paths=("components/$1/$1") ;;
  *)
    printf 'usage: %s [cudf|cugraph|cuvs]\n' "${0##*/}" >&2
    exit 2
    ;;
esac

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

cd "${repo_root}"
[[ -f .gitmodules ]] || fail ".gitmodules is missing"

mapfile -t registered_paths < <(
  git config -f .gitmodules --get-regexp '^submodule\..*\.path$' \
    | awk '{print $2}' \
    | sort
)
[[ "${#registered_paths[@]}" -eq "${#expected_paths[@]}" ]] \
  || fail ".gitmodules must register exactly the three native component sources"

for index in "${!expected_paths[@]}"; do
  [[ "${registered_paths[${index}]}" == "${expected_paths[${index}]}" ]] \
    || fail ".gitmodules native source paths do not match the root component registry"
done

for path in "${selected_paths[@]}"; do
  name=${path}
  configured_path=$(git config -f .gitmodules --get "submodule.${name}.path") \
    || fail ".gitmodules is missing submodule.${name}.path"
  [[ "${configured_path}" == "${path}" ]] \
    || fail "submodule.${name}.path must be ${path}"

  configured_url=$(git config -f .gitmodules --get "submodule.${name}.url") \
    || fail ".gitmodules is missing submodule.${name}.url"
  [[ -n "${configured_url}" ]] || fail "submodule.${name}.url must not be empty"

  index_entry=$(git ls-files --stage -- "${path}") \
    || fail "cannot inspect the ${path} gitlink"
  read -r mode pinned stage indexed_path extra <<<"${index_entry}"
  [[ "${mode}" == 160000 && "${stage}" == 0 && "${indexed_path}" == "${path}" && -z "${extra:-}" ]] \
    || fail "${path} must be one stage-0 gitlink in the root index"
  [[ "${pinned}" =~ ^[0-9a-f]{40}$ ]] || fail "${path} has an invalid gitlink revision"

  head_entry=$(git ls-tree HEAD -- "${path}") \
    || fail "cannot inspect the HEAD gitlink for ${path}"
  read -r head_mode head_type head_revision head_path head_extra <<<"${head_entry}"
  [[ "${head_mode}" == 160000 && "${head_type}" == commit && "${head_revision}" == "${pinned}" \
    && "${head_path}" == "${path}" && -z "${head_extra:-}" ]] \
    || fail "the index gitlink for ${path} must match the root HEAD"

  [[ -d "${path}" && -e "${path}/.git" && ! -L "${path}" ]] \
    || fail "the ${path} submodule must be initialized and must not be a symlink"
  checkout_revision=$(git -C "${path}" rev-parse HEAD) \
    || fail "cannot inspect the ${path} checkout revision"
  [[ "${checkout_revision}" == "${pinned}" ]] \
    || fail "${path} is at ${checkout_revision}; expected gitlink ${pinned}"

  checkout_url=$(git -C "${path}" config --get remote.origin.url) \
    || fail "${path} has no remote.origin.url"
  [[ "${checkout_url}" == "${configured_url}" ]] \
    || fail "${path} origin ${checkout_url} does not match .gitmodules URL ${configured_url}"

  dirty=$(git -C "${path}" status --porcelain=v1 --untracked-files=all) \
    || fail "cannot inspect the ${path} source state"
  [[ -z "${dirty}" ]] || fail "${path} has build-input changes:\n${dirty}"
done

printf 'native source registry contract OK\n'
