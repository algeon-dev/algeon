#!/usr/bin/env bash
set -euo pipefail

readonly source_script="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/check_native_sources.sh"
readonly test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

# Fixture commits must not depend on the host's git setup: a signing key that
# needs a passphrase fails without a terminal, and a CI runner has no identity.
# Environment config also reaches the submodule clones the test commits in.
export GIT_CONFIG_COUNT=3 \
  GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false \
  GIT_CONFIG_KEY_1=user.email GIT_CONFIG_VALUE_1=test@example.invalid \
  GIT_CONFIG_KEY_2=user.name GIT_CONFIG_VALUE_2='Native Source Policy Test'

for component in cudf cugraph cuml cuvs; do
  origin="${test_root}/${component}-origin"
  git init -q "${origin}"
  printf '%s\n' "${component}" >"${origin}/source.txt"
  git -C "${origin}" add source.txt
  git -C "${origin}" commit -qm initial
done

repo="${test_root}/repo"
git init -q "${repo}"
mkdir -p "${repo}/scripts/ci"
cp "${source_script}" "${repo}/scripts/ci/check_native_sources.sh"
git -C "${repo}" add scripts/ci/check_native_sources.sh
git -C "${repo}" commit -qm policy
for component in cudf cugraph cuml cuvs; do
  git -C "${repo}" -c protocol.file.allow=always submodule add -q \
    "${test_root}/${component}-origin" "components/${component}/${component}"
done
git -C "${repo}" commit -qam sources

check="${repo}/scripts/ci/check_native_sources.sh"
"${check}" >/dev/null

expect_failure() {
  local label=$1
  if "${check}" >"${test_root}/${label}.log" 2>&1; then
    printf 'error: native source check accepted %s\n' "${label}" >&2
    exit 1
  fi
}

printf 'dirty\n' >>"${repo}/components/cuvs/cuvs/source.txt"
expect_failure dirty-source
git -C "${repo}/components/cuvs/cuvs" restore source.txt

old_url=$(git -C "${repo}" config -f .gitmodules --get submodule.components/cuvs/cuvs.url)
git -C "${repo}" config -f .gitmodules submodule.components/cuvs/cuvs.url "${test_root}/wrong-origin"
expect_failure url-drift
git -C "${repo}" config -f .gitmodules submodule.components/cuvs/cuvs.url "${old_url}"

printf 'new revision\n' >>"${repo}/components/cuvs/cuvs/source.txt"
git -C "${repo}/components/cuvs/cuvs" add source.txt
git -C "${repo}/components/cuvs/cuvs" commit -qm next
expect_failure checkout-revision
git -C "${repo}" add components/cuvs/cuvs
expect_failure staged-gitlink

printf 'native source policy fault-injection tests OK\n'
