#!/usr/bin/env bash
# Tests for the hidden-API consumer probe. A throwaway three-package workspace
# (an owner with hidden items, a library consumer, and a `tools` binary) runs
# through scripts/ci/check_public_api_consumers.py: the matching ledger passes,
# and a wrong consumer, a missing `exposed-by`, a malformed consumer column,
# and an unprobeable symbol each fail closed.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${ROOT}"
PROBE=scripts/ci/check_public_api_consumers.py

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/algeon-public-api-consumers-tests.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT
ws="${tmp_dir}/workspace"
mkdir -p "${ws}/owner/src" "${ws}/owner/tests" "${ws}/user/src" "${ws}/tools/src" "${tmp_dir}/ledgers"

cat >"${ws}/Cargo.toml" <<'TOML'
[workspace]
members = ["owner", "user", "tools"]
resolver = "3"
TOML
cat >"${ws}/owner/Cargo.toml" <<'TOML'
[package]
name = "owner"
version = "0.0.0"
edition = "2024"
publish = false

[features]
test-hooks = []
TOML
cat >"${ws}/owner/src/lib.rs" <<'RS'
mod inner {
    pub struct Reexported;
}

#[doc(hidden)]
pub use inner::Reexported;

#[doc(hidden)]
pub fn used_by_prod() {}

#[doc(hidden)]
pub fn used_by_tests() {}

#[doc(hidden)]
pub mod seam {
    pub fn child() {}

    #[doc(hidden)]
    pub fn listed_child() {}
}

#[cfg(feature = "test-hooks")]
#[doc(hidden)]
pub fn hook() {}

#[doc(hidden)]
pub struct Exposed;

mod private_items {
    pub struct Faced;
}

#[doc(hidden)]
pub mod facade {
    pub use crate::private_items::Faced;
}

pub fn exposed() -> Exposed {
    Exposed
}
RS
cat >"${ws}/owner/tests/integration.rs" <<'RS'
#[test]
fn integration() {
    owner::used_by_tests();
}
RS
cat >"${ws}/user/Cargo.toml" <<'TOML'
[package]
name = "user"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
owner = { path = "../owner" }

[features]
test-hooks = ["owner/test-hooks"]
TOML
cat >"${ws}/user/src/lib.rs" <<'RS'
pub fn production() -> owner::Reexported {
    owner::used_by_prod();
    owner::seam::listed_child();
    let _ = owner::exposed();
    owner::Reexported
}

#[cfg(feature = "test-hooks")]
pub fn support() {
    owner::hook();
}

#[cfg(test)]
mod tests {
    #[test]
    fn uses_hidden_item() {
        owner::used_by_tests();
    }
}
RS
cat >"${ws}/tools/Cargo.toml" <<'TOML'
[package]
name = "tools"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
owner = { path = "../owner" }
TOML
cat >"${ws}/tools/src/main.rs" <<'RS'
fn main() {
    owner::seam::child();
    let _ = owner::facade::Faced;
}
RS
git -C "${ws}" init -q
(cd "${ws}" && cargo generate-lockfile --offline -q)

ledger="${tmp_dir}/ledgers/owner-hidden.tsv"
write_ledger() {
  printf 'symbol\tfeature\towner\tconsumer\tremoval_condition\n' >"${ledger}"
  local row
  for row in "$@"; do
    printf '%s\t%s\towner crate root\t%s\tremove when no consumer needs it\n' \
      "${row%%|*}" "$(cut -d'|' -f2 <<<"${row}")" "${row##*|}" >>"${ledger}"
  done
}
clean_rows=(
  "owner::Exposed|cross-crate|exposed-by: owner::exposed"
  "owner::Reexported|cross-crate|prod: user"
  "owner::facade|cross-crate|tools: tools"
  "owner::hook|test-hooks|tests: user"
  "owner::seam|cross-crate|prod: user; tools: tools"
  "owner::seam::listed_child|cross-crate|prod: user"
  "owner::used_by_prod|cross-crate|prod: user"
  "owner::used_by_tests|cross-crate|tests: owner, user"
)
ledger_with() {
  local symbol=$1 consumer=$2 row rows=()
  for row in "${clean_rows[@]}"; do
    if [[ "${row%%|*}" == "${symbol}" ]]; then
      row="${row%|*}|${consumer}"
    fi
    rows+=("${row}")
  done
  write_ledger "${rows[@]}"
}

run_probe() {
  local name=$1 status=0
  python3 "${PROBE}" "${ws}" "${tmp_dir}/ledgers" "${tmp_dir}/out" >"${tmp_dir}/${name}.output" 2>&1 || status=$?
  printf '%s\n' "${status}" >"${tmp_dir}/${name}.status"
}
assert_status() {
  local actual
  actual=$(cat "${tmp_dir}/$1.status")
  if [[ "${actual}" != "$2" ]]; then
    echo "FAIL: $1 exited ${actual}, expected $2" >&2
    cat "${tmp_dir}/$1.output" >&2
    exit 1
  fi
}
assert_contains() {
  if ! grep -qF -- "$2" "${tmp_dir}/$1.output"; then
    echo "FAIL: $1 output lacks: $2" >&2
    cat "${tmp_dir}/$1.output" >&2
    exit 1
  fi
}

write_ledger "${clean_rows[@]}"
run_probe clean
assert_status clean 0
assert_contains clean "PASS: 8 hidden symbols' consumers match their ledgers"
echo "PASS: production, tools, test-hooks, cfg(test), integration-test, re-export, and module uses classify as measured"

ledger_with owner::used_by_prod "tests: user"
run_probe wrong_kind
assert_status wrong_kind 1
assert_contains wrong_kind "owner::used_by_prod: prod consumers are user, ledger lists none"
assert_contains wrong_kind "owner::used_by_prod: tests consumers are none, ledger lists user"
echo "PASS: a consumer recorded under the wrong kind fails closed"

ledger_with owner::Exposed "tests: owner"
run_probe unexposed
assert_status unexposed 1
assert_contains unexposed "owner::Exposed: no package names it; make it pub(crate) or record exposed-by"
echo "PASS: a hidden symbol no package names needs exposed-by"

ledger_with owner::Reexported "prod: user, user"
run_probe malformed
assert_status malformed 1
assert_contains malformed "consumer 'prod: user, user': prod names must be non-empty, sorted, and unique"
echo "PASS: a malformed consumer column fails closed"

cat >>"${ws}/owner/src/lib.rs" <<'RS'

impl Clone for Exposed {
    #[doc(hidden)]
    fn clone(&self) -> Self {
        Exposed
    }
}
RS
write_ledger "${clean_rows[@]}" "owner::<Exposed as Clone>::clone|cross-crate|prod: user"
run_probe trait_member
assert_status trait_member 1
assert_contains trait_member "cannot probe owner::<Exposed as Clone>::clone: a trait impl member"
echo "PASS: a symbol the probe cannot attribute fails closed"
