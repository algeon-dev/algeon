#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

mapfile -t docs_crates < <(python3 - "${repo_root}/Cargo.toml" <<'PY'
import pathlib
import sys
import tomllib

root_manifest = pathlib.Path(sys.argv[1])
with root_manifest.open("rb") as source:
    root = tomllib.load(source)
manifests = []
for member in root["workspace"]["members"]:
    if member == ".":
        continue
    manifests.append(root_manifest.parent / member / "Cargo.toml")

for manifest in manifests:
    with manifest.open("rb") as source:
        data = tomllib.load(source)
    package = data["package"]
    docs = package.get("metadata", {}).get("docs", {}).get("rs")
    if docs is None:
        continue
    if docs.get("default-target") != "x86_64-unknown-linux-gnu" or docs.get("targets") != []:
        raise SystemExit(f"error: documented crate {package['name']} has an invalid docs.rs target contract")
    expected_features = set(data.get("features", {})) - {"default", "test-hooks", "qualification"}
    documented_features = set(docs.get("features", []))
    if documented_features != expected_features:
        raise SystemExit(
            f"error: documented crate {package['name']} docs.rs features are {sorted(documented_features)}, "
            f"expected {sorted(expected_features)}"
        )
    print(f"{package['name']}\t{','.join(docs.get('features', []))}")
PY
)

[[ "${#docs_crates[@]}" -eq 12 ]] || {
  printf 'error: expected 12 documented docs.rs crates, found %s\n' "${#docs_crates[@]}" >&2
  exit 1
}

for entry in "${docs_crates[@]}"; do
  IFS=$'\t' read -r package features <<<"${entry}"
  feature_args=()
  if [[ -n "${features}" ]]; then
    feature_args=(--features "${features}")
  fi
  DOCS_RS=1 cargo doc -p "${package}" --no-deps --locked "${feature_args[@]}"
done

printf 'docs.rs native-free contracts OK for %s documented crates\n' "${#docs_crates[@]}"
