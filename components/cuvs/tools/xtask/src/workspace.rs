use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::command;

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
    workspace_default_members: Vec<String>,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    version: String,
    publish: Option<Vec<String>>,
    links: Option<String>,
    features: BTreeMap<String, Vec<String>>,
    targets: Vec<Target>,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Target {
    name: String,
    kind: Vec<String>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    rename: Option<String>,
    optional: bool,
}

fn package<'a>(packages: &'a BTreeMap<&str, &Package>, name: &str) -> Result<&'a Package> {
    packages.get(name).copied().with_context(|| format!("workspace package {name} is missing"))
}

fn library_targets(package: &Package) -> Vec<&str> {
    package
        .targets
        .iter()
        .filter(|target| target.kind.iter().any(|kind| kind == "lib"))
        .map(|target| target.name.as_str())
        .collect()
}

pub(crate) fn verify(workspace_root: &Path) -> Result<BTreeMap<String, String>> {
    let mut invocation = command::cargo_at(workspace_root);
    invocation.args(["metadata", "--format-version", "1", "--no-deps", "--locked"]);
    let output = command::output(invocation, "cargo metadata")?;
    let metadata: Metadata = serde_json::from_slice(&output.stdout).context("cargo metadata returned invalid JSON")?;
    let packages = metadata.packages.iter().map(|package| (package.name.as_str(), package)).collect::<BTreeMap<_, _>>();
    let expected = BTreeSet::from(["cuvs-nexus", "cuvs-nexus-sys", "xtask"]);
    ensure!(packages.keys().copied().collect::<BTreeSet<_>>() == expected, "unexpected workspace package set");
    ensure!(
        metadata.workspace_members.iter().collect::<BTreeSet<_>>()
            == metadata.packages.iter().map(|package| &package.id).collect::<BTreeSet<_>>(),
        "all and only declared packages must be workspace members"
    );

    let safe = package(&packages, "cuvs-nexus")?;
    let sys = package(&packages, "cuvs-nexus-sys")?;
    let xtask = package(&packages, "xtask")?;
    let default_members = metadata.workspace_default_members.iter().collect::<BTreeSet<_>>();
    ensure!(
        default_members == BTreeSet::from([&safe.id, &sys.id]),
        "workspace default-members must remain the two library crates"
    );
    ensure!(library_targets(safe) == ["cuvs"], "cuvs-nexus must expose only the cuvs library target");
    ensure!(library_targets(sys) == ["cuvs_sys"], "cuvs-nexus-sys must expose only the cuvs_sys library target");
    ensure!(sys.links.as_deref() == Some("cuvs_c"), "cuvs-nexus-sys must retain links=cuvs_c");
    ensure!(sys.features.is_empty(), "cuvs-nexus-sys must not expose maintenance-only Cargo features");

    let safe_dependencies =
        safe.dependencies.iter().map(|dependency| dependency.name.as_str()).collect::<BTreeSet<_>>();
    ensure!(
        ["cudf-nexus", "cuvs-nexus-sys", "rapids-interop"].into_iter().all(|name| safe_dependencies.contains(name)),
        "cuvs-nexus is missing a required runtime dependency"
    );
    let sys_dependencies = sys.dependencies.iter().map(|dependency| dependency.name.as_str()).collect::<BTreeSet<_>>();
    ensure!(
        ["bindgen", "cudf-nexus", "cudf-nexus-sys", "rapids-interop"]
            .into_iter()
            .all(|name| !sys_dependencies.contains(name)),
        "cuvs-nexus-sys gained a forbidden dependency"
    );
    ensure!(xtask.publish.as_ref().is_some_and(Vec::is_empty), "xtask must remain unpublished");
    ensure!(
        xtask.targets.iter().map(|target| target.name.as_str()).collect::<BTreeSet<_>>() == BTreeSet::from(["xtask"]),
        "xtask must expose only the xtask binary"
    );
    let bindgen_dependencies =
        xtask.dependencies.iter().filter(|dependency| dependency.name == "bindgen").collect::<Vec<_>>();
    ensure!(
        matches!(bindgen_dependencies.as_slice(), [dependency] if dependency.rename.is_none() && dependency.optional),
        "xtask bindgen must be one optional, unrenamed dependency"
    );
    ensure!(
        xtask.features == BTreeMap::from([("bindings".to_owned(), vec!["dep:bindgen".to_owned()])]),
        "xtask must expose only `bindings = [\"dep:bindgen\"]`, with no default feature"
    );

    println!("workspace package/dependency contract OK");
    Ok(metadata.packages.into_iter().map(|package| (package.name, package.version)).collect())
}
