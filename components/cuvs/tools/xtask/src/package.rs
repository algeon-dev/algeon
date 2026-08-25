use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, ensure};

use crate::command;

fn patch_config(package: &str, workspace_root: &Path) -> Vec<String> {
    if package != "cuvs-nexus" {
        return Vec::new();
    }
    [
        ("cudf-nexus", workspace_root.join("components/cudf/cudf-nexus")),
        ("rapids-interop", workspace_root.join("components/cudf/rapids-interop")),
        ("cuvs-nexus-sys", workspace_root.join("components/cuvs/cuvs-nexus-sys")),
    ]
    .into_iter()
    .map(|(name, path)| format!("patch.crates-io.{name}.path='{}'", path.display()))
    .collect()
}

fn cargo_package(workspace_root: &Path, package: &str) -> Command {
    let mut invocation = command::cargo_at(workspace_root);
    invocation.args(["package", "--locked", "--allow-dirty", "-p", package, "--all-features"]);
    invocation.env("CUVS_NEXUS_SYS_SKIP_NATIVE", "1");
    for config in patch_config(package, workspace_root) {
        invocation.arg("--config").arg(config);
    }
    invocation
}

fn archive_path(target_dir: &Path, package: &str, version: &str) -> PathBuf {
    target_dir.join("package").join(format!("{package}-{version}.crate"))
}

pub(crate) fn verify(workspace_root: &Path, versions: &std::collections::BTreeMap<String, String>) -> Result<()> {
    let temporary = tempfile::Builder::new().prefix("cuvs-nexus-package-").tempdir()?;
    for package in ["cuvs-nexus-sys", "cuvs-nexus"] {
        let mut list = cargo_package(workspace_root, package);
        list.arg("--list");
        let output = command::output(list, &format!("cargo package --list for {package}"))?;
        let listing = String::from_utf8(output.stdout).context("cargo package --list returned non-UTF-8 output")?;
        ensure!(listing.lines().any(|line| line == "LICENSE"), "{package} package list omits the workspace LICENSE");

        let target_dir = temporary.path().join(format!("{package}-target"));
        fs::create_dir_all(&target_dir)?;
        let mut package_command = cargo_package(workspace_root, package);
        package_command.arg("--target-dir");
        command::arg_path(&mut package_command, &target_dir);
        command::run(package_command, &format!("cargo package verification for {package}"))?;

        let version = versions.get(package).with_context(|| format!("metadata omitted {package} version"))?;
        let archive = archive_path(&target_dir, package, version);
        ensure!(archive.is_file(), "cargo package did not create {}", archive.display());
        let mut tar = Command::new("tar");
        tar.args(["-tzf"]).arg(&archive);
        let output = command::output(tar, &format!("archive inspection for {package}"))?;
        let archive_listing = String::from_utf8(output.stdout).context("tar returned non-UTF-8 output")?;
        ensure!(
            archive_listing.lines().any(|line| line == format!("{package}-{version}/LICENSE")),
            "{package} archive omits LICENSE"
        );
    }
    println!("crate package/verification contract OK");
    Ok(())
}
