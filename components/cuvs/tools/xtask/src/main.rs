mod command;
mod package;
mod source;
mod workspace;

#[cfg(feature = "bindings")]
mod bindings;

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};

fn workspace_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("xtask must remain under tools/xtask")
}

fn verify(workspace_root: &Path) -> Result<()> {
    source::verify(workspace_root)?;
    let versions = workspace::verify(workspace_root)?;

    let mut check = command::cargo_at(workspace_root);
    check.args(["check", "--locked", "-p", "cuvs-nexus-sys", "--all-targets"]);
    check.env("CUVS_NEXUS_SYS_SKIP_NATIVE", "1");
    command::run(check, "native-free cuvs-nexus-sys all-target check")?;

    let mut cmake = command::cargo_at(workspace_root);
    cmake.args(["test", "--locked", "-p", "cuvs-nexus-sys", "--test", "cmake_root_contract"]);
    cmake.env("CUVS_NEXUS_SYS_SKIP_NATIVE", "1");
    command::run(cmake, "CMake root contract tests")?;

    package::verify(workspace_root, &versions)?;
    println!("cuvs-nexus maintenance contracts OK");
    Ok(())
}

#[cfg(not(feature = "bindings"))]
fn bindings_command(workspace_root: &Path, args: &[OsString]) -> Result<()> {
    let mut invocation = command::cargo_at(workspace_root);
    invocation.args(["run", "--locked", "--package", "xtask", "--features", "bindings", "--"]);
    invocation.args(args);
    command::run(invocation, "binding maintenance command")
}

#[cfg(feature = "bindings")]
fn bindings_command(workspace_root: &Path, args: &[OsString]) -> Result<()> {
    ensure!(args.len() == 2, "usage: cargo xtask bindings <generate|check>");
    bindings::run(workspace_root, args[1].to_str().context("binding action must be valid UTF-8")?)
}

fn usage() -> &'static str {
    "usage:\n  cargo xtask bindings generate\n  cargo xtask bindings check\n  cargo xtask verify"
}

fn main() -> Result<()> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let workspace_root = workspace_root()?;
    match args.first().and_then(|argument| argument.to_str()) {
        Some("bindings") => bindings_command(&workspace_root, &args),
        Some("verify") => {
            ensure!(args.len() == 1, "{}", usage());
            verify(&workspace_root)
        },
        Some("source") => {
            ensure!(args.len() == 2 && args[1] == "check", "{}", usage());
            source::verify(&workspace_root)
        },
        Some("help" | "--help" | "-h") | None => {
            println!("{}", usage());
            Ok(())
        },
        _ => bail!("{}", usage()),
    }
}
