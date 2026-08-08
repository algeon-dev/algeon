use std::{
    path::Path,
    process::{Command, Output},
};

use anyhow::{Context, Result, bail};

pub(crate) fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

pub(crate) fn run(mut command: Command, description: &str) -> Result<()> {
    let status = command.status().with_context(|| format!("failed to run {description}"))?;
    if !status.success() {
        bail!("{description} failed with {status}");
    }
    Ok(())
}

pub(crate) fn output(mut command: Command, description: &str) -> Result<Output> {
    let output = command.output().with_context(|| format!("failed to run {description}"))?;
    if !output.status.success() {
        bail!("{description} failed with {}:\n{}", output.status, String::from_utf8_lossy(&output.stderr).trim());
    }
    Ok(output)
}

pub(crate) fn cargo_at(workspace_root: &Path) -> Command {
    let mut command = cargo();
    command.current_dir(workspace_root);
    command
}

pub(crate) fn arg_path(command: &mut Command, path: &Path) {
    command.arg(path.as_os_str());
}
