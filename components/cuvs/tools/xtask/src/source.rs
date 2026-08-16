use std::{fs, path::Path, process::Command};

use anyhow::{Context, Result, ensure};

use crate::command;

const PINNED_CUVS_URL: &str = "https://github.com/rapidsai/cuvs.git";

fn is_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn git(workspace_root: &Path) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(workspace_root);
    command
}

fn git_text(workspace_root: &Path, args: &[&str], description: &str) -> Result<String> {
    let mut invocation = git(workspace_root);
    invocation.args(args);
    let output = command::output(invocation, description)?;
    String::from_utf8(output.stdout).with_context(|| format!("{description} returned non-UTF-8 output"))
}

/// The `cuvs` submodule gitlink recorded in the superproject is the source
/// pin. Maintenance tooling resolves it from git and must not hard-code a
/// SHA; this verifies the index, HEAD, and working checkout all agree on it.
pub(crate) fn verify(workspace_root: &Path) -> Result<String> {
    let gitmodules = workspace_root.join(".gitmodules");
    ensure!(gitmodules.is_file(), ".gitmodules is missing");

    let path = git_text(
        workspace_root,
        &["config", "-f", ".gitmodules", "--get", "submodule.cuvs.path"],
        "cuVS submodule path inspection",
    )?;
    ensure!(path.trim() == "cuvs", ".gitmodules must define submodule.cuvs.path=cuvs");
    let url = git_text(
        workspace_root,
        &["config", "-f", ".gitmodules", "--get", "submodule.cuvs.url"],
        "cuVS submodule URL inspection",
    )?;
    ensure!(url.trim() == PINNED_CUVS_URL, ".gitmodules must define submodule.cuvs.url={PINNED_CUVS_URL}");

    let index = git_text(workspace_root, &["ls-files", "--stage", "--", "cuvs"], "cuVS gitlink inspection")?;
    let index_fields = index.split_whitespace().collect::<Vec<_>>();
    ensure!(
        index_fields.len() == 4
            && index_fields[0] == "160000"
            && is_git_sha(index_fields[1])
            && index_fields[2] == "0"
            && index_fields[3] == "cuvs",
        "cuvs must be a stage-0 gitlink in the superproject index, found: {}",
        index.trim()
    );
    let pinned = index_fields[1].to_owned();

    let head = git_text(workspace_root, &["ls-tree", "HEAD", "--", "cuvs"], "HEAD cuVS gitlink inspection")?;
    if !head.trim().is_empty() {
        let fields = head.split_whitespace().collect::<Vec<_>>();
        ensure!(
            fields.len() == 4 && fields[0] == "160000" && fields[1] == "commit" && fields[2] == pinned,
            "superproject HEAD must record the same cuvs gitlink as the index ({pinned}), found: {}",
            head.trim()
        );
    }

    let source_dir = workspace_root.join("cuvs");
    ensure!(source_dir.join(".git").exists(), "cuVS submodule is not initialized at {}", source_dir.display());
    let metadata = fs::symlink_metadata(&source_dir)
        .with_context(|| format!("failed to inspect cuVS source path {}", source_dir.display()))?;
    ensure!(!metadata.file_type().is_symlink(), "cuvs source path must not be a symlink");

    let revision = git_text(&source_dir, &["rev-parse", "HEAD"], "cuVS source revision inspection")?;
    ensure!(
        revision.trim() == pinned,
        "cuVS submodule HEAD is {}; expected the recorded gitlink {pinned}",
        revision.trim()
    );
    let dirty =
        git_text(&source_dir, &["status", "--porcelain=v1", "--untracked-files=all"], "cuVS source status inspection")?;
    ensure!(dirty.trim().is_empty(), "cuVS source has build-input changes:\n{}", dirty.trim());

    println!("cuVS source contract OK: {pinned}");
    Ok(pinned)
}
