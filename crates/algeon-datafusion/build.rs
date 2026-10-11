use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    rapids_build_support::configure(rapids_build_support::Config {
        link_arg_targets: rapids_build_support::LinkArgTargets::new(true, false, true),
        consumers: rapids_build_support::RapidsConsumers::new(
            std::env::var_os("CARGO_FEATURE_CUGRAPH").is_some(),
            std::env::var_os("CARGO_FEATURE_CUVS").is_some(),
            std::env::var_os("CARGO_FEATURE_CUML").is_some(),
        ),
        nvtx3interop: false,
    });
    emit_git_commit();
}

/// Exposes the source commit to `build_info()` as `ALGEON_BUILD_GIT_COMMIT`.
/// `ALGEON_GIT_COMMIT` wins so builds from an exported tree (a Docker
/// context without `.git`) can still name their commit.
fn emit_git_commit() {
    println!("cargo:rerun-if-env-changed=ALGEON_GIT_COMMIT");
    // The build-support self-test runs this script outside Cargo, without
    // the manifest directory; the build then simply carries no commit.
    let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from) else {
        return;
    };
    if let Some(git_dir) = find_git_dir(&manifest_dir) {
        let head = git_dir.join("HEAD");
        println!("cargo:rerun-if-changed={}", head.display());
        if let Some(reference) = std::fs::read_to_string(&head)
            .ok()
            .and_then(|head| head.strip_prefix("ref:").map(|reference| reference.trim().to_string()))
        {
            // A worktree keeps HEAD in its own directory and branch refs in
            // the common directory; packed refs have no loose file to watch.
            let common_dir = std::fs::read_to_string(git_dir.join("commondir"))
                .map(|common| git_dir.join(common.trim()))
                .unwrap_or_else(|_| git_dir.clone());
            for candidate in [git_dir.join(&reference), common_dir.join(&reference), common_dir.join("packed-refs")] {
                if candidate.exists() {
                    println!("cargo:rerun-if-changed={}", candidate.display());
                }
            }
        }
    }
    let commit = std::env::var("ALGEON_GIT_COMMIT")
        .ok()
        .map(|commit| commit.trim().to_string())
        .filter(|commit| !commit.is_empty())
        .or_else(|| git_short_head(&manifest_dir));
    if let Some(commit) = commit {
        println!("cargo:rustc-env=ALGEON_BUILD_GIT_COMMIT={commit}");
    }
}

/// The `.git` directory above `start`. In a worktree `.git` is a file
/// holding `gitdir: <path>`.
fn find_git_dir(start: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|dir| {
        let candidate = dir.join(".git");
        if candidate.is_dir() {
            return Some(candidate);
        }
        let pointer = std::fs::read_to_string(&candidate).ok()?;
        let target = pointer.strip_prefix("gitdir:")?.trim();
        Some(dir.join(target))
    })
}

fn git_short_head(manifest_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(manifest_dir)
        .output()
        .ok()
        .filter(|output| output.status.success())?;
    let commit = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!commit.is_empty()).then_some(commit)
}
