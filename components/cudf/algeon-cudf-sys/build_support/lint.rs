//! Native source collection and opt-in clang formatting configuration.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

const NATIVE_CLANG_LINT_ROOTS: [&str; 2] = ["native/src", "native/include"];
const CLANG_LINTABLE_EXTENSIONS: [&str; 4] = ["c", "cpp", "h", "hpp"];

pub(crate) fn native_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("native")
}

pub(crate) fn clang_tidy_config_path(manifest_dir: &Path) -> PathBuf {
    native_dir(manifest_dir).join(".clang-tidy")
}

pub(crate) fn clang_format_config_path(manifest_dir: &Path) -> PathBuf {
    manifest_dir.parent().expect("cudf-sys manifest directory must have a repository parent").join("cudf/.clang-format")
}

pub(crate) fn is_clang_lintable_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
        .map(|extension| CLANG_LINTABLE_EXTENSIONS.contains(&extension))
        .unwrap_or(false)
}

pub(crate) fn collect_native_clang_lint_files(manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut lint_files = Vec::new();

    for relative_dir in NATIVE_CLANG_LINT_ROOTS {
        collect_clang_lint_files_from_dir(&manifest_dir.join(relative_dir), &mut lint_files)?;
    }

    lint_files.sort();
    Ok(lint_files)
}

pub(crate) fn run_clang_format_check(clang_format: &Path, config_path: &Path, lint_files: &[PathBuf]) {
    ensure_file_exists(config_path, "clang-format config");

    let mut command = Command::new(clang_format);
    command.arg("--dry-run").arg("--Werror");
    command.arg(format!("--style=file:{}", config_path.display()));
    for file in lint_files {
        command.arg(file);
    }

    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run clang-format from {}: {error}", clang_format.display()));
    if output.status.success() {
        return;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    panic!(
        "clang-format check failed for cudf-sys/native using config {}.\nstdout:\n{}\nstderr:\n{}",
        config_path.display(),
        stdout,
        stderr
    );
}

pub(crate) fn ensure_file_exists(path: &Path, description: &str) {
    if path.is_file() {
        return;
    }

    panic!("required {description} was not found at {}", path.display());
}

fn collect_clang_lint_files_from_dir(dir: &Path, lint_files: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|error| format!("failed to read clang lint directory {}: {error}", dir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|error| format!("failed to read entry in {}: {error}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_clang_lint_files_from_dir(&path, lint_files)?;
        } else if is_clang_lintable_path(&path) {
            lint_files.push(path);
        }
    }

    Ok(())
}
