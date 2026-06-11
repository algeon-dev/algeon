use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

const CARGO_LINK_MANIFEST_RELATIVE_PATH: &str = "share/cudf_rust/cargo-link.txt";
const NATIVE_CLANG_LINT_ROOTS: [&str; 2] = ["native/src", "native/include"];
const CLANG_LINTABLE_EXTENSIONS: [&str; 4] = ["c", "cpp", "h", "hpp"];

pub fn cargo_link_manifest_path(install_prefix: &Path) -> PathBuf {
    install_prefix.join(CARGO_LINK_MANIFEST_RELATIVE_PATH)
}

pub fn read_cargo_link_directives(install_prefix: &Path) -> Result<Vec<String>, String> {
    let manifest_path = cargo_link_manifest_path(install_prefix);
    let manifest = fs::read_to_string(&manifest_path)
        .map_err(|error| format!("failed to read cargo link manifest {}: {error}", manifest_path.display()))?;
    parse_cargo_link_directives(&manifest).map_err(|error| format!("{error} ({})", manifest_path.display()))
}

pub fn parse_cargo_link_directives(manifest: &str) -> Result<Vec<String>, String> {
    let mut directives = Vec::new();

    for (index, raw_line) in manifest.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with("cargo:") {
            return Err(format!("invalid cargo directive at line {}", index + 1));
        }
        directives.push(line.to_owned());
    }

    if directives.is_empty() {
        return Err("cargo link manifest contained no directives".to_owned());
    }

    Ok(directives)
}

pub fn normalized_cmake_bool_from_env(var_name: &str) -> Result<Option<&'static str>, String> {
    match env::var(var_name) {
        Ok(value) => parse_cmake_bool(var_name, &value).map(Some),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{var_name} contained non-Unicode data")),
    }
}

pub fn parse_cmake_bool(var_name: &str, value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_uppercase().as_str() {
        "1" | "ON" | "TRUE" | "YES" => Ok("ON"),
        "0" | "OFF" | "FALSE" | "NO" => Ok("OFF"),
        _ => Err(format!("{var_name} must be one of 1, 0, ON, OFF, TRUE, FALSE, YES, or NO; got {value:?}")),
    }
}

pub fn ensure_supported_target_os(target_os: &str) -> Result<(), String> {
    if target_os == "linux" {
        return Ok(());
    }

    Err(format!("cudf-rs currently supports Linux only; unsupported target OS: {target_os}"))
}

pub fn native_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("native")
}

pub fn clang_tidy_config_path(manifest_dir: &Path) -> PathBuf {
    native_dir(manifest_dir).join(".clang-tidy")
}

pub fn clang_format_config_path(manifest_dir: &Path) -> PathBuf {
    manifest_dir.parent().unwrap_or(manifest_dir).join("cudf/.clang-format")
}

pub fn is_clang_lintable_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .as_deref()
        .map(|extension| CLANG_LINTABLE_EXTENSIONS.contains(&extension))
        .unwrap_or(false)
}

pub fn collect_native_clang_lint_files(manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut lint_files = Vec::new();

    for relative_dir in NATIVE_CLANG_LINT_ROOTS {
        collect_clang_lint_files_from_dir(&manifest_dir.join(relative_dir), &mut lint_files)?;
    }

    lint_files.sort();
    Ok(lint_files)
}

pub fn find_required_executable(tool: &str) -> Result<PathBuf, String> {
    let path = env::var_os("PATH")
        .ok_or_else(|| format!("required build-time tool `{tool}` was not found because PATH is unset"))?;
    find_executable_in_path(tool, path)
}

pub fn find_executable_in_path(tool: &str, path: OsString) -> Result<PathBuf, String> {
    for directory in env::split_paths(&path) {
        let candidate = directory.join(tool);
        if is_executable_file(&candidate) {
            return Ok(candidate);
        }
    }

    Err(format!("required build-time tool `{tool}` was not found in PATH"))
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

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path).map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0).unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}
