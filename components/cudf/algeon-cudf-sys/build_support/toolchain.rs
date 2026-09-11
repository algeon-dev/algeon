//! Toolchain selection and executable validation for native configuration.

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

/// Explicit override for the clang-format executable used when
/// `CUDF_RUST_CLANG_LINT` is enabled.
pub(crate) const CLANG_FORMAT_EXE_ENV: &str = "CUDF_RUST_CLANG_FORMAT";
/// Explicit override for the clang-tidy executable used when
/// `CUDF_RUST_CLANG_LINT` is enabled.
pub(crate) const CLANG_TIDY_EXE_ENV: &str = "CUDF_RUST_CLANG_TIDY";
/// Optional pin for the CMake executable used by the native shim build.
pub(crate) const CMAKE_EXE_ENV: &str = "CUDF_RUST_CMAKE";

pub(crate) fn ensure_supported_target_os(target_os: &str) -> Result<(), String> {
    if target_os == "linux" {
        return Ok(());
    }

    Err(format!("cudf currently supports Linux only; unsupported target OS: {target_os}"))
}

/// Validate the target and resolve the effective C/C++ compilers once before
/// CMake cache invalidation. The returned paths are absolute where possible.
pub(crate) fn prepare_cmake_toolchain() -> (PathBuf, PathBuf) {
    let target_os =
        env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|error| panic!("failed to read CARGO_CFG_TARGET_OS: {error}"));
    ensure_supported_target_os(&target_os).unwrap_or_else(|error| panic!("{error}"));

    let cc = resolve_compiler_from_env_or_name("CC", "cc").unwrap_or_else(|error| panic!("{error}"));
    let cxx = resolve_compiler_from_env_or_name("CXX", "c++").unwrap_or_else(|error| panic!("{error}"));

    // SAFETY: build scripts are single-threaded and cmake-rs reads these
    // standard variables only after this setup has completed.
    unsafe {
        env::set_var("CC", &cc);
        env::set_var("CXX", &cxx);
    }

    if env::var_os(CMAKE_EXE_ENV).is_some() {
        let cmake = resolve_tool_from_env_or_name(CMAKE_EXE_ENV, "cmake").unwrap_or_else(|error| panic!("{error}"));
        // SAFETY: same single-threaded build-script invariant as above.
        unsafe {
            env::set_var("CMAKE", cmake);
        }
    }

    (cc, cxx)
}

pub(crate) fn find_required_executable(tool: &str) -> Result<PathBuf, String> {
    let path = env::var_os("PATH")
        .ok_or_else(|| format!("required build-time tool `{tool}` was not found because PATH is unset"))?;
    find_executable_in_path(tool, path)
}

pub(crate) fn find_executable_in_path(tool: &str, path: OsString) -> Result<PathBuf, String> {
    for directory in env::split_paths(&path) {
        let candidate = directory.join(tool);
        if is_executable_file(&candidate) {
            return canonicalize_existing(&candidate).or(Ok(candidate));
        }
    }

    Err(format!("required build-time tool `{tool}` was not found in PATH"))
}

/// Resolve a build tool from an optional absolute env override, falling back
/// to a one-shot PATH lookup. The resolved path is canonical when possible so
/// callers fingerprint the actual executable instead of ambient PATH.
pub(crate) fn resolve_tool_from_env_or_name(env_key: &str, default_name: &str) -> Result<PathBuf, String> {
    match env::var(env_key) {
        Ok(value) => resolve_explicit_tool_path(env_key, &value),
        Err(env::VarError::NotPresent) => find_required_executable(default_name),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{env_key} contained non-Unicode data")),
    }
}

/// Resolve `CC` / `CXX`-style compiler env vars. Basename values are resolved
/// once through PATH and returned as absolute paths so later PATH-only changes
/// do not alter the selected toolchain.
pub(crate) fn resolve_compiler_from_env(env_key: &str) -> Result<Option<PathBuf>, String> {
    match env::var(env_key) {
        Ok(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Err(format!("{env_key} is set but empty"));
            }
            if value.contains(char::is_whitespace) {
                return Err(format!("{env_key} must be a single executable path without whitespace; got {value:?}"));
            }
            let path = PathBuf::from(value);
            if path.components().count() > 1 {
                return resolve_explicit_tool_path(env_key, value).map(Some);
            }
            find_required_executable(value).map(Some)
        },
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{env_key} contained non-Unicode data")),
    }
}

pub(crate) fn resolve_compiler_from_env_or_name(env_key: &str, default_name: &str) -> Result<PathBuf, String> {
    match resolve_compiler_from_env(env_key)? {
        Some(compiler) => Ok(compiler),
        None => find_required_executable(default_name),
    }
}

fn resolve_explicit_tool_path(env_key: &str, value: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value.trim());
    if path.as_os_str().is_empty() {
        return Err(format!("{env_key} is set but empty"));
    }
    if !path.is_absolute() {
        return Err(format!("{env_key} must be an absolute path to a build tool executable; got {}", path.display()));
    }
    if !is_executable_file(&path) {
        return Err(format!("{env_key} does not point at an executable file: {}", path.display()));
    }
    canonicalize_existing(&path).or(Ok(path))
}

fn canonicalize_existing(path: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| format!("failed to canonicalize {}: {error}", path.display()))
}

fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path).map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0).unwrap_or(false)
}
