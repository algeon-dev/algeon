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
        Ok(value) => normalized_cmake_bool(var_name, Some(&value)),
        Err(env::VarError::NotPresent) => normalized_cmake_bool(var_name, None),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{var_name} contained non-Unicode data")),
    }
}

pub fn normalized_cmake_bool(var_name: &str, value: Option<&str>) -> Result<Option<&'static str>, String> {
    value.map(|value| parse_cmake_bool(var_name, value)).transpose()
}

pub fn parse_cmake_bool(var_name: &str, value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_uppercase().as_str() {
        "1" | "ON" | "TRUE" | "YES" => Ok("ON"),
        "0" | "OFF" | "FALSE" | "NO" => Ok("OFF"),
        _ => Err(format!("{var_name} must be one of 1, 0, ON, OFF, TRUE, FALSE, YES, or NO; got {value:?}")),
    }
}

pub fn cmake_cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    cache.lines().find_map(|line| {
        let (cache_name_and_type, value) = line.split_once('=')?;
        let (cache_name, _) = cache_name_and_type.split_once(':')?;
        (cache_name == key).then_some(value)
    })
}

pub fn cached_cudf_discovery_matches_env(cache: &str) -> bool {
    let install_dir = env_path("CUDF_INSTALL_DIR");
    let build_dir = env_path("CUDF_CPP_BUILD_DIR");
    cached_cudf_discovery_matches(cache, install_dir.as_deref(), build_dir.as_deref())
}

pub fn cached_cudf_discovery_matches(cache: &str, install_dir: Option<&Path>, build_dir: Option<&Path>) -> bool {
    if let Some(install_dir) = install_dir {
        return cmake_cache_path_matches(cache, "CUDF_INSTALL_DIR", install_dir)
            && cmake_cache_value(cache, "CUDF_CPP_BUILD_DIR").is_none();
    }

    if let Some(build_dir) = build_dir {
        return cmake_cache_path_matches(cache, "CUDF_CPP_BUILD_DIR", build_dir)
            && cmake_cache_value(cache, "CUDF_INSTALL_DIR").is_none();
    }

    true
}

pub fn cmake_cache_path_matches(cache: &str, key: &str, expected: &Path) -> bool {
    let Some(cached) = cmake_cache_value(cache, key) else { return false };
    paths_equivalent(Path::new(cached), expected)
}

pub fn paths_equivalent(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

/// Resolve the exact libcudf shared library selected by the native discovery
/// contract. Installed prefixes use lib/lib64; a C++ build tree exposes
/// libcudf.so at its root.
pub fn locate_prebuilt_libcudf(install_dir: Option<&Path>, build_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(install_dir) = install_dir {
        for subdir in ["lib", "lib64"] {
            let candidate = install_dir.join(subdir).join("libcudf.so");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let candidate = build_dir?.join("libcudf.so");
    candidate.is_file().then_some(candidate)
}

/// Return the install prefix for a libcudf shared library in the conventional
/// `<prefix>/lib{,64}/libcudf.so` layout. A C++ build-tree library has no
/// install prefix and returns `None`.
pub fn installed_prefix_for_libcudf(library_path: &Path) -> Option<&Path> {
    let library_dir = library_path.parent()?;
    matches!(library_dir.file_name().and_then(|name| name.to_str()), Some("lib" | "lib64"))
        .then(|| library_dir.parent())
        .flatten()
}

fn env_path(var_name: &str) -> Option<PathBuf> {
    env::var(var_name).ok().map(PathBuf::from)
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

/// Explicit override for the clang-format executable used when
/// `CUDF_RUST_CLANG_LINT` is enabled. Prefer an absolute path so the tool
/// identity is independent of ambient `PATH` churn.
pub const CLANG_FORMAT_EXE_ENV: &str = "CUDF_RUST_CLANG_FORMAT";

/// Explicit override for the clang-tidy executable used when
/// `CUDF_RUST_CLANG_LINT` is enabled. Prefer an absolute path so the tool
/// identity is independent of ambient `PATH` churn.
pub const CLANG_TIDY_EXE_ENV: &str = "CUDF_RUST_CLANG_TIDY";

/// Optional pin for the CMake executable used by the native shim build.
pub const CMAKE_EXE_ENV: &str = "CUDF_RUST_CMAKE";

pub fn find_required_executable(tool: &str) -> Result<PathBuf, String> {
    let path = env::var_os("PATH")
        .ok_or_else(|| format!("required build-time tool `{tool}` was not found because PATH is unset"))?;
    find_executable_in_path(tool, path)
}

pub fn find_executable_in_path(tool: &str, path: OsString) -> Result<PathBuf, String> {
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
/// callers can fingerprint the real executable instead of ambient PATH.
pub fn resolve_tool_from_env_or_name(env_key: &str, default_name: &str) -> Result<PathBuf, String> {
    match env::var(env_key) {
        Ok(value) => resolve_explicit_tool_path(env_key, &value),
        Err(env::VarError::NotPresent) => find_required_executable(default_name),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{env_key} contained non-Unicode data")),
    }
}

/// Resolve `CC` / `CXX`-style compiler env vars. Basename values are resolved
/// once through PATH and returned as absolute paths so subsequent PATH-only
/// changes do not alter the selected toolchain without changing the env value
/// or the underlying binary identity.
pub fn resolve_compiler_from_env(env_key: &str) -> Result<Option<PathBuf>, String> {
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

/// Sibling libcudf build-tree cache used when discovery is not driven by an
/// install prefix. Path is relative to the `cudf-sys` manifest directory.
pub fn default_sibling_cudf_cmake_cache(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../cudf/cpp/build/CMakeCache.txt")
}

/// `CMakeCache.txt` that participates in the active libcudf discovery
/// contract, if any.
///
/// When `CUDF_INSTALL_DIR` is selected, an unrelated source-tree CMake cache
/// must not invalidate `cudf-sys`, even if `CUDF_CPP_BUILD_DIR` is also present.
/// An explicit build directory tracks only its own cache. With no override the
/// default sibling build tree is the active cache candidate.
pub fn active_cudf_cmake_cache(
    manifest_dir: &Path,
    install_dir: Option<&Path>,
    build_dir: Option<&Path>,
) -> Option<PathBuf> {
    if install_dir.is_some() {
        return None;
    }

    Some(match build_dir {
        Some(build_dir) => build_dir.join("CMakeCache.txt"),
        None => default_sibling_cudf_cmake_cache(manifest_dir),
    })
}

/// Env vars that form the explicit native rebuild contract after ambient PATH
/// was removed as a fingerprint input.
pub fn tracked_native_build_env_vars() -> &'static [&'static str] {
    &[
        "CC",
        "CXX",
        "CUDACXX",
        "CUDAHOSTCXX",
        "CUDF_INSTALL_DIR",
        "CUDF_CPP_BUILD_DIR",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDAToolkit_ROOT",
        "CUDA_STATIC_RUNTIME",
        "CMAKE",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_PREFIX_PATH",
        "CMAKE_LIBRARY_PATH",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUDAARCHS",
        "CUDF_RUST_CLANG_LINT",
        CLANG_FORMAT_EXE_ENV,
        CLANG_TIDY_EXE_ENV,
        CMAKE_EXE_ENV,
        "CUDF_RS_SKIP_STALENESS_CHECK",
    ]
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

fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    fs::metadata(path).map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0).unwrap_or(false)
}
