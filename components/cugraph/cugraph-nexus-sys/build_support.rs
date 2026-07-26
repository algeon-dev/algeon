#![cfg_attr(test, allow(dead_code))]

use std::{
    collections::BTreeSet,
    env,
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const ETXTBSY_RETRY_ATTEMPTS: usize = 6;
const ETXTBSY_RETRY_DELAY: Duration = Duration::from_millis(25);

struct WorkspaceNativeBuildLayout {
    src_dir: PathBuf,
    build_dir: PathBuf,
    cudf_install_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NativeBuildContract {
    pub(crate) native_include_dirs: Vec<PathBuf>,
    pub(crate) native_link_dirs: Vec<PathBuf>,
    pub(crate) runtime_lib_dirs: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedLinkCommand {
    pub(crate) link_dirs: Vec<PathBuf>,
    pub(crate) runtime_dirs: Vec<PathBuf>,
}

pub(crate) fn explicit_native_contract_count(
    include_dirs_env: Option<&OsString>,
    link_dirs_env: Option<&OsString>,
    runtime_lib_dirs_env: Option<&OsString>,
) -> usize {
    [include_dirs_env, link_dirs_env, runtime_lib_dirs_env].into_iter().filter(|value| value.is_some()).count()
}

pub(crate) fn prepare_workspace_cmake_export_dir(
    manifest_dir: &Path,
    cpp_build_dir_env: Option<OsString>,
    cudf_install_dir_env: Option<OsString>,
    auto_build_env: Option<OsString>,
    cuda_architectures_env: Option<OsString>,
) -> Result<PathBuf, String> {
    let layout = workspace_native_build_layout(manifest_dir, cpp_build_dir_env, cudf_install_dir_env)?;
    let cuda_architectures = resolved_cuda_architectures(cuda_architectures_env.as_deref());
    let auto_build_enabled = auto_native_build_enabled(auto_build_env.as_deref())?;
    if layout.build_dir.join("cugraph-config.cmake").is_file() {
        if let Some(reason) = workspace_build_reconfigure_reason(&layout, &cuda_architectures)? {
            if !auto_build_enabled {
                return Err(format!(
                    "The workspace-local cuGraph CMake build at '{}' must be reconfigured ({reason}), but automatic \
                     native builds are disabled by CUGRAPH_SYS_AUTO_BUILD. Reconfigure native cuGraph manually, \
                     unset CUGRAPH_SYS_AUTO_BUILD, or set CUGRAPH_CMAKE_BUILD_DIR / the explicit CUGRAPH_NATIVE_* \
                     contract.",
                    layout.build_dir.display()
                ));
            }
            build_workspace_cugraph_cpp(&layout, &cuda_architectures, &reason)?;
        }
        validate_workspace_runtime_packages(&layout)?;
        return validate_cmake_export_dir(&layout.build_dir, "workspace-local cuGraph CMake build directory");
    }

    if !auto_build_enabled {
        return Err(format!(
            "Could not resolve the cuGraph native contract from '{}', and automatic native builds are disabled by \
             CUGRAPH_SYS_AUTO_BUILD. Either build native cuGraph first, set CUGRAPH_CMAKE_BUILD_DIR, or set all of \
             CUGRAPH_NATIVE_INCLUDE_DIRS, CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS.",
            layout.build_dir.display()
        ));
    }

    build_workspace_cugraph_cpp(&layout, &cuda_architectures, "CMake export is missing")?;
    validate_workspace_runtime_packages(&layout)?;
    validate_cmake_export_dir(&layout.build_dir, "workspace-local cuGraph CMake build directory")
}

pub(crate) fn resolve_native_build_contract(
    manifest_dir: &Path,
    include_dirs_env: Option<OsString>,
    link_dirs_env: Option<OsString>,
    runtime_lib_dirs_env: Option<OsString>,
    cmake_build_dir_env: Option<OsString>,
) -> Result<NativeBuildContract, String> {
    let explicit_contract_count = explicit_native_contract_count(
        include_dirs_env.as_ref(),
        link_dirs_env.as_ref(),
        runtime_lib_dirs_env.as_ref(),
    );

    if explicit_contract_count == 3 {
        return Ok(NativeBuildContract {
            native_include_dirs: resolve_native_include_dirs(include_dirs_env)?,
            native_link_dirs: resolve_native_link_dirs(link_dirs_env)?,
            runtime_lib_dirs: resolve_runtime_lib_dirs(runtime_lib_dirs_env)?,
        });
    }

    if explicit_contract_count != 0 {
        return Err("The native contract must come from exactly one source. Set all of CUGRAPH_NATIVE_INCLUDE_DIRS, \
             CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS together, or set none of them and use \
             a formal CMake export root via CUGRAPH_CMAKE_BUILD_DIR or the workspace-local cugraph/cpp/build export."
            .to_string());
    }

    let cmake_export_dir = resolve_cmake_export_dir(manifest_dir, cmake_build_dir_env)?;
    resolve_native_build_contract_from_cmake_export(&cmake_export_dir)
}

pub(crate) fn resolve_native_include_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let include_dirs = resolve_required_path_list("CUGRAPH_NATIVE_INCLUDE_DIRS", env_val)?;
    primary_cugraph_include_dir(&include_dirs)?;
    Ok(include_dirs)
}

pub(crate) fn resolve_native_link_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let link_dirs = resolve_required_path_list("CUGRAPH_NATIVE_LINK_DIRS", env_val)?;
    ensure_contains_cugraph_libraries("CUGRAPH_NATIVE_LINK_DIRS", &link_dirs)?;
    Ok(link_dirs)
}

pub(crate) fn resolve_runtime_lib_dirs(env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let runtime_dirs = resolve_required_path_list("CUGRAPH_RUNTIME_LIB_DIRS", env_val)?;
    ensure_contains_cugraph_libraries("CUGRAPH_RUNTIME_LIB_DIRS", &runtime_dirs)?;
    Ok(runtime_dirs)
}

pub(crate) fn primary_cugraph_include_dir(include_dirs: &[PathBuf]) -> Result<PathBuf, String> {
    include_dirs
        .iter()
        .find(|path| path.join("cugraph_c").is_dir())
        .cloned()
        .ok_or_else(|| {
            format!(
                "CUGRAPH_NATIVE_INCLUDE_DIRS is invalid. Expected at least one directory containing cugraph_c/. Provided paths: {}.",
                display_paths(include_dirs)
            )
        })
}

pub(crate) fn resolve_cmake_export_dir(manifest_dir: &Path, env_val: Option<OsString>) -> Result<PathBuf, String> {
    if let Some(raw_path) = env_val {
        return validate_cmake_export_dir(Path::new(&raw_path), "CUGRAPH_CMAKE_BUILD_DIR");
    }

    let candidate = manifest_dir.join("../cugraph/cpp/build");
    if candidate.join("cugraph-config.cmake").is_file() {
        return Ok(candidate.canonicalize().unwrap_or(candidate));
    }

    Err(format!(
        "Could not resolve the cuGraph native contract. Either set all of CUGRAPH_NATIVE_INCLUDE_DIRS, \
         CUGRAPH_NATIVE_LINK_DIRS, and CUGRAPH_RUNTIME_LIB_DIRS together, or point CUGRAPH_CMAKE_BUILD_DIR \
         at a build/install directory containing cugraph-config.cmake. The workspace-local export root \
         '{}' was not usable.",
        candidate.display()
    ))
}

#[cfg_attr(test, allow(dead_code))]
pub(crate) fn tracked_cmake_export_files(cmake_build_dir: &Path) -> Vec<PathBuf> {
    vec![
        cmake_build_dir.join("cugraph-config.cmake"),
        cmake_build_dir.join("CMakeCache.txt"),
        cmake_build_dir.join("cugraph-targets.cmake"),
        cmake_build_dir.join("cugraph-dependencies.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-config.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-targets.cmake"),
        cmake_build_dir.join("_deps/rmm-build/rmm-dependencies.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-config.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-targets.cmake"),
        cmake_build_dir.join("_deps/raft-build/raft-dependencies.cmake"),
    ]
}

fn workspace_native_build_layout(
    manifest_dir: &Path,
    cpp_build_dir_env: Option<OsString>,
    cudf_install_dir_env: Option<OsString>,
) -> Result<WorkspaceNativeBuildLayout, String> {
    let repo_root = manifest_dir.parent().ok_or_else(|| {
        format!("Could not determine the repository root from CARGO_MANIFEST_DIR='{}'.", manifest_dir.display())
    })?;
    let workspace_parent = repo_root.parent().unwrap_or(repo_root);
    let src_dir = repo_root.join("cugraph/cpp");
    if !src_dir.join("CMakeLists.txt").is_file() {
        return Err(format!(
            "The cuGraph C++ source tree was not found at '{}'. Initialize/update the cugraph submodule, or set \
             CUGRAPH_CMAKE_BUILD_DIR / the explicit CUGRAPH_NATIVE_* contract to a prebuilt cuGraph.",
            src_dir.display()
        ));
    }

    let build_dir = cpp_build_dir_env.map(PathBuf::from).unwrap_or_else(|| src_dir.join("build"));
    let cudf_install_dir =
        cudf_install_dir_env.map(PathBuf::from).unwrap_or_else(|| workspace_parent.join("cudf-nexus/cudf-install"));
    let rmm_config = cudf_install_dir.join("lib/cmake/rmm/rmm-config.cmake");
    if !rmm_config.is_file() {
        return Err(format!(
            "The canonical cudf install prefix was not usable: '{}' does not exist. Build and install libcudf first \
             or set CUDF_INSTALL_DIR to a prefix that exports lib/cmake/rmm/rmm-config.cmake.",
            rmm_config.display()
        ));
    }
    let rapids_logger_config = cudf_install_dir.join("lib/cmake/rapids_logger/rapids_logger-config.cmake");
    if !rapids_logger_config.is_file() {
        return Err(format!(
            "The canonical cudf install prefix was not usable: '{}' does not exist. Build and install libcudf first \
             or set CUDF_INSTALL_DIR to a prefix that exports lib/cmake/rapids_logger/rapids_logger-config.cmake.",
            rapids_logger_config.display()
        ));
    }

    Ok(WorkspaceNativeBuildLayout { src_dir, build_dir, cudf_install_dir })
}

fn auto_native_build_enabled(env_val: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    let Some(raw) = env_val else {
        return Ok(true);
    };
    let value = raw.to_string_lossy();
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!("CUGRAPH_SYS_AUTO_BUILD='{}' is invalid. Use 1/true/on or 0/false/off.", value)),
    }
}

fn build_workspace_cugraph_cpp(
    layout: &WorkspaceNativeBuildLayout,
    cuda_architectures: &std::ffi::OsStr,
    reason: &str,
) -> Result<(), String> {
    println!("cargo:warning=Configuring native cuGraph at '{}' because {reason}.", layout.build_dir.display(),);

    let rmm_root = layout.cudf_install_dir.as_os_str();
    let rapids_logger_root = layout.cudf_install_dir.as_os_str();

    let mut configure_command = Command::new("cmake");
    configure_command
        .arg("-S")
        .arg(&layout.src_dir)
        .arg("-B")
        .arg(&layout.build_dir)
        .arg("-GNinja")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={}", cuda_architectures.to_string_lossy()))
        .arg("-DBUILD_TESTS=OFF")
        .arg("-DBUILD_CUGRAPH_MG_TESTS=OFF")
        .args([
            "-U",
            "rmm_DIR",
            "-U",
            "CPM_PACKAGE_rmm_*",
            "-U",
            "RMM_*",
            "-U",
            "FETCHCONTENT_SOURCE_DIR_RMM",
            "-U",
            "FETCHCONTENT_UPDATES_DISCONNECTED_RMM",
            "-U",
            "rapids_logger_DIR",
            "-U",
            "CPM_PACKAGE_rapids_logger_*",
            "-U",
            "RAPIDS_LOGGER_*",
            "-U",
            "FETCHCONTENT_SOURCE_DIR_RAPIDS_LOGGER",
            "-U",
            "FETCHCONTENT_UPDATES_DISCONNECTED_RAPIDS_LOGGER",
        ])
        .arg(format!("-Drmm_ROOT={}", Path::new(rmm_root).display()))
        .arg(format!("-Drapids_logger_ROOT={}", Path::new(rapids_logger_root).display()));

    run_command_capture("configure native cuGraph with CMake", &mut configure_command)?;
    validate_workspace_runtime_packages(layout)?;

    let mut build_command = Command::new("cmake");
    build_command.arg("--build").arg(&layout.build_dir);
    run_command_capture("build native cuGraph with CMake", &mut build_command)?;
    reject_stale_private_runtime_artifacts(&layout.build_dir)
}

fn validate_workspace_runtime_packages(layout: &WorkspaceNativeBuildLayout) -> Result<(), String> {
    validate_cmake_cache_package_dir(&layout.build_dir, "rmm_DIR", &layout.cudf_install_dir.join("lib/cmake/rmm"))?;
    validate_cmake_cache_package_dir(
        &layout.build_dir,
        "rapids_logger_DIR",
        &layout.cudf_install_dir.join("lib/cmake/rapids_logger"),
    )?;

    reject_stale_private_runtime_artifacts(&layout.build_dir)
}

fn resolved_cuda_architectures(env_val: Option<&std::ffi::OsStr>) -> OsString {
    env_val.filter(|value| !value.is_empty()).map(OsString::from).unwrap_or_else(|| OsString::from("NATIVE"))
}

fn workspace_build_reconfigure_reason(
    layout: &WorkspaceNativeBuildLayout,
    expected_cuda_architectures: &std::ffi::OsStr,
) -> Result<Option<String>, String> {
    let cache_path = layout.build_dir.join("CMakeCache.txt");
    let cache = match fs::read_to_string(&cache_path) {
        Ok(cache) => cache,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(Some(format!("{} is missing", cache_path.display())));
        },
        Err(error) => {
            return Err(format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()));
        },
    };

    let expected_cuda_architectures = expected_cuda_architectures.to_string_lossy();
    match cmake_cache_value(&cache, "CMAKE_CUDA_ARCHITECTURES") {
        Some(cached) if cached == expected_cuda_architectures => {},
        Some(cached) => {
            return Ok(Some(format!(
                "CMAKE_CUDA_ARCHITECTURES changed from '{}' to '{}'",
                cached, expected_cuda_architectures
            )));
        },
        None => {
            return Ok(Some(format!("CMAKE_CUDA_ARCHITECTURES is missing from {}", cache_path.display())));
        },
    }

    for (cache_key, expected_dir) in [
        ("rmm_DIR", layout.cudf_install_dir.join("lib/cmake/rmm")),
        ("rapids_logger_DIR", layout.cudf_install_dir.join("lib/cmake/rapids_logger")),
    ] {
        let Some(cached) = cmake_cache_value(&cache, cache_key) else {
            return Ok(Some(format!("{cache_key} is missing from {}", cache_path.display())));
        };
        if !paths_equivalent(Path::new(cached), &expected_dir) {
            return Ok(Some(format!("{cache_key} changed from '{}' to '{}'", cached, expected_dir.display())));
        }
    }

    Ok(None)
}

fn validate_cmake_cache_package_dir(build_dir: &Path, cache_key: &str, expected_dir: &Path) -> Result<(), String> {
    let cache_path = build_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .map_err(|error| format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()))?;
    let resolved = cmake_cache_value(&cache, cache_key).ok_or_else(|| {
        format!(
            "The cuGraph native CMake cache '{}' does not contain {cache_key}. Reconfigure native cuGraph with the \
             canonical cudf install prefix.",
            cache_path.display()
        )
    })?;
    let resolved = Path::new(resolved);
    if paths_equivalent(resolved, expected_dir) {
        return Ok(());
    }

    Err(format!(
        "The cuGraph native build resolved {cache_key}='{}' instead of '{}'. Reconfigure native cuGraph with \
         CUDF_INSTALL_DIR pointing at the canonical cudf install prefix so cudf and cugraph share one RMM build.",
        resolved.display(),
        expected_dir.display()
    ))
}

fn reject_stale_private_runtime_artifacts(build_dir: &Path) -> Result<(), String> {
    let stale_artifacts = stale_private_runtime_artifacts(build_dir);
    if stale_artifacts.is_empty() {
        return Ok(());
    }

    Err(format!(
        "Stale private RMM/rapids_logger artifacts remain under '{}':\n{}\nRemove the stale build tree and rerun \
         cargo build so the loader cannot resolve a private RAPIDS runtime copy.",
        build_dir.join("_deps").display(),
        stale_artifacts.iter().map(|path| format!("  {}", path.display())).collect::<Vec<_>>().join("\n")
    ))
}

fn stale_private_runtime_artifacts(build_dir: &Path) -> Vec<PathBuf> {
    let deps_dir = build_dir.join("_deps");
    let mut artifacts = Vec::new();
    let mut stack = vec![(deps_dir, 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                if depth < 3 {
                    stack.push((path, depth + 1));
                }
                continue;
            }
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("librmm.so") || name.starts_with("librapids_logger.so") {
                artifacts.push(path);
            }
        }
    }
    artifacts.sort();
    artifacts
}

fn run_command_capture(action: &str, command: &mut Command) -> Result<(), String> {
    let command_display = format_command(command);
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to {action} using `{command_display}`: {error}"))?;

    if output.status.success() {
        return Ok(());
    }

    Err(format!(
        "Failed to {action} using `{command_display}`; status {}.\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

pub(crate) fn parse_cmake_generated_path_list(contents: &str) -> Result<Vec<PathBuf>, String> {
    let mut resolved = Vec::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        push_unique_path(&mut resolved, normalize_existing_dir(Path::new(trimmed))?);
    }

    Ok(resolved)
}

pub(crate) fn parse_cmake_link_command(command: &str) -> Result<ParsedLinkCommand, String> {
    let mut link_dirs = Vec::new();
    let mut runtime_dirs = Vec::new();

    for token in split_shell_words(command)? {
        if let Some(raw_paths) = token.strip_prefix("-Wl,-rpath,") {
            for path in env::split_paths(&OsString::from(raw_paths)) {
                push_unique_path(&mut runtime_dirs, normalize_existing_dir(&path)?);
            }
            continue;
        }

        if let Some(raw_path) = token.strip_prefix("-L") {
            push_unique_path(&mut link_dirs, normalize_existing_dir(Path::new(raw_path))?);
            continue;
        }

        let path = Path::new(&token);
        if path.is_absolute()
            && looks_like_library_path(path)
            && let Some(parent) = path.parent()
        {
            push_unique_path(&mut link_dirs, normalize_existing_dir(parent)?);
        }
    }

    Ok(ParsedLinkCommand { link_dirs, runtime_dirs })
}

pub(crate) fn resolve_compiler_probe_command(env_val: Option<&str>) -> Result<(String, Vec<String>), String> {
    let raw = env_val.unwrap_or("cc").trim();
    if raw.is_empty() {
        return Err(
            "CC is set but empty. Set CC to the compiler used for this build, or unset CC to fall back to 'cc'."
                .to_string(),
        );
    }

    let parts = split_shell_words(raw)?;
    let (compiler, args) = parts.split_first().ok_or_else(|| {
        "Could not determine a compiler command from CC. Set CC to the compiler used for this build, or unset CC to fall back to 'cc'."
            .to_string()
    })?;

    Ok((compiler.clone(), args.to_vec()))
}

pub(crate) fn detect_system_include_dir(env_val: Option<&str>) -> Result<PathBuf, String> {
    let (compiler, compiler_args) = resolve_compiler_probe_command(env_val)?;
    let mut command = Command::new(&compiler);
    command
        .args(&compiler_args)
        .args(["-v", "-E", "-x", "c", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let compiler_display = format_compiler_command(&compiler, &compiler_args);
    let output = command_output_with_etxtbsy_retry(&mut command).map_err(|err| {
        format!(
            "Failed to run C compiler '{}' while probing system include paths: {err}. Set CC to the compiler used for this build and ensure it is installed.",
            compiler_display
        )
    })?;

    if !output.status.success() {
        return Err(format!(
            "C compiler '{}' exited with status {} while probing system include paths. Set CC to the compiler used for this build and ensure it supports '-v -E -x c -'.",
            compiler_display, output.status
        ));
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_compiler_include_dir(&stderr).ok_or_else(|| {
        format!(
            "Could not determine a system include directory from compiler '{}'. Set CC to the compiler used for this build and ensure it reports a standard '#include <...> search starts here:' list.",
            compiler_display
        )
    })
}

pub(crate) fn has_required_cugraph_libs(dir: &Path) -> bool {
    has_library(dir, "cugraph") && has_library(dir, "cugraph_c")
}

fn resolve_native_build_contract_from_cmake_export(cmake_build_dir: &Path) -> Result<NativeBuildContract, String> {
    let probe_dir = create_probe_dir()?;
    let result = resolve_native_build_contract_from_cmake_export_inner(cmake_build_dir, &probe_dir);
    let _ = fs::remove_dir_all(&probe_dir);
    result
}

fn resolve_native_build_contract_from_cmake_export_inner(
    cmake_build_dir: &Path,
    probe_dir: &Path,
) -> Result<NativeBuildContract, String> {
    write_contract_probe_files(cmake_build_dir, probe_dir)?;
    let package_prefixes = cmake_package_prefixes_from_cache(cmake_build_dir)?;
    run_cmake_probe(probe_dir, &package_prefixes)?;

    let include_output = fs::read_to_string(probe_dir.join("build/includes.txt"))
        .map_err(|error| format!("Failed to read generated include metadata from CMake probe: {error}"))?;
    let includes = parse_cmake_generated_path_list(&include_output)?;

    let link_command = find_cmake_link_command(&probe_dir.join("build"))?;
    let parsed_link = parse_cmake_link_command(&link_command)?;

    Ok(NativeBuildContract {
        native_include_dirs: resolve_native_include_dirs(Some(
            env::join_paths(&includes).expect("CMake probe include paths must be valid OS strings"),
        ))?,
        native_link_dirs: resolve_native_link_dirs(Some(
            env::join_paths(&parsed_link.link_dirs).expect("CMake probe link paths must be valid OS strings"),
        ))?,
        runtime_lib_dirs: resolve_runtime_lib_dirs(Some(
            env::join_paths(&parsed_link.runtime_dirs).expect("CMake probe runtime paths must be valid OS strings"),
        ))?,
    })
}

fn validate_cmake_export_dir(path: &Path, env_name: &str) -> Result<PathBuf, String> {
    let canonical = path.canonicalize().map_err(|_| {
        format!(
            "{env_name}='{}' does not exist or is not accessible. Expected a directory containing cugraph-config.cmake.",
            path.display()
        )
    })?;

    if !canonical.join("cugraph-config.cmake").is_file() {
        return Err(format!(
            "{env_name}='{}' is invalid. Expected '{}/cugraph-config.cmake' to exist.",
            path.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

fn resolve_required_path_list(env_name: &str, env_val: Option<OsString>) -> Result<Vec<PathBuf>, String> {
    let Some(raw_paths) = env_val else {
        return Err(format!(
            "{env_name} is required. Set it to a non-empty OS-native path list that describes the native cuGraph contract."
        ));
    };

    let mut resolved = Vec::new();
    for raw_path in env::split_paths(&raw_paths) {
        let path = raw_path.as_path();
        if path.as_os_str().is_empty() {
            continue;
        }
        push_unique_path(&mut resolved, normalize_existing_dir(path)?);
    }

    if resolved.is_empty() {
        return Err(format!(
            "{env_name} is required. Set it to a non-empty OS-native path list that describes the native cuGraph contract."
        ));
    }

    Ok(resolved)
}

fn ensure_contains_cugraph_libraries(env_name: &str, dirs: &[PathBuf]) -> Result<(), String> {
    if dirs.iter().any(|dir| has_required_cugraph_libs(dir)) {
        Ok(())
    } else {
        Err(format!(
            "{env_name} is invalid. Expected at least one directory containing both libcugraph and libcugraph_c shared libraries. Provided paths: {}.",
            display_paths(dirs)
        ))
    }
}

pub(crate) fn cmake_package_prefixes_from_cache(cmake_build_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let cache_path = cmake_build_dir.join("CMakeCache.txt");
    let contents = match fs::read_to_string(&cache_path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(format!("Failed to read cuGraph native CMake cache '{}': {error}", cache_path.display()));
        },
    };

    let mut prefixes = Vec::new();
    let mut seen = BTreeSet::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let Some((name_and_type, value)) = line.split_once('=') else {
            continue;
        };
        let Some((name, _type_name)) = name_and_type.split_once(':') else {
            continue;
        };
        if name != "CMAKE_PREFIX_PATH" && !name.ends_with("_ROOT") {
            continue;
        }

        for raw_path in value.split(';').map(str::trim).filter(|path| !path.is_empty()) {
            if raw_path.ends_with("-NOTFOUND") {
                continue;
            }
            let path = Path::new(raw_path);
            if !path.is_dir() {
                continue;
            }
            let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if seen.insert(canonical.clone()) {
                prefixes.push(canonical);
            }
        }
    }

    Ok(prefixes)
}

fn cmake_cache_value<'a>(cache: &'a str, key: &str) -> Option<&'a str> {
    for line in cache.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let Some((name_and_type, value)) = line.split_once('=') else {
            continue;
        };
        let Some((name, _type_name)) = name_and_type.split_once(':') else {
            continue;
        };
        if name == key {
            return Some(value.trim());
        }
    }
    None
}

fn paths_equivalent(left: &Path, right: &Path) -> bool {
    let left = left.canonicalize().unwrap_or_else(|_| left.to_path_buf());
    let right = right.canonicalize().unwrap_or_else(|_| right.to_path_buf());
    left == right
}

fn create_probe_dir() -> Result<PathBuf, String> {
    let unique = format!(
        "cugraph-cmake-probe-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
    );
    let probe_dir = env::temp_dir().join(unique);
    fs::create_dir_all(&probe_dir).map_err(|error| {
        format!(
            "Failed to create temporary directory for the cuGraph CMake metadata probe at '{}': {error}",
            probe_dir.display()
        )
    })?;
    Ok(probe_dir)
}

fn write_contract_probe_files(cmake_build_dir: &Path, probe_dir: &Path) -> Result<(), String> {
    let cmake_lists = format!(
        r#"cmake_minimum_required(VERSION 3.30)
project(cugraph_contract_probe LANGUAGES CXX CUDA)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(cugraph CONFIG REQUIRED PATHS "{build_dir}" NO_DEFAULT_PATH)
find_package(rmm CONFIG REQUIRED)
find_package(raft CONFIG REQUIRED)
add_executable(contract_probe probe.cpp)
target_link_libraries(contract_probe PRIVATE cugraph::cugraph_c rmm::rmm raft::raft)
file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/includes.txt" CONTENT "$<JOIN:$<TARGET_PROPERTY:contract_probe,INCLUDE_DIRECTORIES>,\n>")
"#,
        build_dir = cmake_build_dir.display(),
    );
    // Probe representative cugraph_c/RAFT/RMM headers. per_device_resource.hpp
    // exists in both RMM 26.04 and 26.06, unlike the removed polymorphic
    // device_memory_resource.hpp.
    let probe_cpp = r#"#include <cugraph_c/resource_handle.h>
#include <raft/core/handle.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <cuda_runtime_api.h>
int main() { return 0; }
"#;

    fs::write(probe_dir.join("CMakeLists.txt"), cmake_lists)
        .map_err(|error| format!("Failed to write temporary CMakeLists.txt for the cuGraph metadata probe: {error}"))?;
    fs::write(probe_dir.join("probe.cpp"), probe_cpp)
        .map_err(|error| format!("Failed to write temporary probe.cpp for the cuGraph metadata probe: {error}"))?;

    Ok(())
}

fn run_cmake_probe(probe_dir: &Path, package_prefixes: &[PathBuf]) -> Result<(), String> {
    let mut configure_command = Command::new("cmake");
    configure_command.arg("-S").arg(probe_dir).arg("-B").arg(probe_dir.join("build")).arg("-G").arg("Unix Makefiles");
    if !package_prefixes.is_empty() {
        configure_command.arg(format!("-DCMAKE_PREFIX_PATH={}", cmake_list(package_prefixes)));
    }

    let configure_output = configure_command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to run cmake while probing the cuGraph CMake export: {error}"))?;

    if !configure_output.status.success() {
        return Err(format!(
            "Failed to configure the cuGraph CMake metadata probe.\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&configure_output.stdout),
            String::from_utf8_lossy(&configure_output.stderr)
        ));
    }

    let build_output = Command::new("cmake")
        .arg("--build")
        .arg(probe_dir.join("build"))
        .arg("--target")
        .arg("contract_probe")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|error| format!("Failed to build the cuGraph CMake metadata probe: {error}"))?;

    if !build_output.status.success() {
        return Err(format!(
            "The cuGraph CMake export did not provide a complete native closure for the probe target.\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&build_output.stdout),
            String::from_utf8_lossy(&build_output.stderr)
        ));
    }

    Ok(())
}

fn cmake_list(paths: &[PathBuf]) -> String {
    paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(";")
}

fn find_cmake_link_command(build_dir: &Path) -> Result<String, String> {
    let mut stack = vec![build_dir.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|error| {
            format!("Failed to inspect generated CMake build directory '{}': {error}", dir.display())
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| format!("Failed to inspect entry under '{}': {error}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.file_name().is_some_and(|name| name == "link.txt") {
                return fs::read_to_string(&path).map_err(|error| {
                    format!("Failed to read generated CMake link command from '{}': {error}", path.display())
                });
            }
        }
    }

    Err(format!("Failed to locate the generated CMake link command under '{}'.", build_dir.display()))
}

fn normalize_existing_dir(path: &Path) -> Result<PathBuf, String> {
    let canonical =
        path.canonicalize().map_err(|_| format!("Path '{}' does not exist or is not accessible.", path.display()))?;
    if !canonical.is_dir() {
        return Err(format!("Path '{}' is not a directory.", canonical.display()));
    }
    Ok(canonical)
}

fn looks_like_library_path(path: &Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    file_name.ends_with(".so")
        || file_name.contains(".so.")
        || file_name.ends_with(".dylib")
        || file_name.ends_with(".dll")
        || file_name.ends_with(".lib")
        // Static archives (e.g. CUDA's `libcudart_static.a`) are linked by the
        // CMake export via absolute paths. Capture their directory so the
        // toolkit lib dir flows through the formal native-link metadata; the
        // same directory also holds the corresponding shared libraries that
        // the cugraph-nexus build script (`cugraph-nexus/build.rs`) links (e.g. `-lcudart`).
        || file_name.ends_with(".a")
}

fn has_library(dir: &Path, stem: &str) -> bool {
    let exact_names = [
        format!("lib{stem}.so"),
        format!("lib{stem}.dylib"),
        format!("lib{stem}.dll"),
        format!("lib{stem}.lib"),
        format!("{stem}.dll"),
        format!("{stem}.lib"),
    ];

    if exact_names.iter().any(|name| dir.join(name).exists()) {
        return true;
    }

    let versioned_prefix = format!("lib{stem}.so.");
    let Ok(entries) = dir.read_dir() else {
        return false;
    };

    entries.flatten().any(|entry| entry.file_name().to_str().is_some_and(|name| name.starts_with(&versioned_prefix)))
}

fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
}

fn display_paths(paths: &[PathBuf]) -> String {
    paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ")
}

fn command_output_with_etxtbsy_retry(command: &mut Command) -> io::Result<std::process::Output> {
    let mut last_error = None;

    for attempt in 0..ETXTBSY_RETRY_ATTEMPTS {
        match command.output() {
            Ok(output) => return Ok(output),
            Err(err) if err.raw_os_error() == Some(26) && attempt + 1 < ETXTBSY_RETRY_ATTEMPTS => {
                last_error = Some(err);
                thread::sleep(ETXTBSY_RETRY_DELAY);
            },
            Err(err) => return Err(err),
        }
    }

    Err(last_error.expect("ETXTBSY retry loop should retain the last error"))
}

fn parse_compiler_include_dir(stderr: &str) -> Option<PathBuf> {
    let mut in_search_list = false;
    for line in stderr.lines() {
        if line.contains("#include <...> search starts here:") {
            in_search_list = true;
            continue;
        }
        if line.contains("End of search list.") {
            break;
        }
        if in_search_list {
            let path = PathBuf::from(line.trim());
            if path.join("stddef.h").exists() {
                return Some(path);
            }
        }
    }

    None
}

fn format_compiler_command(compiler: &str, args: &[String]) -> String {
    if args.is_empty() { compiler.to_string() } else { format!("{compiler} {}", args.join(" ")) }
}

fn format_command(command: &Command) -> String {
    let mut parts = vec![command.get_program().to_string_lossy().into_owned()];
    parts.extend(command.get_args().map(|arg| arg.to_string_lossy().into_owned()));
    parts.join(" ")
}

fn split_shell_words(input: &str) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars();
    let mut quote = None;

    while let Some(ch) = chars.next() {
        match quote {
            Some(quote_char) if ch == quote_char => quote = None,
            Some(_) => current.push(ch),
            None if ch == '\'' || ch == '"' => quote = Some(ch),
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            },
            None if ch == '\\' => {
                let Some(next) = chars.next() else {
                    return Err("Trailing escape in shell-style command string.".to_string());
                };
                current.push(next);
            },
            None => current.push(ch),
        }
    }

    if quote.is_some() {
        return Err("Unterminated quote in shell-style command string.".to_string());
    }

    if !current.is_empty() {
        words.push(current);
    }

    Ok(words)
}
