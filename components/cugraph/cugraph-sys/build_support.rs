use std::{
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

pub(crate) fn resolve_native_build_contract(
    manifest_dir: &Path,
    include_dirs_env: Option<OsString>,
    link_dirs_env: Option<OsString>,
    runtime_lib_dirs_env: Option<OsString>,
    cmake_build_dir_env: Option<OsString>,
) -> Result<NativeBuildContract, String> {
    let explicit_contract_count =
        [&include_dirs_env, &link_dirs_env, &runtime_lib_dirs_env].into_iter().filter(|value| value.is_some()).count();

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
        if path.is_absolute() && looks_like_library_path(path) {
            if let Some(parent) = path.parent() {
                push_unique_path(&mut link_dirs, normalize_existing_dir(parent)?);
            }
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
    run_cmake_probe(probe_dir)?;

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

fn run_cmake_probe(probe_dir: &Path) -> Result<(), String> {
    let configure_output = Command::new("cmake")
        .arg("-S")
        .arg(probe_dir)
        .arg("-B")
        .arg(probe_dir.join("build"))
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
        // `cugraph-rs/build.rs` links (e.g. `-lcudart`).
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
