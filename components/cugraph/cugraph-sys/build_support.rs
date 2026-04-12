use std::{
    env, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

const ETXTBSY_RETRY_ATTEMPTS: usize = 6;
const ETXTBSY_RETRY_DELAY: Duration = Duration::from_millis(25);

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_include_dir(env_val: Option<&str>, submodule_path: &Path) -> Result<PathBuf, String> {
    let paths = resolve_include_paths(env_val, submodule_path)?;
    paths.into_iter().next().ok_or_else(|| {
        "pkg-config located cuGraph but did not report an explicit include directory. Set CUGRAPH_INCLUDE_DIR to the directory containing cugraph_c/ if bindgen cannot find the headers."
            .to_string()
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_library_dir(env_val: Option<&str>, submodule_candidates: &[PathBuf]) -> Result<PathBuf, String> {
    let dirs = resolve_library_dirs(env_val, submodule_candidates)?;
    dirs.into_iter().next().ok_or_else(|| {
        "pkg-config located cuGraph but did not report an explicit library directory. Set CUGRAPH_LIBRARY_DIR if the linker cannot locate libcugraph and libcugraph_c."
            .to_string()
    })
}

pub(crate) fn resolve_include_paths(env_val: Option<&str>, submodule_path: &Path) -> Result<Vec<PathBuf>, String> {
    let pkg_config = env::var("PKG_CONFIG").ok();
    resolve_include_paths_with_pkg_config(env_val, pkg_config.as_deref(), submodule_path)
}

pub(crate) fn resolve_library_dirs(
    env_val: Option<&str>,
    submodule_candidates: &[PathBuf],
) -> Result<Vec<PathBuf>, String> {
    let pkg_config = env::var("PKG_CONFIG").ok();
    resolve_library_dirs_with_pkg_config(env_val, pkg_config.as_deref(), submodule_candidates)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_include_dir_with_pkg_config(
    env_val: Option<&str>,
    pkg_config_env: Option<&str>,
    submodule_path: &Path,
) -> Result<PathBuf, String> {
    let paths = resolve_include_paths_with_pkg_config(env_val, pkg_config_env, submodule_path)?;
    paths.into_iter().next().ok_or_else(|| {
        "pkg-config located cuGraph but did not report an explicit include directory. Set CUGRAPH_INCLUDE_DIR to the directory containing cugraph_c/ if bindgen cannot find the headers."
            .to_string()
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn resolve_library_dir_with_pkg_config(
    env_val: Option<&str>,
    pkg_config_env: Option<&str>,
    submodule_candidates: &[PathBuf],
) -> Result<PathBuf, String> {
    let dirs = resolve_library_dirs_with_pkg_config(env_val, pkg_config_env, submodule_candidates)?;
    dirs.into_iter().next().ok_or_else(|| {
        "pkg-config located cuGraph but did not report an explicit library directory. Set CUGRAPH_LIBRARY_DIR if the linker cannot locate libcugraph and libcugraph_c."
            .to_string()
    })
}

pub(crate) fn resolve_include_paths_with_pkg_config(
    env_val: Option<&str>,
    pkg_config_env: Option<&str>,
    submodule_path: &Path,
) -> Result<Vec<PathBuf>, String> {
    if let Some(dir_str) = env_val {
        return resolve_include_override(dir_str).map(|dir| vec![dir]);
    }

    if let Some(pkg_config) = probe_pkg_config_paths(pkg_config_env)? {
        return Ok(pkg_config.include_paths);
    }

    let fallback_candidates = include_search_candidates(submodule_path);
    for candidate in &fallback_candidates {
        if let Some(dir) = include_candidate_dir(candidate) {
            return Ok(vec![dir]);
        }
    }

    let searched_paths =
        fallback_candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ");

    Err(format!(
        "Could not find cuGraph headers in fallback search paths [{searched_paths}]. Expected a directory containing cugraph_c/. Activate the target conda environment, initialize the git submodule (git submodule update --init --recursive), or set CUGRAPH_INCLUDE_DIR."
    ))
}

pub(crate) fn resolve_library_dirs_with_pkg_config(
    env_val: Option<&str>,
    pkg_config_env: Option<&str>,
    submodule_candidates: &[PathBuf],
) -> Result<Vec<PathBuf>, String> {
    if let Some(dir_str) = env_val {
        return resolve_library_override(dir_str).map(|dir| vec![dir]);
    }

    if let Some(pkg_config) = probe_pkg_config_paths(pkg_config_env)? {
        return Ok(pkg_config.link_paths);
    }

    let fallback_candidates = library_search_candidates(submodule_candidates);
    for candidate in &fallback_candidates {
        if has_required_cugraph_libs(candidate) {
            return Ok(vec![candidate.canonicalize().unwrap_or_else(|_| candidate.clone())]);
        }
    }

    if fallback_candidates.is_empty() {
        return Err("no cuGraph library search paths were provided".to_string());
    }

    let searched_paths =
        fallback_candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ");

    Err(format!(
        "Could not find cuGraph libraries in fallback search paths [{searched_paths}]. Expected a directory containing both libcugraph and libcugraph_c shared libraries (for example libcugraph.so, libcugraph.dylib, or cugraph.lib). Build the cuGraph C API libraries first, activate the target conda environment, or set CUGRAPH_LIBRARY_DIR."
    ))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PkgConfigPaths {
    pub(crate) include_paths: Vec<PathBuf>,
    pub(crate) link_paths: Vec<PathBuf>,
}

pub(crate) fn library_search_candidates(submodule_candidates: &[PathBuf]) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for env_var in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        if let Some(prefix) = env::var_os(env_var) {
            extend_unique_paths(&mut candidates, conda_prefix_library_candidates(Path::new(&prefix)));
        }
    }

    extend_unique_paths(&mut candidates, submodule_candidates.iter().cloned());
    candidates
}

pub(crate) fn include_search_candidates(submodule_path: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for env_var in ["CONDA_PREFIX", "LIBRARY_PREFIX", "PREFIX"] {
        if let Some(prefix) = env::var_os(env_var) {
            extend_unique_paths(&mut candidates, conda_prefix_include_candidates(Path::new(&prefix)));
        }
    }

    extend_unique_paths(&mut candidates, [submodule_path.to_path_buf()]);
    candidates
}

pub(crate) fn conda_prefix_library_candidates(prefix: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    extend_unique_paths(
        &mut candidates,
        [prefix.to_path_buf(), prefix.join("lib"), prefix.join("Library").join("lib")],
    );
    candidates
}

pub(crate) fn conda_prefix_include_candidates(prefix: &Path) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    extend_unique_paths(&mut candidates, [prefix.join("include"), prefix.join("Library").join("include")]);
    candidates
}

fn resolve_include_override(dir_str: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(dir_str);
    let canonical = path.canonicalize().map_err(|_| {
        format!(
            "CUGRAPH_INCLUDE_DIR='{}' does not exist or is not accessible. Expected a directory containing cugraph_c/.",
            path.display()
        )
    })?;

    if !canonical.join("cugraph_c").is_dir() {
        return Err(format!(
            "CUGRAPH_INCLUDE_DIR='{}' is invalid. Expected '{}/cugraph_c' to exist.",
            path.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

fn resolve_library_override(dir_str: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(dir_str);
    let canonical = path.canonicalize().map_err(|_| {
        format!(
            "CUGRAPH_LIBRARY_DIR='{}' does not exist or is not accessible. Expected a directory containing libcugraph and libcugraph_c shared libraries.",
            path.display()
        )
    })?;

    if !has_required_cugraph_libs(&canonical) {
        return Err(format!(
            "CUGRAPH_LIBRARY_DIR='{}' is invalid. Expected '{}' to contain both libcugraph and libcugraph_c shared libraries (for example libcugraph.so, libcugraph.dylib, or cugraph.lib).",
            path.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
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

fn extend_unique_paths<I>(paths: &mut Vec<PathBuf>, candidates: I)
where I: IntoIterator<Item = PathBuf> {
    for candidate in candidates {
        if !paths.iter().any(|existing| existing == &candidate) {
            paths.push(candidate);
        }
    }
}

fn include_candidate_dir(candidate: &Path) -> Option<PathBuf> {
    if candidate.join("cugraph_c").is_dir() {
        Some(candidate.canonicalize().unwrap_or_else(|_| candidate.to_path_buf()))
    } else {
        None
    }
}

pub(crate) fn resolve_pkg_config_command(env_val: Option<&str>) -> Result<(String, Vec<String>), String> {
    let raw = env_val.unwrap_or("pkg-config").trim();
    if raw.is_empty() {
        return Err(
            "PKG_CONFIG is set but empty. Set PKG_CONFIG to the pkg-config binary used for this build, or unset PKG_CONFIG to fall back to 'pkg-config'."
                .to_string(),
        );
    }

    let parts = split_shell_words(raw)?;
    let (tool, args) = parts.split_first().ok_or_else(|| {
        "Could not determine a pkg-config command from PKG_CONFIG. Set PKG_CONFIG to the pkg-config binary used for this build, or unset PKG_CONFIG to fall back to 'pkg-config'."
            .to_string()
    })?;

    Ok((tool.clone(), args.to_vec()))
}

pub(crate) fn probe_pkg_config_paths(pkg_config_env: Option<&str>) -> Result<Option<PkgConfigPaths>, String> {
    let pkg_config_override = pkg_config_env.map(str::trim).filter(|raw| !raw.is_empty());
    let (tool, args) = resolve_pkg_config_command(pkg_config_override)?;
    let command_display = format_compiler_command(&tool, &args);
    let mut include_paths = Vec::new();
    let mut link_paths = Vec::new();
    let mut found_any = false;

    for package in ["cugraph_c", "cugraph"] {
        let output = match run_pkg_config(&tool, &args, package) {
            Ok(output) => output,
            Err(err) => {
                if pkg_config_override.is_some() {
                    return Err(format!(
                        "Failed to run pkg-config command '{}' while probing {}: {err}. Fix PKG_CONFIG or unset it to skip pkg-config discovery.",
                        command_display, package
                    ));
                }
                return Ok(None);
            },
        };

        if !output.status.success() {
            continue;
        }

        found_any = true;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for token in split_shell_words(stdout.as_ref())? {
            if let Some(path) = token.strip_prefix("-I") {
                extend_unique_paths(&mut include_paths, [normalize_path(Path::new(path))]);
            } else if let Some(path) = token.strip_prefix("-L") {
                extend_unique_paths(&mut link_paths, [normalize_path(Path::new(path))]);
            }
        }
    }

    if found_any { Ok(Some(PkgConfigPaths { include_paths, link_paths })) } else { Ok(None) }
}

fn run_pkg_config(tool: &str, tool_args: &[String], package: &str) -> std::io::Result<std::process::Output> {
    let mut command = Command::new(tool);
    command
        .args(tool_args)
        .args(["--cflags-only-I", "--libs-only-L", package])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    command_output_with_etxtbsy_retry(&mut command)
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

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
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
            Some('\'') => {
                if ch == '\'' {
                    quote = None;
                } else {
                    current.push(ch);
                }
            },
            Some('"') => {
                if ch == '"' {
                    quote = None;
                } else if ch == '\\' {
                    let escaped = chars
                        .next()
                        .ok_or_else(|| format!("Could not parse CC='{}': trailing escape sequence.", input))?;
                    current.push(escaped);
                } else {
                    current.push(ch);
                }
            },
            Some(_) => unreachable!("unsupported quote state"),
            None => match ch {
                '\'' | '"' => quote = Some(ch),
                '\\' => {
                    let escaped = chars
                        .next()
                        .ok_or_else(|| format!("Could not parse CC='{}': trailing escape sequence.", input))?;
                    current.push(escaped);
                },
                c if c.is_whitespace() => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                },
                _ => current.push(ch),
            },
        }
    }

    if quote.is_some() {
        return Err(format!("Could not parse CC='{}': unterminated quote.", input));
    }

    if !current.is_empty() {
        words.push(current);
    }

    Ok(words)
}
