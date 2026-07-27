use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

pub(crate) fn resolve_include_dir(env_val: Option<&str>, submodule_path: &Path) -> Result<PathBuf, String> {
    if let Some(dir_str) = env_val {
        return resolve_include_override(dir_str);
    }

    resolve_submodule_include_dir(submodule_path)
}

pub(crate) fn resolve_library_dir(env_val: Option<&str>, submodule_candidates: &[PathBuf]) -> Result<PathBuf, String> {
    if let Some(dir_str) = env_val {
        return resolve_library_override(dir_str);
    }

    for candidate in submodule_candidates {
        if has_required_cugraph_libs(candidate) {
            return Ok(candidate.canonicalize().unwrap_or_else(|_| candidate.clone()));
        }
    }

    if submodule_candidates.is_empty() {
        return Err("no cuGraph library search paths were provided".to_string());
    }

    let searched_paths =
        submodule_candidates.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join(", ");

    Err(format!(
        "Could not find cuGraph libraries in fallback search paths [{searched_paths}]. Expected a directory containing both libcugraph and libcugraph_c. Build the cuGraph C API libraries first (see README) or set CUGRAPH_LIBRARY_DIR."
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
    let output = command.output().map_err(|err| {
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

fn resolve_submodule_include_dir(submodule_path: &Path) -> Result<PathBuf, String> {
    let canonical = submodule_path.canonicalize().map_err(|e| {
        format!(
            "Could not resolve cuGraph include directory '{}': {e}. Initialize the git submodule (git submodule update --init --recursive) or set CUGRAPH_INCLUDE_DIR.",
            submodule_path.display()
        )
    })?;

    if !canonical.join("cugraph_c").is_dir() {
        return Err(format!(
            "cuGraph include directory '{}' does not contain a cugraph_c/ subdirectory. The git submodule may be incomplete; try: git submodule update --init --recursive",
            canonical.display()
        ));
    }

    Ok(canonical)
}

fn resolve_library_override(dir_str: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(dir_str);
    let canonical = path.canonicalize().map_err(|_| {
        format!(
            "CUGRAPH_LIBRARY_DIR='{}' does not exist or is not accessible. Expected a directory containing libcugraph and libcugraph_c.",
            path.display()
        )
    })?;

    if !has_required_cugraph_libs(&canonical) {
        return Err(format!(
            "CUGRAPH_LIBRARY_DIR='{}' is invalid. Expected '{}' to contain both libcugraph and libcugraph_c.",
            path.display(),
            canonical.display()
        ));
    }

    Ok(canonical)
}

fn has_library(dir: &Path, stem: &str) -> bool {
    dir.join(format!("lib{stem}.so")).exists()
        || dir.join(format!("lib{stem}.so.0")).exists()
        || dir.join(format!("lib{stem}.so.1")).exists()
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
