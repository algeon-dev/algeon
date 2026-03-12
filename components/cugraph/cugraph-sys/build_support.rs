use std::path::{Path, PathBuf};

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

    let fallback = submodule_candidates
        .first()
        .cloned()
        .ok_or_else(|| "no cuGraph library search paths were provided".to_string())?;

    match fallback.canonicalize() {
        Ok(path) => Ok(path),
        Err(_) => Ok(fallback),
    }
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
    #[cfg(target_os = "macos")]
    {
        return dir.join(format!("lib{stem}.dylib")).exists();
    }

    #[cfg(target_os = "windows")]
    {
        return dir.join(format!("{stem}.dll")).exists()
            || dir.join(format!("lib{stem}.dll")).exists()
            || dir.join(format!("{stem}.lib")).exists()
            || dir.join(format!("lib{stem}.lib")).exists();
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        dir.join(format!("lib{stem}.so")).exists()
            || dir.join(format!("lib{stem}.so.0")).exists()
            || dir.join(format!("lib{stem}.so.1")).exists()
    }
}
