use super::{ConfigSources, PathBuf, Result, file::ServerDataFileConfig};

/// Validates the `[data]` table. The root must exist and be a directory at
/// load, so `--check-config` reports a misconfigured root before any table
/// is registered.
pub(super) fn data_root(file: &ServerDataFileConfig, sources: &ConfigSources) -> Result<Option<PathBuf>> {
    let Some(root) = &file.root else {
        return Ok(None);
    };
    let root = root.canonicalize().map_err(|error| {
        sources.invalid_with_source("data.root", format!("`{}` cannot be resolved", root.display()), error)
    })?;
    if !root.is_dir() {
        return Err(sources.invalid("data.root", format!("`{}` is not a directory", root.display())));
    }
    Ok(Some(root))
}
