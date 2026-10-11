use super::{file::ServerModelsFileConfig, *};

/// Validates the `[models]` table. The root must exist and be a directory at
/// startup, so a misconfigured root fails the server before it accepts
/// connections.
pub(super) fn models_config(
    file: &ServerModelsFileConfig,
    sources: &ConfigSources,
) -> Result<Option<CumlModelStoreConfig>> {
    let Some(root) = &file.root else {
        if file.max_model_bytes.is_some() {
            return Err(sources.invalid("models.max_model_bytes", "requires models.root"));
        }
        return Ok(None);
    };
    let max_model_bytes = file.max_model_bytes.unwrap_or(CumlModelStoreConfig::DEFAULT_MAX_MODEL_BYTES);
    if max_model_bytes == 0 {
        return Err(sources.invalid("models.max_model_bytes", "must be greater than zero"));
    }
    let root = root.canonicalize().map_err(|error| {
        sources.invalid_with_source("models.root", format!("`{}` cannot be resolved", root.display()), error)
    })?;
    if !root.is_dir() {
        return Err(sources.invalid("models.root", format!("`{}` is not a directory", root.display())));
    }
    Ok(Some(CumlModelStoreConfig { root, max_model_bytes }))
}
