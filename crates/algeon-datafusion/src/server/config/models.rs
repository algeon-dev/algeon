use super::{file::ServerModelsFileConfig, *};

/// Validates the `[models]` table. The root must exist and be a directory at
/// startup, so a misconfigured root fails the server before it accepts
/// connections.
pub(super) fn models_config(file: &ServerModelsFileConfig) -> Result<Option<CumlModelStoreConfig>> {
    let Some(root) = &file.root else {
        if file.max_model_bytes.is_some() {
            return Err(ServerError::invalid_config("models.max_model_bytes", "requires models.root"));
        }
        return Ok(None);
    };
    let max_model_bytes = file.max_model_bytes.unwrap_or(CumlModelStoreConfig::DEFAULT_MAX_MODEL_BYTES);
    if max_model_bytes == 0 {
        return Err(ServerError::invalid_config("models.max_model_bytes", "must be positive"));
    }
    let root = root.canonicalize().map_err(|error| {
        ServerError::invalid_config_with_source(
            "models.root",
            format!("`{}` cannot be resolved", root.display()),
            error,
        )
    })?;
    if !root.is_dir() {
        return Err(ServerError::invalid_config("models.root", format!("`{}` is not a directory", root.display())));
    }
    Ok(Some(CumlModelStoreConfig { root, max_model_bytes }))
}
