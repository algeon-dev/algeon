use super::{env::EnvValues, *};

/// DataFusion catalog name that hosts the mutable session workspace overlay.
/// Defaults to `datafusion` when a backing source schema is configured.
pub const NEXUS_SERVER_WORKSPACE_CATALOG_ENV: &str = "NEXUS_SERVER_WORKSPACE_CATALOG";
/// DataFusion schema name that hosts the mutable session workspace overlay.
/// Defaults to `public` when a backing source schema is configured.
pub const NEXUS_SERVER_WORKSPACE_SCHEMA_ENV: &str = "NEXUS_SERVER_WORKSPACE_SCHEMA";
/// Source catalog that the mutable workspace overlay can read through.
pub const NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_CATALOG";
/// Source schema that the mutable workspace overlay can read through.
pub const NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA";
/// Optional comma-separated source names exposed through the workspace overlay.
/// When unset, all backing schema table names are visible through the overlay.
pub const NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV: &str = "NEXUS_SERVER_WORKSPACE_BACKING_ALIASES";
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceOverlayConfig {
    pub catalog: String,
    pub schema: String,
    pub backing_catalog: String,
    pub backing_schema: String,
    /// `None` means every table reported by the backing schema is visible.
    pub backing_aliases: Option<Vec<String>>,
}
pub(super) fn workspace_overlay_config(values: &mut EnvValues<'_>) -> Result<Option<WorkspaceOverlayConfig>> {
    let requested = values.any_present([
        NEXUS_SERVER_WORKSPACE_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_SCHEMA_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
        NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV,
    ]);
    if !requested {
        return Ok(None);
    }

    let backing_catalog = values.string(NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV);
    let backing_schema = values.string(NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV);
    let (backing_catalog, backing_schema) = match (backing_catalog, backing_schema) {
        (Some(catalog), Some(schema)) => (catalog, schema),
        (None, None) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
                format!("requires {NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV} when workspace overlay is configured"),
            ));
        },
        (None, Some(_)) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV,
                format!("is required when {NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV} is set"),
            ));
        },
        (Some(_), None) => {
            return Err(ServerError::invalid_config(
                NEXUS_SERVER_WORKSPACE_BACKING_SCHEMA_ENV,
                format!("is required when {NEXUS_SERVER_WORKSPACE_BACKING_CATALOG_ENV} is set"),
            ));
        },
    };

    let catalog = values.string(NEXUS_SERVER_WORKSPACE_CATALOG_ENV).unwrap_or_else(|| "datafusion".to_string());
    let schema = values.string(NEXUS_SERVER_WORKSPACE_SCHEMA_ENV).unwrap_or_else(|| "public".to_string());
    if catalog == backing_catalog && schema == backing_schema {
        return Err(ServerError::invalid_config(
            NEXUS_SERVER_WORKSPACE_CATALOG_ENV,
            "workspace catalog/schema must differ from backing catalog/schema",
        ));
    }

    let backing_aliases = values
        .string(NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV)
        .map(|raw| parse_workspace_aliases(NEXUS_SERVER_WORKSPACE_BACKING_ALIASES_ENV, &raw))
        .transpose()?;

    Ok(Some(WorkspaceOverlayConfig { catalog, schema, backing_catalog, backing_schema, backing_aliases }))
}

pub(super) fn parse_workspace_aliases(env_name: &'static str, raw: &str) -> Result<Vec<String>> {
    let mut aliases = Vec::new();
    for alias in raw.split(',').map(str::trim).filter(|alias| !alias.is_empty()) {
        if aliases.iter().any(|existing| existing == alias) {
            return Err(ServerError::invalid_config(env_name, format!("contains duplicate alias `{alias}`")));
        }
        aliases.push(alias.to_string());
    }
    if aliases.is_empty() {
        return Err(ServerError::invalid_config(env_name, "must contain at least one alias"));
    }
    Ok(aliases)
}
