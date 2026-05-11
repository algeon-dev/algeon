use std::{collections::BTreeSet, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    catalog::{
        CatalogProvider, SchemaProvider,
        memory::{MemoryCatalogProvider, MemorySchemaProvider},
    },
    common::Result as DataFusionResult,
    datasource::{TableProvider, TableType},
    prelude::SessionContext,
};

use crate::{
    config::WorkspaceOverlayConfig,
    error::{Result, ServerError},
};

/// Mutable local workspace schema with optional read-through to one source schema.
///
/// DDL only affects `local`; backing tables remain read-only and are visible
/// only through `table`, `table_type`, `table_exist`, and `table_names`.
#[derive(Debug)]
pub(crate) struct NexusWorkspaceSchemaProvider {
    local: Arc<MemorySchemaProvider>,
    backing: Arc<dyn SchemaProvider>,
    backing_aliases: Option<BTreeSet<String>>,
}

impl NexusWorkspaceSchemaProvider {
    pub(crate) fn new(
        backing: Arc<dyn SchemaProvider>,
        backing_aliases: Option<impl IntoIterator<Item = String>>,
    ) -> Self {
        Self {
            local: Arc::new(MemorySchemaProvider::new()),
            backing,
            backing_aliases: backing_aliases.map(|aliases| aliases.into_iter().collect()),
        }
    }

    fn backing_name_visible(&self, name: &str) -> bool {
        self.backing_aliases.as_ref().is_none_or(|aliases| aliases.contains(name))
    }

    fn backing_table_names(&self) -> Vec<String> {
        let names = match &self.backing_aliases {
            Some(aliases) => aliases.iter().cloned().collect::<Vec<_>>(),
            None => self.backing.table_names(),
        };
        if self.backing_aliases.is_none() {
            #[cfg(feature = "iceberg")]
            {
                let mut names = names;
                if let Some(iceberg) =
                    self.backing.downcast_ref::<crate::table_format::iceberg::catalog::NexusIcebergSchemaProvider>()
                {
                    names.extend(iceberg.known_or_seeded_table_names());
                }
                return names;
            }
        }
        names
    }

    fn visible_backing_table_names(&self) -> impl Iterator<Item = String> + '_ {
        self.backing_table_names().into_iter().filter(|name| self.backing_name_visible(name))
    }

    fn backing_table_exists(&self, name: &str) -> bool {
        if !self.backing_name_visible(name) {
            return false;
        }
        if self.backing_aliases.as_ref().is_some_and(|aliases| aliases.contains(name)) {
            return true;
        }
        if self.backing.table_exist(name) || self.backing_table_names().iter().any(|table_name| table_name == name) {
            return true;
        }
        #[cfg(feature = "iceberg")]
        if let Some(iceberg) =
            self.backing.downcast_ref::<crate::table_format::iceberg::catalog::NexusIcebergSchemaProvider>()
        {
            return iceberg.known_or_seeded_table_exists(name);
        }
        false
    }
}

#[async_trait]
impl SchemaProvider for NexusWorkspaceSchemaProvider {
    fn table_names(&self) -> Vec<String> {
        let mut names = self.visible_backing_table_names().collect::<BTreeSet<_>>();
        names.extend(self.local.table_names());
        names.into_iter().collect()
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        if let Some(table) = self.local.table(name).await? {
            return Ok(Some(table));
        }
        if !self.backing_name_visible(name) {
            return Ok(None);
        }
        self.backing.table(name).await
    }

    async fn table_type(&self, name: &str) -> DataFusionResult<Option<TableType>> {
        if self.local.table_exist(name) {
            return self.local.table_type(name).await;
        }
        if !self.backing_name_visible(name) {
            return Ok(None);
        }
        self.backing.table_type(name).await
    }

    fn register_table(
        &self,
        name: String,
        table: Arc<dyn TableProvider>,
    ) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        if self.backing_table_exists(&name) {
            return Err(crate::adapter_error::Error::new(
                crate::adapter_error::ErrorCode::InvalidRequest,
                "datafusion_nexus.server.workspace.shadow_guard",
                format!(
                    "workspace table `{name}` would shadow a backing source table; choose a different workspace name"
                ),
            )
            .into());
        }
        self.local.register_table(name, table)
    }

    fn deregister_table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        self.local.deregister_table(name)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.local.table_exist(name) || self.backing_table_exists(name)
    }
}

pub(crate) fn install_workspace_overlay(ctx: &SessionContext, config: &WorkspaceOverlayConfig) -> Result<()> {
    let backing_catalog = ctx.catalog(&config.backing_catalog).ok_or_else(|| {
        ServerError::configuration(format!(
            "workspace overlay backing catalog `{}` is not registered",
            config.backing_catalog
        ))
    })?;
    let backing_schema = backing_catalog.schema(&config.backing_schema).ok_or_else(|| {
        ServerError::configuration(format!(
            "workspace overlay backing schema `{}.{}` is not registered",
            config.backing_catalog, config.backing_schema
        ))
    })?;

    let workspace_schema = Arc::new(NexusWorkspaceSchemaProvider::new(backing_schema, config.backing_aliases.clone()));
    let workspace_schema: Arc<dyn SchemaProvider> = workspace_schema;

    match ctx.catalog(&config.catalog) {
        Some(catalog) => {
            catalog
                .register_schema(&config.schema, workspace_schema)
                .map_err(|error| ServerError::session("failed to register workspace overlay schema", error))?;
        },
        None => {
            let catalog = MemoryCatalogProvider::new();
            catalog
                .register_schema(&config.schema, workspace_schema)
                .map_err(|error| ServerError::session("failed to register workspace overlay schema", error))?;
            ctx.register_catalog(&config.catalog, Arc::new(catalog));
        },
    }

    tracing::info!(
        workspace_catalog = config.catalog,
        workspace_schema = config.schema,
        backing_catalog = config.backing_catalog,
        backing_schema = config.backing_schema,
        backing_aliases =
            config.backing_aliases.as_ref().map(|aliases| aliases.join(",")).unwrap_or_else(|| "all".to_string()),
        "registered mutable workspace overlay"
    );
    Ok(())
}

pub(crate) fn warn_if_default_schema_is_read_only(ctx: &SessionContext, config: &crate::config::ServerConfig) {
    let state = ctx.state();
    let options = state.config().options();
    let default_catalog = options.catalog.default_catalog.as_str();
    let default_schema = options.catalog.default_schema.as_str();
    if config
        .workspace
        .as_ref()
        .is_some_and(|workspace| workspace.catalog == default_catalog && workspace.schema == default_schema)
    {
        return;
    }
    let Some(catalog) = ctx.catalog(default_catalog) else {
        return;
    };
    let Some(schema) = catalog.schema(default_schema) else {
        return;
    };
    if schema_supports_session_ddl(schema.as_ref()) {
        return;
    }
    tracing::warn!(
        default_catalog,
        default_schema,
        iceberg_enabled = iceberg_enabled(config),
        workspace_catalog = config.workspace.as_ref().map(|workspace| workspace.catalog.as_str()).unwrap_or("none"),
        workspace_schema = config.workspace.as_ref().map(|workspace| workspace.schema.as_str()).unwrap_or("none"),
        "default DataFusion catalog/schema is read-only for session DDL; use datafusion.public or enable the Nexus workspace overlay"
    );
}

#[cfg(feature = "iceberg")]
fn iceberg_enabled(config: &crate::config::ServerConfig) -> bool {
    config.iceberg.is_some()
}

#[cfg(not(feature = "iceberg"))]
fn iceberg_enabled(_config: &crate::config::ServerConfig) -> bool {
    false
}

fn schema_supports_session_ddl(schema: &dyn SchemaProvider) -> bool {
    if schema.is::<MemorySchemaProvider>() || schema.is::<NexusWorkspaceSchemaProvider>() {
        return true;
    }
    #[cfg(feature = "iceberg")]
    if schema.is::<crate::table_format::iceberg::catalog::NexusIcebergSchemaProvider>() {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::{
        catalog::{CatalogProvider, SchemaProvider, memory::MemoryCatalogProvider},
        datasource::{MemTable, TableProvider, TableType},
    };

    use super::*;

    fn table() -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        Arc::new(MemTable::try_new(schema, vec![vec![]]).unwrap())
    }

    #[tokio::test]
    async fn workspace_reads_backing_and_keeps_writes_local() {
        let backing = Arc::new(MemorySchemaProvider::new());
        backing.register_table("edges_by_src".to_string(), table()).unwrap();
        let workspace = NexusWorkspaceSchemaProvider::new(backing.clone(), Some(["edges_by_src".to_string()]));

        assert!(workspace.table("edges_by_src").await.unwrap().is_some());
        assert_eq!(workspace.table_type("edges_by_src").await.unwrap(), Some(TableType::Base));
        workspace.register_table("target_edges".to_string(), table()).unwrap();
        assert!(workspace.table("target_edges").await.unwrap().is_some());
        assert!(!backing.table_exist("target_edges"));
        assert!(workspace.deregister_table("target_edges").unwrap().is_some());
        assert!(workspace.table("target_edges").await.unwrap().is_none());
    }

    #[test]
    fn workspace_rejects_accidental_source_shadowing() {
        let backing = Arc::new(MemorySchemaProvider::new());
        backing.register_table("edges_by_src".to_string(), table()).unwrap();
        let workspace = NexusWorkspaceSchemaProvider::new(backing, Option::<Vec<String>>::None);

        let error = workspace.register_table("edges_by_src".to_string(), table()).unwrap_err();

        assert!(error.to_string().contains("would shadow a backing source table"));
    }

    #[test]
    fn workspace_catalog_can_be_installed_into_memory_catalog() {
        let ctx = SessionContext::new();
        let backing_catalog = MemoryCatalogProvider::new();
        let backing_schema = Arc::new(MemorySchemaProvider::new());
        backing_schema.register_table("edges_by_src".to_string(), table()).unwrap();
        backing_catalog.register_schema("nexus_graph", backing_schema).unwrap();
        ctx.register_catalog("glue", Arc::new(backing_catalog));

        install_workspace_overlay(
            &ctx,
            &WorkspaceOverlayConfig {
                catalog: "datafusion".to_string(),
                schema: "public".to_string(),
                backing_catalog: "glue".to_string(),
                backing_schema: "nexus_graph".to_string(),
                backing_aliases: Some(vec!["edges_by_src".to_string()]),
            },
        )
        .unwrap();

        let workspace = ctx.catalog("datafusion").unwrap().schema("public").unwrap();
        assert!(workspace.table_exist("edges_by_src"));
    }
}
