use std::sync::Arc;

use datafusion::{
    catalog::{MemoryCatalogProvider, MemorySchemaProvider},
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use super::{config::ServerConfig, error::ServerError};
use crate::CudfSessionStateBuilderExt;

pub async fn build_session_context(config: &ServerConfig) -> super::error::Result<SessionContext> {
    let session_config = SessionConfig::from_env()
        .map_err(|error| ServerError::session("failed to build DataFusion session config from environment", error))?
        .with_information_schema(true);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native(config.native_optimizer_config.clone())
        .build();
    let ctx = SessionContext::new_with_state(state);

    #[cfg(feature = "iceberg")]
    register_iceberg_tables(&ctx, config).await?;

    Ok(ctx)
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_tables(ctx: &SessionContext, config: &ServerConfig) -> super::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    // Ensure the named catalog and namespace/schema exist in DataFusion's
    // catalog list before registering any tables.  This is required so that
    // three-part addressing (catalog.namespace.table) resolves correctly.
    ensure_catalog_namespace(ctx, &iceberg.catalog_name, &iceberg.namespace)
        .map_err(|error| ServerError::session("failed to create Iceberg catalog/namespace in DataFusion", error))?;

    for table in &iceberg.tables {
        // glue_catalog_handle is the internal iceberg-rust catalog identifier.
        // It is distinct from iceberg.catalog_name (the DataFusion connector
        // name) so users can configure `DATAFUSION_CUDF_ICEBERG_CATALOG_NAME=iceberg`
        // for Trino-style SQL without changing the Glue catalog handle.
        let mut glue = crate::iceberg::GlueIcebergTableConfig::new(
            iceberg.glue_catalog_handle.clone(),
            iceberg.namespace.clone(),
            table.table.clone(),
            iceberg.warehouse.clone(),
            iceberg.region.clone(),
        );
        if let Some(catalog_id) = &iceberg.catalog_id {
            glue = glue.with_catalog_id(catalog_id.clone());
        }
        if let Some(profile) = &iceberg.profile {
            glue = glue.with_profile(profile.clone());
        }

        let loaded = crate::iceberg::load_glue_table(glue)
            .await
            .map_err(|error| ServerError::session("failed to load Glue Iceberg table", error))?;
        let diagnostics = loaded.diagnostics.clone();
        let provider: Arc<dyn datafusion::datasource::TableProvider> = Arc::new(
            crate::iceberg::table_provider(loaded.table)
                .map_err(|error| ServerError::session("failed to build CudfIcebergTableProvider", error))?
                .with_catalog_type("glue"),
        );

        // Three-part registration: catalog.namespace.real_table_name
        register_full(ctx, &iceberg.catalog_name, &iceberg.namespace, &table.table, Arc::clone(&provider))
            .map_err(|error| ServerError::session("failed to register Iceberg table under three-part name", error))?;

        // catalog.namespace.alias — only when alias differs from real table name
        if table.alias != table.table {
            register_full(ctx, &iceberg.catalog_name, &iceberg.namespace, &table.alias, Arc::clone(&provider))
                .map_err(|error| {
                    ServerError::session("failed to register Iceberg table under three-part alias", error)
                })?;
        }

        // Legacy bare alias for backward compatibility.
        if let Some(replaced) =
            ctx.register_table(table.alias.as_str(), Arc::clone(&provider))
                .map_err(|error| ServerError::session("failed to register Iceberg table as bare alias", error))?
        {
            let _ = replaced;
            tracing::warn!(
                catalog_name = iceberg.catalog_name,
                namespace = iceberg.namespace,
                table = table.table,
                alias = table.alias,
                "bare alias registration replaced an existing provider; check for duplicate alias entries"
            );
        }

        tracing::info!(
            catalog_type = "glue",
            df_catalog_name = iceberg.catalog_name,
            glue_catalog_handle = diagnostics.catalog_name,
            namespace = iceberg.namespace,
            table = diagnostics.table,
            alias = table.alias,
            region = diagnostics.region,
            credential_source = %diagnostics.credential_source,
            "registered Iceberg table"
        );
    }

    Ok(())
}

#[cfg(feature = "iceberg")]
fn ensure_catalog_namespace(
    ctx: &SessionContext,
    catalog_name: &str,
    namespace: &str,
) -> datafusion::common::Result<()> {
    if ctx.catalog(catalog_name).is_none() {
        let catalog = Arc::new(MemoryCatalogProvider::new());
        ctx.register_catalog(catalog_name, catalog);
    }
    let catalog = ctx.catalog(catalog_name).expect("catalog was just registered");
    if catalog.schema(namespace).is_none() {
        let schema = Arc::new(MemorySchemaProvider::new());
        catalog.register_schema(namespace, schema)?;
    }
    Ok(())
}

#[cfg(feature = "iceberg")]
fn register_full(
    ctx: &SessionContext,
    catalog: &str,
    namespace: &str,
    table: &str,
    provider: Arc<dyn datafusion::datasource::TableProvider>,
) -> datafusion::common::Result<()> {
    use datafusion::common::TableReference;
    if let Some(replaced) = ctx.register_table(TableReference::full(catalog, namespace, table), provider)? {
        let _ = replaced;
        tracing::warn!(
            %catalog,
            %namespace,
            %table,
            "three-part table registration replaced an existing provider; check for duplicate table/alias entries"
        );
    }
    Ok(())
}
