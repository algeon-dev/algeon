#[cfg(feature = "iceberg")]
use std::sync::Arc;

use datafusion::{
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
    register_iceberg_catalog(&ctx, config).await?;

    Ok(ctx)
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_catalog(ctx: &SessionContext, config: &ServerConfig) -> super::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    // Build the Glue catalog backend once; it holds a connected catalog handle
    // that is reused for all namespace/table listing and lazy table loading.
    let catalog_config = crate::iceberg::catalog::GlueIcebergCatalogConfig::new(
        iceberg.glue_catalog_handle.clone(),
        iceberg.warehouse.clone(),
        iceberg.region.clone(),
    );
    let catalog_config = match &iceberg.catalog_id {
        Some(id) => catalog_config.with_catalog_id(id.clone()),
        None => catalog_config,
    };
    let catalog_config = match &iceberg.profile {
        Some(p) => catalog_config.with_profile(p.clone()),
        None => catalog_config,
    };

    let backend: Arc<dyn crate::iceberg::catalog::IcebergCatalogBackend> = Arc::new(
        crate::iceberg::catalog::GlueIcebergCatalogBackend::from_config_env(catalog_config)
            .await
            .map_err(|error| ServerError::session("failed to connect to Glue Iceberg catalog backend", error))?,
    );

    // Discover only the configured namespace scope. Individual tables are
    // loaded lazily on first SQL access through the backend.
    let catalog_provider =
        crate::iceberg::catalog::build_scoped_catalog_provider_from_backend(backend, "glue", &iceberg.namespace)
            .await
            .map_err(|error| {
                ServerError::session("failed to build Iceberg catalog provider from Glue backend", error)
            })?;

    // Seed pre-declared alias mappings before registering the catalog.
    // This makes catalog.namespace.alias resolve in addition to the real name.
    for table in &iceberg.tables {
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
            .map_err(|error| ServerError::session("failed to load Glue Iceberg table for alias seeding", error))?;
        let diagnostics = loaded.diagnostics.clone();
        let provider: Arc<dyn datafusion::datasource::TableProvider> = Arc::new(
            crate::iceberg::table_provider(loaded.table)
                .map_err(|error| ServerError::session("failed to build CudfIcebergTableProvider", error))?
                .with_catalog_type("glue"),
        );

        // Seed the pre-loaded provider into the dynamic schema provider so that:
        //  - catalog.namespace.real_table_name reuses this load (skips a
        //    redundant backend round-trip on first query)
        //  - catalog.namespace.alias also resolves when alias differs from the
        //    real name
        // The seed happens before catalog registration so there is no window
        // where a concurrent query races against the seeding.
        match catalog_provider.schema_concrete(&iceberg.namespace) {
            Some(schema) => {
                schema.seed_table(table.table.clone(), Arc::clone(&provider)).await;
                if table.alias != table.table {
                    schema.seed_table(table.alias.clone(), Arc::clone(&provider)).await;
                }
            },
            None => {
                tracing::warn!(
                    df_catalog_name = iceberg.catalog_name,
                    namespace = iceberg.namespace,
                    table = table.table,
                    alias = table.alias,
                    "pre-declared Iceberg table namespace was not available in the scoped catalog provider; \
                     three-part alias addressing will not resolve (bare alias still registered)"
                );
            },
        }

        // Legacy bare alias in the default DataFusion catalog.
        if ctx
            .register_table(table.alias.as_str(), Arc::clone(&provider))
            .map_err(|error| ServerError::session("failed to register Iceberg table as bare alias", error))?
            .is_some()
        {
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
            "registered Iceberg table alias"
        );
    }

    // Register the dynamic catalog — from this point three-part addressing
    // (catalog_name.namespace.real_table_name) resolves through the backend.
    ctx.register_catalog(&iceberg.catalog_name, Arc::new(catalog_provider));

    tracing::info!(
        df_catalog_name = iceberg.catalog_name,
        glue_catalog_handle = iceberg.glue_catalog_handle,
        namespace = iceberg.namespace,
        pre_declared_aliases = iceberg.tables.len(),
        "registered dynamic Iceberg catalog provider"
    );

    Ok(())
}
