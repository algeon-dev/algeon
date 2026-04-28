#[cfg(feature = "iceberg")]
use std::sync::Arc;

use datafusion::{
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use super::{config::ServerConfig, error::ServerError};
use crate::{backend::NexusGpuBackend, cugraph_sql::CugraphSqlConfig, session::NexusSessionStateBuilderExt};

pub async fn build_session_context(_config: &ServerConfig) -> super::error::Result<SessionContext> {
    Err(ServerError::builder_validation(
        "server_backend_required",
        "building a Flight SQL session requires an owned NexusGpuBackend",
    ))
}

pub async fn build_session_context_with_backend(
    config: &ServerConfig,
    backend: &NexusGpuBackend,
) -> super::error::Result<SessionContext> {
    let session_config = build_session_config(config)?;
    let builder = SessionStateBuilder::new_with_default_features().with_config(session_config);
    let builder = match &config.cugraph {
        Some(cugraph) => backend
            .install_on_with_cugraph_sql(
                builder,
                cugraph_sql_config(cugraph, config.native_optimizer_config.native_execution_config().clone()),
            )
            .map_err(|error| ServerError::session("failed to install backend native session", error))?,
        None => backend
            .install_on(builder)
            .map_err(|error| ServerError::session("failed to install backend native session", error))?,
    };
    let builder = match &config.host_execution_qos {
        Some(host_execution_qos) => builder.with_host_execution_qos(host_execution_qos.clone()),
        None => builder,
    };
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);

    #[cfg(feature = "iceberg")]
    register_iceberg_catalog(&ctx, config).await?;
    if let Some(workspace) = &config.workspace {
        super::workspace::install_workspace_overlay(&ctx, workspace)?;
    }
    super::workspace::warn_if_default_schema_is_read_only(&ctx, config);

    Ok(ctx)
}

pub(crate) fn cugraph_sql_config(
    config: &super::config::CugraphServerConfig,
    native_execution_config: crate::native_exec::NexusNativeExecutionConfig,
) -> CugraphSqlConfig {
    let sql = CugraphSqlConfig { default_graph_options: config.default_graph_options.clone(), ..Default::default() };
    sql.with_native_execution_config(native_execution_config)
}

/// Build the `SessionConfig` for the Flight SQL server from DataFusion's
/// standard environment configuration. The server enables catalog discovery
/// but does not change optimizer or execution settings.
pub fn build_session_config(_config: &ServerConfig) -> super::error::Result<SessionConfig> {
    SessionConfig::from_env()
        .map(enable_session_discovery)
        .map_err(|error| ServerError::session("failed to build DataFusion session config from environment", error))
}

pub(super) fn enable_session_discovery(config: SessionConfig) -> SessionConfig {
    config.with_information_schema(true)
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_catalog(ctx: &SessionContext, config: &ServerConfig) -> super::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    // Connect the leaf catalog once. It owns catalog/FileIO credential
    // application and returns a `ResolvedIcebergTable` that freezes the
    // table, snapshot, schema, and source policy together.
    let (backend, catalog_type): (Arc<nexus_iceberg::IcebergCatalogBackend>, &'static str) = match &iceberg.backend {
        super::config::IcebergCatalogBackendConfig::Glue(glue) => {
            let mut catalog_config = nexus_iceberg::GlueIcebergCatalogConfig::new(
                glue.catalog_handle.clone(),
                glue.warehouse.clone(),
                glue.region.clone(),
            )
            .with_source_access_policy(iceberg.source_access_policy().clone());
            if let Some(id) = &glue.catalog_id {
                catalog_config = catalog_config.with_catalog_id(id.clone());
            }
            if let Some(profile) = &glue.profile {
                catalog_config = catalog_config.with_profile(profile.clone());
            }
            if let Some(uri) = &glue.uri {
                catalog_config = catalog_config.with_uri(uri.clone());
            }
            let backend = nexus_iceberg::GlueIcebergCatalogBackend::connect(catalog_config)
                .await
                .map_err(|error| ServerError::session("failed to connect to Glue Iceberg catalog backend", error))?
                .into_backend();
            (Arc::new(backend), "glue")
        },
        super::config::IcebergCatalogBackendConfig::Rest(rest) => {
            let catalog_config = rest.clone().with_source_access_policy(iceberg.source_access_policy().clone());
            let backend = nexus_iceberg::RestIcebergCatalogBackend::connect(catalog_config)
                .await
                .map_err(|error| ServerError::session("failed to connect to REST Iceberg catalog backend", error))?
                .into_backend();
            (Arc::new(backend), "rest")
        },
    };

    // The root catalog wrapper only adapts DataFusion identities. It delegates
    // every table load to the leaf backend and retains the validated server
    // read strategy; it never reconstructs storage credentials.
    let catalog_backend: Arc<dyn crate::table_format::iceberg::catalog::IcebergCatalogBackend> = Arc::new(
        crate::table_format::iceberg::catalog::NexusIcebergCatalogSource::from_shared_backend(Arc::clone(&backend))
            .with_parquet_read_strategy_config(iceberg.parquet_read_strategy_config.clone()),
    );

    // Discover only the configured namespace scope. Individual tables are
    // loaded lazily on first SQL access through the backend.
    let provider_options = crate::table_format::iceberg::catalog::IcebergProviderOptions::default()
        .with_scan_planning_timeout(Some(iceberg.scan_planning_timeout));
    let catalog_provider =
        crate::table_format::iceberg::catalog::build_scoped_catalog_provider_from_backend_with_options(
            catalog_backend,
            catalog_type,
            &iceberg.namespace,
            provider_options,
        )
        .await
        .map_err(|error| ServerError::session("failed to build Iceberg catalog provider from backend", error))?;

    // Seed pre-declared alias mappings before registering the catalog.
    // This makes catalog.namespace.alias resolve in addition to the real name.
    // Loading routes through the backend-agnostic IcebergCatalogBackend so the
    // same path serves Glue and REST.
    for table in &iceberg.tables {
        let table_ident = nexus_iceberg::IcebergTableIdentifier::new(
            iceberg.namespace.split('.').map(str::to_owned).collect(),
            table.table.clone(),
        )
        .map_err(|error| ServerError::session("invalid Iceberg table identifier for alias seeding", error))?;
        let loaded_table = backend
            .load_table(table_ident)
            .await
            .map_err(|error| ServerError::session("failed to load Iceberg table for alias seeding", error))?;
        let provider = build_alias_provider(
            loaded_table,
            catalog_type,
            iceberg.parquet_read_strategy_config.clone(),
            iceberg.scan_planning_timeout,
        )?;

        // Seed both real name and alias before catalog registration so first
        // queries reuse this load and cannot race the seed.
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
            catalog_type,
            df_catalog_name = iceberg.catalog_name,
            namespace = iceberg.namespace,
            table = table.table,
            alias = table.alias,
            "registered Iceberg table alias"
        );
    }

    // Register the dynamic catalog — from this point three-part addressing
    // (catalog_name.namespace.real_table_name) resolves through the backend.
    ctx.register_catalog(&iceberg.catalog_name, Arc::new(catalog_provider));

    tracing::info!(
        catalog_type,
        df_catalog_name = iceberg.catalog_name,
        namespace = iceberg.namespace,
        provider_mode = "dual_mode",
        pre_declared_aliases = iceberg.tables.len(),
        "registered dynamic Iceberg catalog provider"
    );

    Ok(())
}

/// Builds the [`TableProvider`] used to seed a pre-declared alias.
///
/// Uses the same dual-mode provider as the dynamic three-part catalog path.
#[cfg(feature = "iceberg")]
fn build_alias_provider(
    resolved: nexus_iceberg::ResolvedIcebergTable,
    catalog_type: &str,
    parquet_read_strategy_config: crate::table_format::scan::TableFormatParquetReadStrategyConfig,
    scan_planning_timeout: std::time::Duration,
) -> super::error::Result<Arc<dyn datafusion::datasource::TableProvider>> {
    let provider =
        crate::table_format::iceberg::NexusIcebergTableProvider::try_new_resolved_with_parquet_read_strategy_config(
            resolved,
            parquet_read_strategy_config,
        )
        .map_err(|error| ServerError::session("failed to build NexusIcebergTableProvider", error))?
        .with_catalog_type(catalog_type)
        .with_scan_planning_timeout(Some(scan_planning_timeout));
    Ok(Arc::new(provider))
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use std::collections::BTreeMap;

    use nexus_query_engine::runtime::MemoryControlPolicy;

    use super::*;
    use crate::server::{NEXUS_SERVER_CUGRAPH_ENABLED_ENV, NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV};

    #[test]
    fn cugraph_sql_config_inherits_server_native_memory_policy() {
        let env = BTreeMap::from([
            (NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string()),
            (NEXUS_SERVER_NATIVE_MEMORY_POLICY_ENV, "unbounded".to_string()),
        ]);
        let server = ServerConfig::from_env_get(|name| env.get(name).cloned()).unwrap();
        let cugraph = server.cugraph.as_ref().expect("cuGraph must be enabled");
        let sql = cugraph_sql_config(cugraph, server.native_optimizer_config.native_execution_config().clone());

        assert_eq!(sql.native_execution_config.memory_control_policy(), MemoryControlPolicy::Unbounded);
    }
}
