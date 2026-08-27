#[cfg(feature = "iceberg")]
use std::sync::Arc;

use datafusion::{
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use crate::{backend::NexusGpuBackend, config::ServerConfig, cugraph_sql::CugraphSqlConfig, error::ServerError};

pub async fn build_session_context(_config: &ServerConfig) -> crate::error::Result<SessionContext> {
    Err(ServerError::builder_validation(
        "server_backend_required",
        "building a Flight SQL session requires an owned NexusGpuBackend",
    ))
}

pub async fn build_session_context_with_backend(
    config: &ServerConfig,
    backend: &NexusGpuBackend,
) -> crate::error::Result<SessionContext> {
    let mut effective_config = config.clone();
    effective_config.native_optimizer_config = backend.optimizer_config().clone();
    let session_config = build_session_config(&effective_config)?;
    let builder = SessionStateBuilder::new_with_default_features().with_config(session_config);
    let builder = match &effective_config.cugraph {
        Some(cugraph) => backend
            .install_on_with_cugraph_sql(
                builder,
                cugraph_sql_config(cugraph, backend.optimizer_config().native_execution_config().clone()),
            )
            .map_err(|error| ServerError::session("failed to install backend native session", error))?,
        None => backend
            .install_on(builder)
            .map_err(|error| ServerError::session("failed to install backend native session", error))?,
    };
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);

    #[cfg(feature = "iceberg")]
    register_iceberg_catalog(&ctx, &effective_config).await?;
    if let Some(workspace) = &effective_config.workspace {
        crate::workspace::install_workspace_overlay(&ctx, workspace)?;
    }
    crate::workspace::warn_if_default_schema_is_read_only(&ctx, &effective_config);

    Ok(ctx)
}

pub(crate) fn cugraph_sql_config(
    config: &crate::config::CugraphServerConfig,
    native_execution_config: crate::native_exec::NexusNativeExecutionConfig,
) -> CugraphSqlConfig {
    let sql = CugraphSqlConfig { default_graph_options: config.default_graph_options.clone(), ..Default::default() };
    sql.with_native_execution_config(native_execution_config)
}

/// Build the `SessionConfig` for the Flight SQL server from DataFusion's
/// standard environment configuration. The server enables catalog discovery
/// but does not change optimizer or execution settings beyond what the
/// configured final plan requirement implies (see below).
pub fn build_session_config(config: &ServerConfig) -> crate::error::Result<SessionConfig> {
    let gpu_only = config
        .native_optimizer_config
        .final_plan_requirements()
        .contains(&crate::planner::FinalPlanRequirement::NoDataFusionCpu);
    SessionConfig::from_env()
        .map(enable_session_discovery)
        .map(|session_config| {
            // Under the GPU-only requirement no query can execute on the CPU,
            // so DataFusion-side parallelism is unreachable. Single-group
            // scan planning keeps sort-declared fixtures (the source-sorted
            // DBLP layout) eligible for native lowering: a declared scan
            // ordering cannot be reproduced across parallel byte-range
            // groups.
            if gpu_only { session_config.with_target_partitions(1) } else { session_config }
        })
        .map_err(|error| ServerError::session("failed to build DataFusion session config from environment", error))
}

pub(super) fn enable_session_discovery(config: SessionConfig) -> SessionConfig {
    config.with_information_schema(true)
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_catalog(ctx: &SessionContext, config: &ServerConfig) -> crate::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    // Connect the leaf catalog once. It owns catalog/FileIO credential
    // application and returns a `ResolvedIcebergTable` that freezes the
    // table, snapshot, schema, and source policy together.
    let (backend, catalog_type): (Arc<nexus_iceberg::IcebergCatalogBackend>, &'static str) = match &iceberg.backend {
        crate::config::IcebergCatalogBackendConfig::Glue(glue) => {
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
        crate::config::IcebergCatalogBackendConfig::Rest(rest) => {
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

        // Additional bare alias for unqualified references in the default DataFusion catalog.
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

/// Builds the [`datafusion::datasource::TableProvider`] used to seed a pre-declared alias.
///
/// Uses the same dual-mode provider as the dynamic three-part catalog path.
#[cfg(feature = "iceberg")]
fn build_alias_provider(
    resolved: nexus_iceberg::ResolvedIcebergTable,
    catalog_type: &str,
    parquet_read_strategy_config: crate::table_format::scan::TableFormatParquetReadStrategyConfig,
    scan_planning_timeout: std::time::Duration,
) -> crate::error::Result<Arc<dyn datafusion::datasource::TableProvider>> {
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
    use crate::NEXUS_SERVER_CUGRAPH_ENABLED_ENV;

    fn server_config_with_test_device_profile(mut env: BTreeMap<&'static str, String>) -> ServerConfig {
        let config_file = tempfile::NamedTempFile::new().expect("create server config fixture");
        std::fs::write(config_file.path(), "[[admission.device_profiles]]\ndevice_ordinal = 0\n")
            .expect("write server config fixture");
        env.insert(crate::NEXUS_SERVER_CONFIG_FILE_ENV, config_file.path().to_string_lossy().into_owned());
        ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("load server config fixture")
    }

    #[test]
    fn cugraph_sql_config_inherits_server_bounded_native_config() {
        let env = BTreeMap::from([(NEXUS_SERVER_CUGRAPH_ENABLED_ENV, "true".to_string())]);
        let server = server_config_with_test_device_profile(env);
        let cugraph = server.cugraph.as_ref().expect("cuGraph must be enabled");
        let sql = cugraph_sql_config(cugraph, server.native_optimizer_config.native_execution_config().clone());

        assert!(matches!(sql.native_execution_config.memory_control_policy(), MemoryControlPolicy::Bounded { .. }));
    }
}
