#[cfg(feature = "iceberg")]
use std::sync::Arc;

use datafusion::{
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use super::{
    config::{ServerConfig, ServerSessionProfile},
    error::ServerError,
};
use crate::CudfSessionStateBuilderExt;
#[cfg(feature = "cugraph")]
use crate::CugraphSqlConfig;

pub async fn build_session_context(config: &ServerConfig) -> super::error::Result<SessionContext> {
    let session_config = build_session_config(config)?;
    let builder = SessionStateBuilder::new_with_default_features()
        .with_config(session_config)
        .with_cudf_native(config.native_optimizer_config.clone());
    #[cfg(feature = "cugraph")]
    let builder = match &config.cugraph {
        Some(cugraph) => builder.with_cugraph_sql(cugraph_sql_config(
            cugraph,
            config.native_optimizer_config.native_execution_config().clone(),
        )),
        None => builder,
    };
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);

    #[cfg(feature = "iceberg")]
    register_iceberg_catalog(&ctx, config).await?;

    Ok(ctx)
}

#[cfg(feature = "cugraph")]
fn cugraph_sql_config(
    config: &super::config::CugraphServerConfig,
    native_execution_config: crate::NativeCudfExecutionConfig,
) -> CugraphSqlConfig {
    let mut sql = match &config.allowed_algorithms {
        Some(algorithms) => CugraphSqlConfig::default().with_allowed_algorithms(algorithms.iter().copied()),
        None => CugraphSqlConfig::default(),
    };
    sql.default_graph_options = config.default_graph_options.clone();
    sql.with_native_execution_config(native_execution_config)
}

/// Build the `SessionConfig` for the Flight SQL server, honoring the
/// configured [`ServerSessionProfile`].
///
/// Starts from `DATAFUSION_*` env vars, then overrides native-sensitive
/// DataFusion settings for [`ServerSessionProfile::Native`] so server plans
/// match the workload harness shape.
pub fn build_session_config(config: &ServerConfig) -> super::error::Result<SessionConfig> {
    let base = SessionConfig::from_env()
        .map_err(|error| ServerError::session("failed to build DataFusion session config from environment", error))?;
    Ok(apply_session_profile(base, config.session_profile))
}

/// Apply [`ServerSessionProfile`] to a base `SessionConfig`. Always enables
/// `information_schema` so Flight SQL clients can discover the catalog. The
/// native profile additionally pins the native-friendly DataFusion settings;
/// the CPU-fallback profile only enables `information_schema` and leaves the
/// rest of `base` (e.g. values loaded from `DATAFUSION_*` env vars) untouched.
///
/// Embedders can call this directly when they do not want to inherit
/// `DATAFUSION_*` process env.
pub fn apply_session_profile(base: SessionConfig, profile: ServerSessionProfile) -> SessionConfig {
    let base = base.with_information_schema(true);
    match profile {
        ServerSessionProfile::Native => apply_native_friendly_profile(base),
        ServerSessionProfile::CpuFallback => base,
    }
}

/// Pin the DataFusion physical-plan profile to the native-friendly defaults
/// the workload harnesses use:
///
/// - `target_partitions=1`
/// - file-scan / aggregation / join / sort repartitioning all disabled
/// - top-k, join, aggregate, and umbrella dynamic-filter pushdown all
///   disabled (DataFusion checks each per-operator flag directly, so the
///   umbrella alone does not suppress joins or top-k pushdown)
/// - physical uncorrelated scalar subqueries disabled so scalar subquery
///   shapes stay relationalized for native planning
///
/// These overrides intentionally take precedence over `DATAFUSION_*` env
/// vars; the env vars still apply to other settings (for example
/// `batch_size`).
fn apply_native_friendly_profile(config: SessionConfig) -> SessionConfig {
    let mut config = config
        .with_target_partitions(1)
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_joins(false)
        .with_repartition_sorts(false);
    let optimizer = &mut config.options_mut().optimizer;
    optimizer.enable_topk_dynamic_filter_pushdown = false;
    optimizer.enable_join_dynamic_filter_pushdown = false;
    optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    optimizer.enable_dynamic_filter_pushdown = false;
    optimizer.enable_physical_uncorrelated_scalar_subquery = false;
    config
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_catalog(ctx: &SessionContext, config: &ServerConfig) -> super::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    // Select which DataFusion TableProvider backs loaded Iceberg tables, based
    // on the session profile. The CPU path is only available when built with
    // the `iceberg-cpu` feature; without it Cpu+Iceberg falls back to the
    // GPU-only provider (server emits a startup WARN — see [`is_testing_profile_active`]
    // in `super::is_testing_profile_active`).
    let provider_kind = iceberg_provider_kind_for(config.session_profile);

    // Build the catalog backend once; it holds a connected catalog handle that
    // is reused for all namespace/table listing and lazy table loading. The
    // backend kind is selected by the validated config enum.
    let (backend, catalog_type): (Arc<dyn crate::table_format::iceberg::catalog::IcebergCatalogBackend>, &'static str) =
        match &iceberg.backend {
            super::config::IcebergCatalogBackendConfig::Glue(glue) => {
                let mut catalog_config = crate::table_format::iceberg::catalog::GlueIcebergCatalogConfig::new(
                    glue.catalog_handle.clone(),
                    glue.warehouse.clone(),
                    glue.region.clone(),
                );
                if let Some(id) = &glue.catalog_id {
                    catalog_config = catalog_config.with_catalog_id(id.clone());
                }
                if let Some(profile) = &glue.profile {
                    catalog_config = catalog_config.with_profile(profile.clone());
                }
                if let Some(uri) = &glue.uri {
                    catalog_config = catalog_config.with_uri(uri.clone());
                }
                if let Some(endpoint) = &glue.s3_endpoint {
                    catalog_config = catalog_config.with_s3_endpoint(endpoint.clone());
                }
                if let Some(region) = &glue.s3_region {
                    catalog_config = catalog_config.with_s3_region(region.clone());
                }
                if let Some(path_style) = glue.s3_path_style_access {
                    catalog_config = catalog_config.with_s3_path_style_access(path_style);
                }
                let backend =
                    crate::table_format::iceberg::catalog::GlueIcebergCatalogBackend::from_config_with_parquet_read_strategy_config(
                        catalog_config,
                        |name| std::env::var(name).ok(),
                        iceberg.parquet_read_strategy_config.clone(),
                    )
                    .await
                    .map_err(|error| {
                        ServerError::session("failed to connect to Glue Iceberg catalog backend", error)
                    })?;
                (Arc::new(backend), "glue")
            },
            super::config::IcebergCatalogBackendConfig::Rest(rest) => {
                let backend =
                    crate::table_format::iceberg::catalog::RestIcebergCatalogBackend::from_config_with_parquet_read_strategy_config(
                        rest.clone(),
                        |name| std::env::var(name).ok(),
                        iceberg.parquet_read_strategy_config.clone(),
                    )
                    .await
                    .map_err(|error| {
                        ServerError::session("failed to connect to REST Iceberg catalog backend", error)
                    })?;
                (Arc::new(backend), "rest")
            },
        };

    // Discover only the configured namespace scope. Individual tables are
    // loaded lazily on first SQL access through the backend.
    let provider_options = crate::table_format::iceberg::catalog::IcebergProviderOptions::default()
        .with_scan_planning_timeout(Some(iceberg.scan_planning_timeout));
    let catalog_provider =
        crate::table_format::iceberg::catalog::build_scoped_catalog_provider_from_backend_with_options(
            Arc::clone(&backend),
            catalog_type,
            &iceberg.namespace,
            provider_kind,
            provider_options,
        )
        .await
        .map_err(|error| ServerError::session("failed to build Iceberg catalog provider from backend", error))?;

    // Seed pre-declared alias mappings before registering the catalog.
    // This makes catalog.namespace.alias resolve in addition to the real name.
    // Loading routes through the backend-agnostic IcebergCatalogBackend so the
    // same path serves Glue and REST.
    for table in &iceberg.tables {
        let table_ident =
            crate::table_format::iceberg::catalog::table_ident_from_flat_name(&iceberg.namespace, &table.table)
                .map_err(|error| ServerError::session("invalid Iceberg table identifier for alias seeding", error))?;
        let loaded_table = backend
            .load_table(&table_ident)
            .await
            .map_err(|error| ServerError::session("failed to load Iceberg table for alias seeding", error))?;
        let provider = build_alias_provider(
            loaded_table,
            provider_kind,
            catalog_type,
            backend.parquet_read_strategy_config(),
            Some(iceberg.scan_planning_timeout),
        )
        .await?;

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
        provider_kind = provider_kind.as_str(),
        pre_declared_aliases = iceberg.tables.len(),
        "registered dynamic Iceberg catalog provider"
    );

    Ok(())
}

/// Maps a [`ServerSessionProfile`] to the Iceberg provider kind the dynamic
/// catalog should use.
///
/// Without the `iceberg-cpu` feature the CPU path is unavailable, so
/// `CpuFallback` still routes through the GPU-only provider; the server's
/// startup WARN (see `super::is_testing_profile_active`) covers that
/// configuration.
#[cfg(feature = "iceberg")]
fn iceberg_provider_kind_for(
    profile: ServerSessionProfile,
) -> crate::table_format::iceberg::catalog::IcebergProviderKind {
    use crate::table_format::iceberg::catalog::IcebergProviderKind;
    match profile {
        ServerSessionProfile::Native => IcebergProviderKind::Gpu,
        #[cfg(feature = "iceberg-cpu")]
        ServerSessionProfile::CpuFallback => IcebergProviderKind::Cpu,
        #[cfg(not(feature = "iceberg-cpu"))]
        ServerSessionProfile::CpuFallback => IcebergProviderKind::Gpu,
    }
}

/// Builds the [`TableProvider`] used to seed a pre-declared alias.
///
/// Mirrors the dispatch in [`crate::table_format::iceberg::catalog::CudfIcebergSchemaProvider`]
/// so the dynamic three-part path and the legacy bare-alias path resolve to
/// the same provider kind.
#[cfg(feature = "iceberg")]
async fn build_alias_provider(
    table: iceberg::table::Table,
    kind: crate::table_format::iceberg::catalog::IcebergProviderKind,
    catalog_type: &str,
    parquet_read_strategy_config: Option<crate::table_format::scan::TableFormatParquetReadStrategyConfig>,
    scan_planning_timeout: Option<std::time::Duration>,
) -> super::error::Result<Arc<dyn datafusion::datasource::TableProvider>> {
    use crate::table_format::iceberg::catalog::IcebergProviderKind;
    match kind {
        IcebergProviderKind::Gpu => {
            let provider = match parquet_read_strategy_config {
                Some(config) => {
                    crate::table_format::iceberg::CudfIcebergTableProvider::try_new_with_parquet_read_strategy_config(
                        table, config,
                    )
                },
                None => crate::table_format::iceberg::CudfIcebergTableProvider::try_new(table),
            }
            .map_err(|error| ServerError::session("failed to build CudfIcebergTableProvider", error))?
            .with_catalog_type(catalog_type)
            .with_scan_planning_timeout(scan_planning_timeout);
            Ok(Arc::new(provider))
        },
        #[cfg(feature = "iceberg-cpu")]
        IcebergProviderKind::Cpu => {
            let _ = (parquet_read_strategy_config, scan_planning_timeout);
            let provider =
                iceberg_datafusion::IcebergStaticTableProvider::try_new_from_table(table).await.map_err(|error| {
                    ServerError::session("failed to build iceberg-datafusion CPU table provider", error)
                })?;
            Ok(Arc::new(provider))
        },
        #[cfg(not(feature = "iceberg-cpu"))]
        IcebergProviderKind::Cpu => Err(ServerError::session(
            "iceberg-cpu feature is required to build CPU IcebergStaticTableProvider",
            crate::Error::new(
                crate::ErrorCode::CudfInvalidInput,
                "datafusion_nexus.server.session.build_alias_provider",
                "iceberg-cpu feature disabled at build time",
            ),
        )),
    }
}
