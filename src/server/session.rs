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
/// The base config is loaded from `DATAFUSION_*` env vars (single-underscore
/// names matching DataFusion's
/// [`ConfigOptions::from_env`](datafusion::common::config::ConfigOptions::from_env)
/// convention) so operators can still tune the CPU-fallback execution profile
/// through e.g. `DATAFUSION_EXECUTION_BATCH_SIZE`. When the profile is
/// [`ServerSessionProfile::Native`] the relevant DataFusion settings are
/// overridden afterwards so the native cudf optimizer rule sees the same
/// physical-plan shape the workload harnesses already validate. This avoids
/// the audit scenario in `docs/issue/2026-05-22-flight-sql-server-audit.md`
/// where DataFusion repartitioning injects a `SortPreservingMergeExec` over a
/// GPU-only `CudfTableFormatScanExec` and the native rule rejects the root.
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
/// Embedders that build their own [`SessionConfig`] (and thus do not want to
/// inherit `DATAFUSION_*` process env) can call this directly to opt into the
/// same plan-shape contract the Flight SQL server enforces.
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
    config
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
