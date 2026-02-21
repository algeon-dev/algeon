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
    register_iceberg_tables(&ctx, config).await?;

    Ok(ctx)
}

#[cfg(feature = "iceberg")]
async fn register_iceberg_tables(ctx: &SessionContext, config: &ServerConfig) -> super::error::Result<()> {
    let Some(iceberg) = &config.iceberg else {
        return Ok(());
    };

    for table in &iceberg.tables {
        let mut glue = crate::iceberg::GlueIcebergTableConfig::new(
            iceberg.catalog_name.clone(),
            iceberg.database.clone(),
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
        let registration = crate::iceberg::register_glue_table(ctx, table.alias.as_str(), glue)
            .await
            .map_err(|error| ServerError::session("failed to register Glue Iceberg table", error))?;
        tracing::info!(
            catalog_type = "glue",
            catalog_name = registration.diagnostics.catalog_name,
            database = registration.diagnostics.database,
            table = registration.diagnostics.table,
            alias = table.alias,
            region = registration.diagnostics.region,
            credential_source = %registration.diagnostics.credential_source,
            "registered Iceberg table"
        );
    }

    Ok(())
}
