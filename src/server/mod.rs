//! Opt-in Flight SQL server for `datafusion-cudf`.

use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use tonic::transport::Server;

mod admission;
mod cancel;
pub mod config;
pub mod error;
mod flight;
mod session;

pub use config::{
    DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_ID_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_CATALOG_NAME_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_DATABASE_ENV, DATAFUSION_CUDF_ICEBERG_GLUE_TABLES_ENV,
    DATAFUSION_CUDF_ICEBERG_GLUE_WAREHOUSE_ENV, DATAFUSION_CUDF_SERVER_BIND_ENV, DATAFUSION_CUDF_SERVER_LOG_ENV,
    DATAFUSION_CUDF_SERVER_MAX_IN_FLIGHT_ENV, DATAFUSION_CUDF_SERVER_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    DATAFUSION_CUDF_SERVER_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, ServerConfig,
};
pub use error::{Result, ServerError};

/// Start the Flight SQL server and serve until the process is stopped.
pub async fn serve(config: ServerConfig) -> Result<()> {
    let addr = config.bind_addr;
    let ctx = Arc::new(session::build_session_context(&config).await?);
    let service = flight::DatafusionCudfFlightSqlService::new(ctx, config.max_in_flight);

    tracing::info!(
        bind_addr = %addr,
        max_in_flight = config.max_in_flight,
        "starting datafusion-cudf Flight SQL server"
    );

    Server::builder().add_service(FlightServiceServer::new(service)).serve(addr).await.map_err(ServerError::transport)
}
