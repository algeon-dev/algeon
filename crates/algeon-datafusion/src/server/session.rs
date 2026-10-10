use std::{fs, path::Path};

use datafusion::{
    execution::SessionStateBuilder,
    logical_expr::{DdlStatement, LogicalPlan},
    prelude::{SessionConfig, SessionContext},
    sql::{parser::DFParserBuilder, sqlparser::dialect::dialect_from_str},
};

use crate::server::{
    adapter_error::Error as AdapterError,
    adapter_session::AlgeonSessionStateBuilderExt,
    backend::AlgeonGpuBackend,
    config::{CugraphServerConfig, ServerConfig},
    cugraph_sql::CugraphSqlConfig,
    cuml_sql::CumlSqlConfig,
    error::{Result as ServerResult, ServerError},
    flight::{reject_drop_object_type_mismatch, verify_update_ddl},
    observability::RedactedSql,
};

/// Builds the server's base DataFusion session bound to a shared GPU backend.
///
/// The backend's optimizer template overrides the config template. The function
/// installs cuVS and cuML SQL and optional cuGraph SQL. Session installation
/// failures retain a typed server error. [`ServerConfig::init_sql`] is not run
/// here; [`crate::server::FlightSqlServerBuilder::build`] runs it on the
/// shared session.
pub async fn build_session_context_with_backend(
    config: &ServerConfig,
    backend: &AlgeonGpuBackend,
) -> ServerResult<SessionContext> {
    let mut effective_config = config.clone();
    effective_config.native_optimizer_config = backend.optimizer_config().clone();
    let session_config = build_session_config(&effective_config)?;
    let builder = SessionStateBuilder::new_with_default_features().with_config(session_config);
    let builder = backend
        .install_on(builder)
        .map_err(|error| ServerError::session("failed to install backend native session", error))?;
    let builder = builder
        .try_with_cuvs_sql()
        .map_err(|error| ServerError::session("failed to install cuVS SQL family", error))?
        .try_with_cuml_sql_config(CumlSqlConfig { model_store: effective_config.models.clone() })
        .map_err(|error| ServerError::session("failed to install cuML SQL family", error))?;
    let builder = match &effective_config.cugraph {
        Some(cugraph) => builder
            .try_with_cugraph_sql(cugraph_sql_config(cugraph))
            .map_err(|error| ServerError::session("failed to install cuGraph SQL family", error))?,
        None => builder,
    };
    Ok(SessionContext::new_with_state(builder.build()))
}

/// Runs every statement of the `init_sql` file on `ctx`, in order. Each
/// statement must plan to DDL; any other statement, a parse error, or a
/// failed statement is `invalid_configuration` naming `init_sql`, the 1-based
/// statement index, and the start of the statement.
pub(crate) async fn run_init_sql(ctx: &SessionContext, path: &Path) -> ServerResult<()> {
    let sql = fs::read_to_string(path).map_err(|error| {
        ServerError::invalid_config_with_source("init_sql", format!("`{}` cannot be read", path.display()), error)
    })?;
    let state = ctx.state();
    // The parsed dialect is not `Send`; keep it out of the awaits below.
    let statements = {
        let options = state.config_options();
        let dialect = dialect_from_str(options.sql_parser.dialect).ok_or_else(|| {
            ServerError::invalid_config(
                "init_sql",
                format!("cannot be parsed with the configured SQL dialect `{}`", options.sql_parser.dialect),
            )
        })?;
        DFParserBuilder::new(sql.as_str())
            .with_dialect(dialect.as_ref())
            .with_recursion_limit(options.sql_parser.recursion_limit.get())
            .build()
            .and_then(|mut parser| parser.parse_statements())
            .map_err(|error| {
                ServerError::invalid_config_with_source(
                    "init_sql",
                    format!("`{}` is not valid SQL: {error}", path.display()),
                    error,
                )
            })?
    };
    for (index, statement) in statements.into_iter().enumerate() {
        let index = index + 1;
        let text = statement.to_string();
        // The excerpt reaches the startup log. DataFusion's rendering already
        // omits an OPTIONS clause; masking it as well keeps the excerpt safe
        // if that rendering ever includes one.
        let excerpt = RedactedSql(&text).to_string().chars().take(120).collect::<String>();
        let statement_error = |reason: String| {
            ServerError::invalid_config("init_sql", format!("statement {index} ({excerpt}) {reason}"))
                .with_fact("statement_index", index)
        };
        let plan = state
            .statement_to_plan(statement)
            .await
            .map_err(|error| statement_error(format!("failed to plan: {error}")))?;
        let LogicalPlan::Ddl(ddl) = &plan else {
            return Err(statement_error(
                "is not DDL; init_sql accepts DDL such as CREATE EXTERNAL TABLE or CREATE VIEW only".to_string(),
            )
            .with_fact("reason", "init_sql_statement_not_ddl"));
        };
        // The same policy as a Flight SQL client's DDL, so a file that starts
        // the server never contains a statement a client would be refused.
        let rejected = |error: AdapterError| {
            statement_error(format!("is not accepted: {}", error.message()))
                .with_fact("reason", "init_sql_statement_rejected")
        };
        verify_update_ddl(ddl, "algeon_datafusion.server.init_sql").map_err(rejected)?;
        reject_drop_object_type_mismatch(ctx, &plan, "algeon_datafusion.server.init_sql").await.map_err(rejected)?;
        let kind = ddl.name().to_string();
        let object = ddl_object_name(ddl);
        let frame =
            ctx.execute_logical_plan(plan).await.map_err(|error| statement_error(format!("failed: {error}")))?;
        frame.collect().await.map_err(|error| statement_error(format!("failed: {error}")))?;
        tracing::info!(statement_index = index, statement_kind = %kind, object = object.as_deref(), "ran init_sql statement");
    }
    Ok(())
}

fn ddl_object_name(ddl: &DdlStatement) -> Option<String> {
    match ddl {
        DdlStatement::CreateExternalTable(table) => Some(table.name.to_string()),
        DdlStatement::CreateMemoryTable(table) => Some(table.name.to_string()),
        DdlStatement::CreateView(view) => Some(view.name.to_string()),
        DdlStatement::CreateCatalogSchema(schema) => Some(schema.schema_name.clone()),
        DdlStatement::CreateCatalog(catalog) => Some(catalog.catalog_name.clone()),
        DdlStatement::CreateFunction(function) => Some(function.name.clone()),
        DdlStatement::DropTable(table) => Some(table.name.to_string()),
        DdlStatement::DropView(view) => Some(view.name.to_string()),
        DdlStatement::DropFunction(function) => Some(function.name.clone()),
        DdlStatement::CreateIndex(index) => index.name.clone(),
        DdlStatement::DropCatalogSchema(_) => None,
    }
}

pub(crate) fn cugraph_sql_config(config: &CugraphServerConfig) -> CugraphSqlConfig {
    CugraphSqlConfig {
        default_graph_options: config.default_graph_options.clone(),
        host_boundary_config: config.host_boundary_config,
    }
}

/// Build the `SessionConfig` for the Flight SQL server from DataFusion's
/// standard environment configuration. The server enables catalog discovery
/// but does not change optimizer or execution settings beyond what the
/// configured final plan requirement implies (see below).
pub(crate) fn build_session_config(config: &ServerConfig) -> ServerResult<SessionConfig> {
    let gpu_only = config.native_optimizer_config.execution_mode().rejects_datafusion_cpu();
    SessionConfig::from_env()
        .map(enable_session_discovery)
        .map(|session_config| {
            // Under the GPU-only requirement no query can execute on the CPU,
            // so DataFusion-side parallelism is unreachable. Single-group
            // scan planning keeps sort-declared fixtures (the source-sorted
            // DBLP layout) eligible for native lowering. A declared scan
            // ordering cannot be reproduced across parallel byte-range groups.
            if gpu_only { session_config.with_target_partitions(1) } else { session_config }
        })
        .map_err(|error| ServerError::session("failed to build DataFusion session config from environment", error))
}

pub(super) fn enable_session_discovery(config: SessionConfig) -> SessionConfig {
    config.with_information_schema(true)
}

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use super::*;
    use crate::server::planner::AlgeonGpuExecutionMode;

    fn server_config_with_execution_mode(mode: &str) -> ServerConfig {
        ServerConfig::from_vars([("ALGEON_SERVER_NATIVE_EXECUTION_MODE", mode)]).expect("load server config")
    }

    #[test]
    fn execution_mode_env_maps_to_functions_only_and_preserves_partitions() {
        let server = server_config_with_execution_mode("functions_only");
        assert_eq!(server.native_optimizer_config.execution_mode(), AlgeonGpuExecutionMode::FunctionsOnly);
        let session = build_session_config(&server).expect("session config");
        let baseline = SessionConfig::from_env().expect("baseline session config");
        assert_eq!(
            session.options().execution.target_partitions,
            baseline.options().execution.target_partitions,
            "functions-only must preserve DataFusion's configured parallelism"
        );
    }

    #[test]
    fn execution_mode_env_maps_native_required_to_single_partition() {
        let server = server_config_with_execution_mode("native_required");
        let session = build_session_config(&server).expect("session config");
        assert_eq!(session.options().execution.target_partitions, 1);
    }
}
