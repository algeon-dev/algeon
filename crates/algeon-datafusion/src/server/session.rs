use datafusion::{
    execution::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};

use crate::server::{
    adapter_session::AlgeonSessionStateBuilderExt,
    backend::AlgeonGpuBackend,
    config::{CugraphServerConfig, ServerConfig},
    cugraph_sql::CugraphSqlConfig,
    cuml_sql::CumlSqlConfig,
    error::{Result as ServerResult, ServerError},
    workspace::install_workspace_overlay,
};

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
    let state = builder.build();
    let ctx = SessionContext::new_with_state(state);

    if let Some(workspace) = &effective_config.workspace {
        install_workspace_overlay(&ctx, workspace)?;
    }

    Ok(ctx)
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
    use std::collections::BTreeMap;

    use super::*;
    use crate::server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, config::ALGEON_SERVER_EXECUTION_MODE_ENV, planner::AlgeonGpuExecutionMode,
    };

    fn server_config_with_test_device_profile(mut env: BTreeMap<&'static str, String>) -> ServerConfig {
        let config_file = tempfile::NamedTempFile::new().expect("create server config fixture");
        std::fs::write(config_file.path(), "[[admission.device_profiles]]\ndevice_ordinal = 0\n")
            .expect("write server config fixture");
        env.insert(ALGEON_SERVER_CONFIG_FILE_ENV, config_file.path().to_string_lossy().into_owned());
        ServerConfig::from_env_get(|name| env.get(name).cloned()).expect("load server config fixture")
    }

    #[test]
    fn execution_mode_env_maps_to_functions_only_and_preserves_partitions() {
        let env = BTreeMap::from([(ALGEON_SERVER_EXECUTION_MODE_ENV, "functions_only".to_string())]);
        let server = server_config_with_test_device_profile(env);
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
        let env = BTreeMap::from([(ALGEON_SERVER_EXECUTION_MODE_ENV, "native_required".to_string())]);
        let server = server_config_with_test_device_profile(env);
        let session = build_session_config(&server).expect("session config");
        assert_eq!(session.options().execution.target_partitions, 1);
    }
}
