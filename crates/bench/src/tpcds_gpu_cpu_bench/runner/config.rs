use std::error::Error;

use super::{
    BenchConfig, BenchRuntime, Engine, GpuDeviceMemoryLimit, RelationalExecutionSelection, SessionContext,
    SessionStateBuilder, TPCDS_QUERIES, TpcdsQuery, register_tpcds_tables_with_native_metadata,
    tpcds_native_analytics_optimizer_config_with_profile_settings,
    tpcds_native_profile_settings_and_device_memory_limit_from_env, tpcds_native_profile_settings_with_env,
    tpcds_native_report_session_config,
};
use crate::{
    gpu_backend::{build_report_gpu_backend, close_report_gpu_backend},
    report_workload::tpcds::tpcds_tuned_cpu_session_config,
};

const DEFAULT_QUERY_IDS: &[&str] =
    &["q03", "q07", "q22", "q26", "q32", "q37", "q42", "q52", "q55", "q62", "q82", "q85", "q92", "q93", "q97"];

impl BenchConfig {
    pub(super) fn from_env() -> Result<Self, Box<dyn Error>> {
        let engine = match std::env::var("ALGEON_TPCDS_BENCH_ENGINE").as_deref() {
            Ok("native_gpu") => Engine::NativeGpu,
            Ok("cpu_datafusion") => Engine::CpuDataFusion,
            Ok(other) => return Err(format!("unsupported ALGEON_TPCDS_BENCH_ENGINE={other}").into()),
            Err(_) => return Err("ALGEON_TPCDS_BENCH_ENGINE must be set to native_gpu or cpu_datafusion".into()),
        };
        let root = std::env::var(workloads::ALGEON_PARQUET_DIR_ENV)
            .map_err(|_| "ALGEON_PARQUET_DIR must be set to the SF100 parquet root")?;
        let root = std::path::PathBuf::from(root);
        let queries_csv = std::env::var("ALGEON_TPCDS_BENCH_QUERIES").unwrap_or_else(|_| DEFAULT_QUERY_IDS.join(","));
        let queries = bench_queries_from_csv(&queries_csv)?;
        let warmups = env_usize("ALGEON_TPCDS_BENCH_WARMUPS", 1)?;
        let iterations = env_usize("ALGEON_TPCDS_BENCH_ITERATIONS", 2)?;
        if iterations == 0 {
            return Err("ALGEON_TPCDS_BENCH_ITERATIONS must be >= 1".into());
        }
        let (native_profile, native_device_memory_limit) = match engine {
            Engine::CpuDataFusion => {
                (tpcds_native_profile_settings_with_env(|_| None), GpuDeviceMemoryLimit::default())
            },
            Engine::NativeGpu => tpcds_native_profile_settings_and_device_memory_limit_from_env(),
        };
        Ok(Self {
            engine,
            root,
            queries,
            warmups,
            iterations,
            native_profile,
            native_device_memory_limit,
            relational_execution: match engine {
                Engine::CpuDataFusion => RelationalExecutionSelection::CpuBaseline,
                Engine::NativeGpu => RelationalExecutionSelection::strict_native(),
            },
        })
    }
}

pub(super) fn bench_queries_from_csv(queries_csv: &str) -> Result<Vec<&'static TpcdsQuery>, Box<dyn Error>> {
    if queries_csv.trim().eq_ignore_ascii_case("all") {
        return Ok(TPCDS_QUERIES.iter().collect());
    }
    let query_ids: Vec<String> =
        queries_csv.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let queries: Vec<&'static TpcdsQuery> = query_ids
        .iter()
        .map(|id| TPCDS_QUERIES.iter().find(|q| q.id == id).ok_or_else(|| format!("unknown TPC-DS query id: {id}")))
        .collect::<Result<_, _>>()?;
    Ok(queries)
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|e| format!("{name}={value}: {e}").into()),
        Err(_) => Ok(default),
    }
}

pub(super) async fn build_runtime(cfg: &BenchConfig) -> Result<BenchRuntime, Box<dyn Error>> {
    match cfg.engine {
        Engine::CpuDataFusion => {
            let session_config = tpcds_tuned_cpu_session_config();
            let ctx = SessionContext::new_with_config(session_config);
            register_tpcds_tables_with_native_metadata(&ctx, &cfg.root).await?;
            Ok(BenchRuntime { execution_ctx: ctx, native_rule_installed: false, backend: None })
        },
        Engine::NativeGpu => {
            let optimizer_config = cfg
                .relational_execution
                .configure_native_optimizer(tpcds_native_analytics_optimizer_config_with_profile_settings(
                    &cfg.native_profile,
                ))
                .expect("native GPU benchmark installs the native rule");
            let backend = build_report_gpu_backend(optimizer_config, cfg.native_device_memory_limit)?;
            let native = async {
                let builder =
                    SessionStateBuilder::new_with_default_features().with_config(tpcds_native_report_session_config());
                let state = backend.install_on(builder)?.build();
                let ctx = SessionContext::new_with_state(state);
                register_tpcds_tables_with_native_metadata(&ctx, &cfg.root).await?;
                Ok(ctx)
            }
            .await;
            let ctx = match native {
                Ok(native) => native,
                Err(error) => {
                    let _ = close_report_gpu_backend(backend);
                    return Err(error);
                },
            };
            Ok(BenchRuntime { execution_ctx: ctx, native_rule_installed: true, backend: Some(backend) })
        },
    }
}
