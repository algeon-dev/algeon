use super::*;

const DEFAULT_QUERY_IDS: &[&str] =
    &["q03", "q07", "q22", "q26", "q32", "q37", "q42", "q52", "q55", "q62", "q82", "q85", "q92", "q93", "q97"];

impl BenchConfig {
    pub(super) fn from_env() -> Result<Self, Box<dyn Error>> {
        let engine = match std::env::var("NEXUS_TPCDS_BENCH_ENGINE").as_deref() {
            Ok("native_gpu") => Engine::NativeGpu,
            Ok("cpu_datafusion") => Engine::CpuDataFusion,
            Ok("cost_declined_datafusion") => Engine::CostDeclinedDataFusion,
            Ok(other) => return Err(format!("unsupported NEXUS_TPCDS_BENCH_ENGINE={other}").into()),
            Err(_) => {
                return Err(
                    "NEXUS_TPCDS_BENCH_ENGINE must be set to native_gpu, cpu_datafusion, or cost_declined_datafusion"
                        .into(),
                );
            },
        };
        let root = std::env::var(nexus_workloads::NEXUS_PARQUET_DIR_ENV)
            .map_err(|_| "NEXUS_PARQUET_DIR must be set to the SF100 parquet root")?;
        let root = std::path::PathBuf::from(root);
        let queries_csv = std::env::var("NEXUS_TPCDS_BENCH_QUERIES").unwrap_or_else(|_| DEFAULT_QUERY_IDS.join(","));
        let queries = bench_queries_from_csv(&queries_csv)?;
        let warmups = env_usize("NEXUS_TPCDS_BENCH_WARMUPS", 1)?;
        let iterations = env_usize("NEXUS_TPCDS_BENCH_ITERATIONS", 2)?;
        if iterations == 0 {
            return Err("NEXUS_TPCDS_BENCH_ITERATIONS must be >= 1".into());
        }
        let native_profile = match engine {
            Engine::CpuDataFusion => tpcds_native_profile_settings_with_env(|_| None),
            Engine::NativeGpu | Engine::CostDeclinedDataFusion => tpcds_native_profile_settings_from_env(),
        };
        let native_pipeline_execution_mode =
            tpcds_native_analytics_optimizer_config_with_profile_settings(&native_profile)
                .native_pipeline_execution_mode();
        Ok(Self {
            engine,
            root,
            queries,
            warmups,
            iterations,
            native_profile,
            relational_execution: match engine {
                Engine::CpuDataFusion => RelationalExecutionSelection::CpuBaseline,
                Engine::NativeGpu => RelationalExecutionSelection::strict_native(),
                Engine::CostDeclinedDataFusion => RelationalExecutionSelection::native(),
            },
            native_pipeline_execution_mode,
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
            let session_config = tuned_cpu_session_config();
            let ctx = SessionContext::new_with_config(session_config);
            register_tpcds_tables_with_native_metadata(&ctx, &cfg.root).await?;
            Ok(BenchRuntime { execution_ctx: ctx, optimizer: None, native_planning: None, skip_cost_declined: false })
        },
        Engine::NativeGpu => {
            let optimizer_config = cfg
                .relational_execution
                .configure_native_optimizer(
                    tpcds_native_analytics_optimizer_config_with_profile_settings(&cfg.native_profile)
                        .with_native_pipeline_execution_mode(cfg.native_pipeline_execution_mode),
                )
                .expect("native GPU benchmark installs the native rule");
            let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
            let state = SessionStateBuilder::new_with_default_features()
                .with_config(tpcds_native_report_session_config())
                .with_nexus_native_rule(Arc::clone(&rule))
                .build();
            let ctx = SessionContext::new_with_state(state);
            register_tpcds_tables_with_native_metadata(&ctx, &cfg.root).await?;
            Ok(BenchRuntime {
                execution_ctx: ctx,
                optimizer: Some(rule),
                native_planning: None,
                skip_cost_declined: true,
            })
        },
        Engine::CostDeclinedDataFusion => {
            let cpu_ctx = SessionContext::new_with_config(tuned_cpu_session_config());
            register_tpcds_tables_with_native_metadata(&cpu_ctx, &cfg.root).await?;
            let (native_ctx, optimizer) = build_native_planning_context(
                &cfg.root,
                &cfg.native_profile,
                cfg.relational_execution,
                cfg.native_pipeline_execution_mode,
            )
            .await?;
            Ok(BenchRuntime {
                execution_ctx: cpu_ctx,
                optimizer: None,
                native_planning: Some(NativePlanningRuntime { ctx: native_ctx, optimizer }),
                skip_cost_declined: false,
            })
        },
    }
}

async fn build_native_planning_context(
    root: &Path,
    native_profile: &TpcdsNativeProfileSettings,
    relational_execution: RelationalExecutionSelection,
    native_pipeline_execution_mode: NativePipelineExecutionMode,
) -> Result<(SessionContext, Arc<NexusNativePhysicalOptimizerRule>), Box<dyn Error>> {
    let optimizer_config = relational_execution
        .configure_native_optimizer(
            tpcds_native_analytics_optimizer_config_with_cost_selection_and_profile_settings(native_profile)
                .with_native_pipeline_execution_mode(native_pipeline_execution_mode),
        )
        .expect("cost-decision planning session installs the native rule");
    let rule = Arc::new(NexusNativePhysicalOptimizerRule::new(optimizer_config));
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(tpcds_native_report_session_config())
        .with_nexus_native_rule(Arc::clone(&rule))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_tpcds_tables_with_native_metadata(&ctx, root).await?;
    Ok((ctx, rule))
}

fn tuned_cpu_session_config() -> SessionConfig {
    let target_partitions =
        std::env::var("NEXUS_TPCDS_CPU_TARGET_PARTITIONS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16);
    let batch_size =
        std::env::var("NEXUS_TPCDS_CPU_BATCH_SIZE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(8192);
    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size)
        .with_repartition_file_scans(true)
        .with_repartition_aggregations(true)
        .with_repartition_joins(true)
        .with_repartition_sorts(true);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_join_dynamic_filter_pushdown = true;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = true;
    config.options_mut().execution.parquet.pushdown_filters = false;
    config
}
