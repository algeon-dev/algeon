use std::{
    collections::BTreeMap,
    fs, panic,
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::{
    native_exec::NexusNativeExecutionConfig,
    native_report::{
        NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPlanningWorkloadInput,
        ParityFloatTolerance,
    },
    planner::NexusNativeOptimizerConfig,
};
use nexus_bench::{
    report_metadata::ReportDatasetMetadataExt,
    report_workload::{
        RelationalExecutionSelection,
        tpcds::{
            NEXUS_TPCDS_NATIVE_COST_SELECTION_ENV, NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV,
            NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV,
            NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV,
            NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV,
            NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH_ENV, NEXUS_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV,
            NEXUS_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, NEXUS_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
            NEXUS_TPCDS_NATIVE_PROFILE_ENV, NEXUS_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV,
            TPCDS_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS, TPCDS_NATIVE_PROFILE_CONSERVATIVE,
            TPCDS_NATIVE_PROFILE_CUSTOM, TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT,
            TPCDS_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES, TPCDS_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
            TPCDS_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES, TpcdsNativeProfileSettings, TpcdsParityProgressEvent,
            tpcds_datafusion_planning_workload_items, tpcds_native_analytics_optimizer_config_with_profile_settings,
            tpcds_native_cost_selection_options_with_env, tpcds_native_optimizer_config_with_profile_settings,
            tpcds_native_profile_settings_with_env, tpcds_native_report_session_config,
            tpcds_parity_report_rows_with_external_oracles, tpcds_report_session_config,
        },
    },
};
use nexus_query_engine::mode::NativePipelineExecutionMode;
use nexus_workloads::tpcds::{
    TPCDS_QUERIES, TPCDS_QUERY_COUNT, TPCDS_TABLES, describe_tpcds_dataset, tiny_tpcds_dataset_metadata,
    tpcds_query_ids, write_tiny_tpcds_parquet_dataset,
};

#[test]
fn tpcds_query_count_matches_official_99_queries() {
    assert_eq!(TPCDS_QUERY_COUNT, 99);
    assert_eq!(TPCDS_QUERIES.len(), 99);
}

#[test]
fn strict_native_selection_requires_a_completed_plan_without_datafusion_cpu() {
    let optimizer = RelationalExecutionSelection::strict_native()
        .configure_native_optimizer(NexusNativeOptimizerConfig::default())
        .expect("strict native execution installs a native rule");

    assert_eq!(
        optimizer.final_plan_requirements(),
        &[datafusion_nexus::planner::FinalPlanRequirement::NoDataFusionCpu]
    );
}

#[test]
fn tpcds_query_ids_are_q01_through_q99_in_order() {
    let expected: Vec<String> = (1..=99).map(|n| format!("q{n:02}")).collect();
    let actual: Vec<&'static str> = tpcds_query_ids().collect();
    assert_eq!(actual, expected.iter().map(String::as_str).collect::<Vec<_>>());
    let from_table: Vec<&'static str> = TPCDS_QUERIES.iter().map(|query| query.id).collect();
    assert_eq!(from_table, expected.iter().map(String::as_str).collect::<Vec<_>>());
}

#[test]
fn tpcds_query_sql_strings_are_non_empty_and_have_no_trailing_semicolon() {
    for query in TPCDS_QUERIES {
        let sql = query.sql;
        assert!(!sql.trim().is_empty(), "{} sql is empty", query.id);
        let trimmed = sql.trim_end();
        assert!(!trimmed.ends_with(';'), "{} sql has a trailing semicolon", query.id);
    }
}

#[test]
fn tpcds_tables_list_has_24_official_tpcds_tables() {
    assert_eq!(TPCDS_TABLES.len(), 24);
    for table in ["store_sales", "catalog_sales", "web_sales", "date_dim", "customer", "item", "web_site"] {
        assert!(TPCDS_TABLES.contains(&table), "TPCDS_TABLES missing {table}");
    }
}

#[test]
fn tpcds_production_planning_builder_preserves_order_and_retained_diagnostics() {
    std::thread::Builder::new()
        .name("tpcds-production-planning-builder".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
                let dir = unique_temp_dir("datafusion-nexus-tpcds-production-planning-builder");
                write_tiny_tpcds_parquet_dataset(&dir).unwrap();

                let items =
                    tpcds_datafusion_planning_workload_items(&dir, tpcds_report_session_config()).await.unwrap();
                let expected_ids = tpcds_query_ids().collect::<Vec<_>>();
                assert_eq!(items.len(), TPCDS_QUERY_COUNT);
                assert_eq!(items.iter().map(|item| item.query_id.as_str()).collect::<Vec<_>>(), expected_ids);
                for item in &items {
                    match &item.input {
                        NativeDataFusionPlanningWorkloadInput::Planned { native_diagnostics, .. } => {
                            assert!(!native_diagnostics.report.outcomes().is_empty(), "{}", item.query_id);
                        },
                        NativeDataFusionPlanningWorkloadInput::PlanningFailure(failure) => {
                            panic!(
                                "{} unexpectedly failed at {}: {}",
                                item.query_id, failure.stage, failure.reason_code
                            );
                        },
                    }
                }

                let report = NativeDataFusionPlanningReport::collect(items);
                let mut report_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
                report_ids.dedup();
                assert_eq!(report_ids, expected_ids);
                assert_eq!(report.selected_count(), 99);
                assert_eq!(report.not_selected_count(), 0);
                for query_id in ["q01", "q39", "q72"] {
                    let row = report.rows.iter().find(|row| row.query_id == query_id).unwrap();
                    assert_eq!(row.outcome, Some(NativeDataFusionPlanningOutcome::Selected), "{query_id}");
                    assert_eq!(row.reason_code, None, "{query_id}");
                    assert_eq!(row.unsupported_category, None, "{query_id}");
                }

                fs::remove_dir_all(dir).unwrap();
            });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn tpcds_native_report_session_config_avoids_datafusion_statistics_estimates() {
    let report_config = tpcds_report_session_config();
    let native_config = tpcds_native_report_session_config();

    assert!(
        report_config.options().execution.collect_statistics,
        "shared TPC-DS report config remains CPU-compatible; native contexts apply their own statistics safety"
    );
    assert!(
        !native_config.options().execution.collect_statistics,
        "native TPC-DS report planning reads source metadata directly and must not depend on DataFusion join statistics"
    );
    assert!(
        report_config.options().optimizer.enable_physical_uncorrelated_scalar_subquery,
        "shared TPC-DS report config must preserve DataFusion's CPU scalar-subquery default"
    );
    assert!(
        !native_config.options().optimizer.enable_physical_uncorrelated_scalar_subquery,
        "native TPC-DS report contexts must preserve the relationalized scalar subquery contract"
    );
    assert!(!native_config.options().optimizer.repartition_joins);
    assert_eq!(native_config.options().execution.target_partitions, 1);
}

#[test]
fn tpcds_native_optimizer_configs_disable_cost_selection_for_gpu_only_sql_surface() {
    let profile = tpcds_native_profile_settings_with_env(|_| None);
    let ordinary_config = tpcds_native_optimizer_config_with_profile_settings(&profile);
    let analytics_config = tpcds_native_analytics_optimizer_config_with_profile_settings(&profile);
    let env_overridden_cost_selection = tpcds_native_cost_selection_options_from([
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_ENV, "on"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV, "999"),
    ]);

    assert!(env_overridden_cost_selection.enabled);
    assert_eq!(env_overridden_cost_selection.max_small_selective_residual_join_row_groups, 999);
    assert!(
        !ordinary_config.native_plan_cost_options().enabled,
        "TPC-DS native surfaces must not emit cost-selection rows while SQL may contain GPU-only cugraph_* calls"
    );
    assert!(
        !analytics_config.native_plan_cost_options().enabled,
        "TPC-DS native analytics/GPU benchmark surfaces must keep all supported plans on native GPU"
    );
    assert_eq!(analytics_config.native_pipeline_execution_mode(), NativePipelineExecutionMode::StrictBounded);
}

#[test]
fn tpcds_cost_declined_datafusion_config_enables_cost_selection() {
    let profile = tpcds_native_profile_settings_with_env(|_| None);
    let config = tpcds_native_analytics_optimizer_config_with_profile_settings(&profile)
        .with_native_plan_cost_options(tpcds_native_cost_selection_options_from([]));

    assert!(
        config.native_plan_cost_options().enabled,
        "cost_declined_datafusion must classify small/selective native plans before its DataFusion comparison"
    );
    assert_eq!(
        config.native_plan_cost_options().max_small_selective_residual_join_row_groups,
        TPCDS_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS,
        "TPC-DS cost selection should admit low-byte residual joins spread over a modest number of row groups"
    );
}

#[test]
fn tpcds_cost_selection_options_apply_env_overrides() {
    let options = tpcds_native_cost_selection_options_from([
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_ENV, "off"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_ROW_GROUPS_ENV, "17"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_SMALL_SELECTIVE_SOURCE_BYTES_ENV, "123456"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_ROW_GROUPS_ENV, "24"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_RESIDUAL_JOIN_SOURCE_BYTES_ENV, "654321"),
        (NEXUS_TPCDS_NATIVE_COST_SELECTION_MAX_TOPK_FETCH_ENV, "42"),
    ]);

    assert!(!options.enabled);
    assert_eq!(options.max_small_selective_row_groups, 17);
    assert_eq!(options.max_small_selective_source_bytes, 123456);
    assert_eq!(options.max_small_selective_residual_join_row_groups, 24);
    assert_eq!(options.max_small_selective_residual_join_source_bytes, 654321);
    assert_eq!(options.max_topk_fetch, 42);
}

#[test]
fn tpcds_native_profile_defaults_to_conservative() {
    let profile = tpcds_native_profile_settings_from([]);

    assert_eq!(profile.profile_name, TPCDS_NATIVE_PROFILE_CONSERVATIVE);
    assert_eq!(profile.profile_source, "default");
    assert_eq!(profile.max_source_chunk_bytes, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn tpcds_native_profile_resolves_sf100_gpu_throughput() {
    let profile = tpcds_native_profile_settings_from([(NEXUS_TPCDS_NATIVE_PROFILE_ENV, "sf100_gpu_throughput")]);

    assert_eq!(profile.profile_name, TPCDS_NATIVE_PROFILE_SF100_GPU_THROUGHPUT);
    assert_eq!(profile.profile_source, "env_profile");
    assert_eq!(profile.device_budget_bytes, TPCDS_SF100_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES);
    assert_eq!(profile.max_source_chunk_bytes, TPCDS_SF100_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, TPCDS_SF100_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn tpcds_native_profile_numeric_overrides_become_custom() {
    let profile = tpcds_native_profile_settings_from([
        (NEXUS_TPCDS_NATIVE_PROFILE_ENV, "conservative"),
        (NEXUS_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV, "123456789"),
        (NEXUS_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "268435456"),
        (NEXUS_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8"),
    ]);

    assert_eq!(profile.profile_name, TPCDS_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(profile.device_budget_bytes, 123456789);
    assert_eq!(profile.max_source_chunk_bytes, 268435456);
    assert_eq!(profile.max_row_groups_per_chunk, 8);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn tpcds_native_profile_source_chunk_planning_mode_override_becomes_custom() {
    let profile = tpcds_native_profile_settings_from([
        (NEXUS_TPCDS_NATIVE_PROFILE_ENV, "conservative"),
        (NEXUS_TPCDS_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput"),
    ]);

    assert_eq!(profile.profile_name, TPCDS_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn tpcds_native_profile_rejects_zero_numeric_values() {
    let result = panic::catch_unwind(|| {
        tpcds_native_profile_settings_from([
            (NEXUS_TPCDS_NATIVE_PROFILE_ENV, "sf100_gpu_throughput"),
            (NEXUS_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "0"),
        ]);
    });

    assert!(result.is_err(), "zero row-group caps must be rejected at profile resolution");
}

#[test]
fn write_tiny_tpcds_parquet_dataset_emits_24_parquet_files() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-fixture");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let mut files: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            name.strip_suffix(".parquet").map(|stem| stem.to_string())
        })
        .collect();
    files.sort();

    let mut expected = TPCDS_TABLES.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    expected.sort();
    assert_eq!(files, expected);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_tpcds_dataset_metadata_describes_24_tables() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-describe");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpcds_dataset_metadata(&dir).unwrap();
    assert_eq!(metadata.table_count, 24);
    assert_eq!(metadata.file_count, 24);
    assert_eq!(metadata.scale_factor, "tiny");
    assert_eq!(metadata.provenance, "generated_tiny");
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.is_none());

    let table_names: Vec<&str> = metadata.tables.iter().map(|table| table.table.as_str()).collect();
    for expected in TPCDS_TABLES {
        assert!(table_names.contains(expected), "metadata.tables missing {expected}");
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn tiny_tpcds_dataset_metadata_projects_into_workload_dataset_metadata() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-tiny-workload-projection");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let metadata = describe_tpcds_dataset(&dir, "tiny", "generated_tiny", "test_projection").unwrap();
    let workload = metadata.to_workload_dataset_metadata();
    assert_eq!(workload.workload, "tpcds");
    assert_eq!(workload.scale_factor.as_deref(), Some("tiny"));
    assert_eq!(workload.provenance, "generated_tiny");
    assert_eq!(workload.generation_command, "test_projection");
    assert_eq!(workload.table_count, 24);
    assert_eq!(workload.file_count, 24);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpcds_parity_report_rows_emit_progress_per_completed_query() {
    let dir = unique_temp_dir("datafusion-nexus-tpcds-streaming-parity");
    write_tiny_tpcds_parquet_dataset(&dir).unwrap();

    let query_filter = vec!["q01".to_string()];
    let mut events = Vec::new();
    let mut emitted_lines = Vec::new();
    let mut progress = |event: TpcdsParityProgressEvent<'_>| match event {
        TpcdsParityProgressEvent::QueryStarted { query_index, total, query_id } => {
            events.push(format!("start:{query_index}/{total}:{query_id}"));
        },
        TpcdsParityProgressEvent::RowReady { query_index, total, query_id, row } => {
            events.push(format!("row:{query_index}/{total}:{query_id}"));
            emitted_lines.push(row.to_tsv_line());
        },
    };

    let rows = tpcds_parity_report_rows_with_external_oracles(
        &dir,
        tpcds_report_session_config(),
        NexusNativeOptimizerConfig::default(),
        Some(&query_filter),
        None,
        ParityFloatTolerance::exact(),
        Some(&mut progress),
    )
    .await
    .unwrap();

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].query_id, "q01");
    assert_eq!(events, vec!["start:1/1:q01", "row:1/1:q01"]);
    assert_eq!(emitted_lines, vec![rows[0].to_tsv_line()]);

    fs::remove_dir_all(dir).unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn tpcds_native_profile_settings_from(
    pairs: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> TpcdsNativeProfileSettings {
    let env = pairs.into_iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
    tpcds_native_profile_settings_with_env(|name| env.get(name).cloned())
}

fn tpcds_native_cost_selection_options_from(
    pairs: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> nexus_query_engine::plan::NativePlanCostOptions {
    let env = pairs.into_iter().map(|(key, value)| (key, value.to_string())).collect::<BTreeMap<_, _>>();
    tpcds_native_cost_selection_options_with_env(|name| env.get(name).cloned())
}
