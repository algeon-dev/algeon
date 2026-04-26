use std::panic;

use datafusion_nexus::native_exec::NexusNativeExecutionConfig;
use datafusion_nexus_bench::report_workload::clickbench::{
    CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES, CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK,
    CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES, CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE,
    CLICKBENCH_NATIVE_PROFILE_CUSTOM, CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT,
    NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV,
    NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, clickbench_native_optimizer_config_with_profile_settings,
    clickbench_native_profile_settings_with_env,
};

fn clickbench_native_profile_settings_from<const N: usize>(
    env: [(&'static str, &str); N],
) -> datafusion_nexus_bench::report_workload::clickbench::ClickbenchNativeProfileSettings {
    clickbench_native_profile_settings_with_env(|name| {
        env.iter().find_map(|(key, value)| (*key == name).then(|| (*value).to_string()))
    })
}

#[test]
fn clickbench_native_profile_defaults_to_conservative() {
    let profile = clickbench_native_profile_settings_from([]);

    assert_eq!(profile.profile_name, CLICKBENCH_NATIVE_PROFILE_CONSERVATIVE);
    assert_eq!(profile.profile_source, "default");
    assert_eq!(profile.max_source_chunk_bytes, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, NexusNativeExecutionConfig::DEFAULT_BOUNDED_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn clickbench_native_profile_resolves_gpu_throughput() {
    let profile =
        clickbench_native_profile_settings_from([(NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "clickbench_gpu_throughput")]);

    assert_eq!(profile.profile_name, CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT);
    assert_eq!(profile.profile_source, "env_profile");
    assert_eq!(profile.device_budget_bytes, CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES);
    assert_eq!(profile.max_source_chunk_bytes, CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES);
    assert_eq!(profile.max_row_groups_per_chunk, CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn clickbench_native_profile_numeric_overrides_become_custom() {
    let profile = clickbench_native_profile_settings_from([
        (NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "conservative"),
        (NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, "123456789"),
        (NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, "268435456"),
        (NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "8"),
    ]);

    assert_eq!(profile.profile_name, CLICKBENCH_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(profile.device_budget_bytes, 123456789);
    assert_eq!(profile.max_source_chunk_bytes, 268435456);
    assert_eq!(profile.max_row_groups_per_chunk, 8);
    assert_eq!(profile.source_chunk_planning_mode, nexus_query_engine::runtime::SourceChunkPlanningMode::Conservative);
}

#[test]
fn clickbench_native_profile_source_chunk_planning_mode_override_becomes_custom() {
    let profile = clickbench_native_profile_settings_from([
        (NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "conservative"),
        (NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput"),
    ]);

    assert_eq!(profile.profile_name, CLICKBENCH_NATIVE_PROFILE_CUSTOM);
    assert_eq!(profile.profile_source, "custom_overrides");
    assert_eq!(
        profile.source_chunk_planning_mode,
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
}

#[test]
fn clickbench_native_profile_exact_overrides_resolve_to_named_profile() {
    let profile = clickbench_native_profile_settings_from([
        (NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "conservative"),
        (NEXUS_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, &CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES.to_string()),
        (
            NEXUS_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
            &CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES.to_string(),
        ),
        (
            NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
            &CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK.to_string(),
        ),
        (NEXUS_CLICKBENCH_NATIVE_SOURCE_CHUNK_PLANNING_MODE_ENV, "local_parquet_throughput"),
    ]);

    assert_eq!(profile.profile_name, CLICKBENCH_NATIVE_PROFILE_GPU_THROUGHPUT);
    assert_eq!(profile.profile_source, "resolved_values");
}

#[test]
fn clickbench_native_profile_rejects_zero_numeric_values() {
    let result = panic::catch_unwind(|| {
        clickbench_native_profile_settings_from([
            (NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "clickbench_gpu_throughput"),
            (NEXUS_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, "0"),
        ]);
    });

    assert!(result.is_err(), "zero row-group caps must be rejected at profile resolution");
}

#[test]
fn clickbench_native_optimizer_config_uses_resolved_profile_knobs() {
    let profile =
        clickbench_native_profile_settings_from([(NEXUS_CLICKBENCH_NATIVE_PROFILE_ENV, "clickbench_gpu_throughput")]);
    let config = clickbench_native_optimizer_config_with_profile_settings(&profile);
    let execution = config.native_execution_config();

    assert_eq!(
        execution.source_chunk_planning_mode(),
        nexus_query_engine::runtime::SourceChunkPlanningMode::LocalParquetThroughput
    );
    match execution.memory_control_policy() {
        nexus_query_engine::runtime::MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes,
            max_row_groups_per_chunk,
            ..
        } => {
            assert_eq!(device_budget_bytes, CLICKBENCH_GPU_THROUGHPUT_DEVICE_BUDGET_BYTES);
            assert_eq!(max_source_chunk_bytes, CLICKBENCH_GPU_THROUGHPUT_MAX_SOURCE_CHUNK_BYTES);
            assert_eq!(max_row_groups_per_chunk, CLICKBENCH_GPU_THROUGHPUT_MAX_ROW_GROUPS_PER_CHUNK);
        },
        other => panic!("expected bounded memory control policy, got {other:?}"),
    }
}
