use std::num::{NonZeroU64, NonZeroUsize};

use super::{
    AlgeonNativeExecutionConfig, AlgeonNativeOptimizerConfig, ConfigSources, Result, SourceChunkPlanningMode,
    SourceChunking, SourceSchedulerPolicy, StatisticalAggregateExecutionMode, file::ServerNativeFileConfig,
};
use crate::server::planner::AlgeonGpuExecutionMode;

pub(super) fn native_optimizer_config(
    file: &ServerNativeFileConfig,
    sources: &ConfigSources,
) -> Result<AlgeonNativeOptimizerConfig> {
    let defaults = SourceChunking::default();
    let mut execution_config = AlgeonNativeExecutionConfig::new(SourceChunking {
        max_source_chunk_bytes: nonzero(
            sources,
            "native.max_source_chunk_bytes",
            file.max_source_chunk_bytes,
            NonZeroU64::new,
        )?
        .unwrap_or(defaults.max_source_chunk_bytes),
        max_row_groups_per_chunk: nonzero(
            sources,
            "native.max_row_groups_per_chunk",
            file.max_row_groups_per_chunk,
            NonZeroUsize::new,
        )?
        .unwrap_or(defaults.max_row_groups_per_chunk),
        max_retry_attempts: nonzero(sources, "native.max_retry_attempts", file.max_retry_attempts, NonZeroUsize::new)?
            .unwrap_or(defaults.max_retry_attempts),
        ..defaults
    });
    if let Some(reads) =
        nonzero(sources, "native.max_source_reads_in_flight", file.max_source_reads_in_flight, NonZeroUsize::new)?
    {
        execution_config = execution_config.with_source_scheduler_policy(SourceSchedulerPolicy::new(reads.get()));
    }
    if let Some(raw) = &file.source_chunk_planning_mode {
        execution_config =
            execution_config.with_source_chunk_planning_mode(parse_source_chunk_planning_mode(sources, raw)?);
    }
    if let Some(raw) = &file.statistical_aggregate_execution_mode {
        execution_config = execution_config
            .with_statistical_aggregate_execution_mode(parse_statistical_aggregate_execution_mode(sources, raw)?);
    }

    let mut optimizer_config = AlgeonNativeOptimizerConfig::default().with_native_execution_config(execution_config);
    if let Some(raw) = &file.execution_mode {
        optimizer_config = optimizer_config.with_execution_mode(parse_execution_mode(sources, raw)?);
    }
    Ok(optimizer_config)
}

/// A configured count that must be greater than zero.
fn nonzero<T, N>(
    sources: &ConfigSources,
    path: &'static str,
    value: Option<T>,
    nonzero: impl FnOnce(T) -> Option<N>,
) -> Result<Option<N>> {
    value.map(|value| nonzero(value).ok_or_else(|| sources.invalid(path, "must be greater than zero"))).transpose()
}

fn parse_source_chunk_planning_mode(sources: &ConfigSources, raw: &str) -> Result<SourceChunkPlanningMode> {
    let raw = raw.trim().to_ascii_lowercase();
    [SourceChunkPlanningMode::Conservative, SourceChunkPlanningMode::LocalParquetThroughput]
        .into_iter()
        .find(|mode| mode.as_str() == raw)
        .ok_or_else(|| {
            sources.invalid(
                "native.source_chunk_planning_mode",
                format!("must be `conservative` or `local_parquet_throughput`, got `{raw}`"),
            )
        })
}

fn parse_statistical_aggregate_execution_mode(
    sources: &ConfigSources,
    raw: &str,
) -> Result<StatisticalAggregateExecutionMode> {
    let raw = raw.trim().to_ascii_lowercase();
    [
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    ]
    .into_iter()
    .find(|mode| mode.as_str() == raw)
    .ok_or_else(|| {
        sources.invalid(
            "native.statistical_aggregate_execution_mode",
            format!("must be `datafusion_exact_host_welford` or `gpu_native_tolerant`, got `{raw}`"),
        )
    })
}

fn parse_execution_mode(sources: &ConfigSources, raw: &str) -> Result<AlgeonGpuExecutionMode> {
    let raw = raw.trim().to_ascii_lowercase();
    [
        AlgeonGpuExecutionMode::FunctionsOnly,
        AlgeonGpuExecutionMode::NativePreferred,
        AlgeonGpuExecutionMode::NativeRequired,
    ]
    .into_iter()
    .find(|mode| mode.as_str() == raw)
    .ok_or_else(|| {
        sources.invalid(
            "native.execution_mode",
            format!("must be `functions_only`, `native_preferred` or `native_required`, got `{raw}`"),
        )
    })
}
