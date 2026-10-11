use std::{collections::BTreeMap, fmt::Write as _, time::Duration};

use algeon_datafusion::native_report::{
    format_native_aggregate_strategy_metric_rows, format_native_dynamic_filter_runtime_metric_rows,
    format_native_graph_runtime_metric_rows, format_native_join_strategy_metric_rows,
    format_native_task_operator_metric_rows, format_native_task_runtime_metric_rows,
};

use super::{
    CapabilityDecisionRow, MetricsSet, NativeAggregateStrategyMetricRow, NativeCapabilityCompiledGraphDiagnostics,
    NativeDynamicFilterRuntimeMetricRow, NativeGraphRuntimeMetricRow, NativeJoinStrategyMetricRow,
    NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow, NativeTpchMemoryPolicy, TpchDatasetMetadata,
};
use crate::report_workload::RelationalExecutionSelection;

pub(super) fn write_native_execution_policy_metadata(
    output: &mut String,
    relational_execution: RelationalExecutionSelection,
) {
    push_line(output, format_args!("# tpch_benchmark_relational_execution={}", relational_execution.as_str()));
}

/// `device_budget_bytes` keeps its historical meaning of the managed capacity
/// the run executed under: the resolved device capacity once a backend was
/// installed, otherwise the configured budget (`automatic` before resolution).
pub(super) fn write_native_memory_policy_metadata(
    output: &mut String,
    memory_policy: NativeTpchMemoryPolicy,
    max_source_reads_in_flight: usize,
    resolved_device_capacity_bytes: Option<u64>,
) {
    let NativeTpchMemoryPolicy {
        profile_name,
        profile_source,
        device_budget,
        max_source_chunk_bytes,
        max_row_groups_per_chunk,
        max_retry_attempts,
        min_retry_chunk_row_groups,
        source_chunk_planning_mode,
    } = memory_policy;
    push_line(output, format_args!("# tpch_benchmark_native_profile_name={profile_name}"));
    push_line(output, format_args!("# tpch_benchmark_native_profile_source={profile_source}"));
    push_line(output, format_args!("# tpch_benchmark_native_device_budget_policy={}", device_budget.as_policy_str()));
    let device_budget_bytes =
        resolved_device_capacity_bytes.map_or_else(|| device_budget.to_string(), |bytes| bytes.to_string());
    push_line(output, format_args!("# tpch_benchmark_native_device_budget_bytes={device_budget_bytes}"));
    push_line(output, format_args!("# tpch_benchmark_native_max_source_chunk_bytes={max_source_chunk_bytes}"));
    push_line(output, format_args!("# tpch_benchmark_native_max_row_groups_per_chunk={max_row_groups_per_chunk}"));
    push_line(output, format_args!("# tpch_benchmark_native_max_retry_attempts={max_retry_attempts}"));
    push_line(output, format_args!("# tpch_benchmark_native_min_retry_chunk_row_groups={min_retry_chunk_row_groups}"));
    push_line(
        output,
        format_args!("# tpch_benchmark_native_source_chunk_planning_mode={}", source_chunk_planning_mode.as_str()),
    );
    push_line(output, format_args!("# tpch_benchmark_native_max_source_reads_in_flight={max_source_reads_in_flight}"));
    push_line(output, format_args!("# tpch_benchmark_native_runtime_policy=shared_per_benchmark_run"));
}

pub(super) fn write_tpch_dataset_metadata(output: &mut String, metadata: &TpchDatasetMetadata) {
    push_line(output, format_args!("# tpch_dataset_root={}", metadata.root.display()));
    push_line(output, format_args!("# tpch_dataset_scale_factor={}", metadata_value(&metadata.scale_factor)));
    push_line(output, format_args!("# tpch_dataset_provenance={}", metadata_value(&metadata.provenance)));
    push_line(
        output,
        format_args!("# tpch_dataset_generation_command={}", metadata_value(&metadata.generation_command)),
    );
    push_line(output, format_args!("# tpch_dataset_table_count={}", metadata.table_count));
    push_line(output, format_args!("# tpch_dataset_file_count={}", metadata.file_count));
    push_line(output, format_args!("# tpch_dataset_total_rows={}", metadata.total_rows));
    push_line(output, format_args!("# tpch_dataset_total_bytes={}", metadata.total_bytes));
    push_line(output, format_args!("# tpch_dataset_metadata_fingerprint={}", metadata.metadata_fingerprint));
    push_line(
        output,
        format_args!(
            "# tpch_dataset_content_fingerprint={}",
            metadata.content_fingerprint.as_deref().unwrap_or("not_computed")
        ),
    );
    push_line(output, format_args!("# tpch_dataset_table_file_counts={}", metadata.table_file_counts()));
    push_line(output, format_args!("# tpch_dataset_table_row_counts={}", metadata.table_row_counts()));
    push_line(output, format_args!("# tpch_dataset_table_byte_sizes={}", metadata.table_byte_sizes()));
    push_line(output, format_args!("# tpch_dataset_table_row_group_counts={}", metadata.table_row_group_counts()));
}

pub(super) fn format_capability_decision_rows(rows: &[CapabilityDecisionRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.decision_id
            .to_string()
            .cmp(&right.decision_id.to_string())
            .then_with(|| left.boundary.cmp(right.boundary))
            .then_with(|| left.owner.to_string().cmp(&right.owner.to_string()))
    });
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_capability_compiled_graphs(graphs: &[NativeCapabilityCompiledGraphDiagnostics]) -> String {
    if graphs.is_empty() {
        return "none".to_string();
    }

    let mut graphs = graphs.iter().collect::<Vec<_>>();
    graphs.sort_by_key(|graph| {
        (graph.task_count, graph.edge_count, graph.barrier_count, graph.resource_count, graph.sink_kind.clone())
    });
    graphs
        .iter()
        .map(|graph| {
            format!(
                "task_count={},edge_count={},barrier_count={},resource_count={},sink_kind={}",
                graph.task_count,
                graph.edge_count,
                graph.barrier_count,
                graph.resource_count,
                stable_tsv_field(&graph.sink_kind),
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

pub(super) fn format_join_strategy_rows(rows: &[NativeJoinStrategyMetricRow]) -> String {
    format_native_join_strategy_metric_rows(rows)
}

pub(super) fn format_aggregate_strategy_rows(rows: &[NativeAggregateStrategyMetricRow]) -> String {
    format_native_aggregate_strategy_metric_rows(rows)
}

pub(super) fn format_graph_runtime_rows(rows: &[NativeGraphRuntimeMetricRow]) -> String {
    format_native_graph_runtime_metric_rows(rows)
}

pub(super) fn format_task_operator_rows(rows: &[NativeTaskOperatorMetricRow]) -> String {
    format_native_task_operator_metric_rows(rows)
}

pub(super) fn format_task_runtime_rows(rows: &[NativeTaskRuntimeMetricRow]) -> String {
    format_native_task_runtime_metric_rows(rows)
}

pub(super) fn format_dynamic_filter_runtime_rows(rows: &[NativeDynamicFilterRuntimeMetricRow]) -> String {
    format_native_dynamic_filter_runtime_metric_rows(rows)
}

pub(super) fn format_failure_facts(facts: &BTreeMap<String, String>) -> String {
    format_key_value_facts(facts)
}

fn format_key_value_facts(facts: &BTreeMap<String, String>) -> String {
    if facts.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, (key, value)) in facts.iter().enumerate() {
        if index > 0 {
            output.push(';');
        }
        output.push_str(&stable_tsv_field(key));
        output.push('=');
        output.push_str(&stable_tsv_field(value));
    }
    output
}

pub(super) fn metric_duration_ms(metrics: &MetricsSet, name: &str) -> Option<f64> {
    metrics
        .sum_by_name(name)
        .map(|value| duration_ms(Duration::from_nanos(value.as_usize().min(u64::MAX as usize) as u64)))
}

pub(super) fn mean_duration(samples: &[Duration]) -> Duration {
    let total_nanos = samples.iter().map(Duration::as_nanos).sum::<u128>();
    Duration::from_nanos((total_nanos / samples.len().max(1) as u128).min(u128::from(u64::MAX)) as u64)
}

pub(super) fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub(super) fn optional_f64_text(value: Option<f64>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| format!("{value:.3}"))
}

pub(super) fn optional_usize_text(value: Option<usize>) -> String {
    value.map_or_else(|| "not_run".to_string(), |value| value.to_string())
}

pub(super) fn optional_usize_text_none(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

pub(super) fn optional_text(value: Option<&str>, missing: &str) -> String {
    value.map_or_else(|| missing.to_string(), stable_tsv_field)
}

pub(super) fn mean_optional_f64(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0_u64;
    for value in values.into_iter().flatten() {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn metadata_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn push_line(output: &mut String, args: std::fmt::Arguments<'_>) {
    let _ = writeln!(output, "{args}");
}
