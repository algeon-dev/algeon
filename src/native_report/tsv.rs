//! TSV escaping and nested-row formatting helpers for native report rows.
//!
//! All formatters here are `pub(super)` and shape the canonical pipe-delimited
//! key/value text emitted into TSV cells. Tabs, newlines, and carriage returns
//! are replaced with spaces before serialization so they cannot break the
//! outer TSV grid; empty nested rows serialize as `none`.

use std::fmt::Write;

use datafusion::physical_plan::metrics::MetricsSet;
use nexus_query_engine::{AdaptivePipelineReportRow, StrictBoundedContractGateRow};

use super::*;
use crate::planner::NativeStrictBoundedCompiledGraphDiagnostics;

pub(crate) trait TsvRow {
    const COLUMNS: &'static [&'static str];
    const SCHEMA_VERSION: u32;

    fn fields(&self) -> Vec<String>;

    fn to_tsv_line(&self) -> String {
        self.fields().join("\t")
    }
}

pub(super) fn format_join_diagnostics(diagnostics: &[NativeJoinDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let _ = write!(
            output,
            "operator={},join_type={},key_count={},key_shape={},residual_predicate={},null_equality={},build_side={},probe_side={}",
            diagnostics.operator,
            diagnostics.join_type,
            diagnostics.key_count,
            diagnostics.key_shape.replace(['\t', '\n', '\r', '|'], " "),
            diagnostics.residual_predicate,
            diagnostics.null_equality,
            diagnostics.build_side,
            diagnostics.probe_side
        );
    }
    output
}

pub(super) fn format_source_diagnostics(diagnostics: &[NativeSourceDiagnosticSummary]) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut output = String::new();
    for (index, diagnostics) in diagnostics.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let rejected_features = if diagnostics.rejected_features.is_empty() {
            "none".to_string()
        } else {
            diagnostics.rejected_features.join(",")
        };
        let known_bytes = diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let selected_row_groups =
            diagnostics.selected_row_groups.map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string());
        let total_row_groups =
            diagnostics.total_row_groups.map_or_else(|| "unknown".to_string(), |row_groups| row_groups.to_string());
        let selected_source_bytes =
            diagnostics.selected_source_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let total_source_bytes =
            diagnostics.total_source_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string());
        let selected_rows = diagnostics.selected_rows.map_or_else(|| "unknown".to_string(), |rows| rows.to_string());
        let total_rows = diagnostics.total_rows.map_or_else(|| "unknown".to_string(), |rows| rows.to_string());
        let _ = write!(
            output,
            "format={},source_read_strategy={},source_decode_input={},credential_mode={},lakehouse_cache_status={},lakehouse_cache_read_mode={},lakehouse_cache_fill_mode={},source_fallback_reason={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},total_row_groups={},selected_source_bytes={},total_source_bytes={},selected_rows={},total_rows={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},scan_limit={},schema_mapping={},rejected_features={}",
            diagnostics.format,
            diagnostics.source_read_strategy,
            diagnostics.source_decode_input,
            diagnostics.credential_mode,
            diagnostics.lakehouse_cache_status,
            diagnostics.lakehouse_cache_read_mode,
            diagnostics.lakehouse_cache_fill_mode,
            diagnostics.source_fallback_reason,
            diagnostics.direct_device_read,
            diagnostics.file_count,
            diagnostics.ranged_file_count,
            known_bytes,
            selected_row_groups,
            total_row_groups,
            selected_source_bytes,
            total_source_bytes,
            selected_rows,
            total_rows,
            diagnostics.read_column_count,
            diagnostics.projected_output_column_count,
            diagnostics.pure_filter_column_count,
            diagnostics.predicate_disposition,
            diagnostics.byte_range_disposition,
            diagnostics.partition_column_disposition,
            diagnostics.zero_column_count_disposition,
            diagnostics.scan_limit_disposition,
            diagnostics.schema_mapping_disposition,
            rejected_features,
        );
    }
    output
}

pub(super) fn format_strict_bounded_contract_rows(rows: &[StrictBoundedContractGateRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.gate_id
            .to_string()
            .cmp(&right.gate_id.to_string())
            .then_with(|| left.reservation_kind.as_str().cmp(right.reservation_kind.as_str()))
            .then_with(|| left.owner.to_string().cmp(&right.owner.to_string()))
    });
    let mut output = String::new();
    for (index, row) in rows.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        output.push_str(&stable_nested_tsv_field(&row.to_stable_kv_row()));
    }
    output
}

pub(super) fn format_strict_bounded_compiled_graphs(graphs: &[NativeStrictBoundedCompiledGraphDiagnostics]) -> String {
    if graphs.is_empty() {
        return "none".to_string();
    }

    let mut graphs = graphs.iter().collect::<Vec<_>>();
    graphs.sort_by_key(|graph| {
        (
            graph.task_count,
            graph.edge_count,
            graph.barrier_count,
            graph.resource_count,
            graph.gate_count,
            graph.sink_kind.clone(),
            graph.sink_gate_count,
        )
    });
    let mut output = String::new();
    for (index, graph) in graphs.iter().enumerate() {
        if index > 0 {
            output.push('|');
        }
        let _ = write!(
            output,
            "task_count={},edge_count={},barrier_count={},resource_count={},gate_count={},sink_kind={},sink_gate_count={}",
            graph.task_count,
            graph.edge_count,
            graph.barrier_count,
            graph.resource_count,
            graph.gate_count,
            stable_nested_tsv_field(&graph.sink_kind),
            graph.sink_gate_count,
        );
    }
    output
}

pub(super) fn format_adaptive_pipeline_rows(rows: &[AdaptivePipelineReportRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| stable_nested_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_dynamic_filter_runtime_rows(rows: &[NativeDynamicFilterRuntimeMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeDynamicFilterRuntimeMetricRow::to_stable_kv_row)
}

pub(super) fn format_lakehouse_cache_fill_event_snapshot_rows(
    rows: &[NativeLakehouseCacheFillEventMetricRow],
) -> String {
    format_stable_nested_metric_rows(rows, NativeLakehouseCacheFillEventMetricRow::to_stable_kv_row)
}

pub(super) fn format_source_read_runtime_rows(rows: &[NativeSourceReadRuntimeMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeSourceReadRuntimeMetricRow::to_stable_kv_row)
}

pub(super) fn format_native_optimizer_diagnostics(
    diagnostics: &nexus_query_engine::NativeOptimizerDiagnostics,
) -> String {
    let rows = diagnostics.stable_rule_rows();
    if rows.is_empty() {
        return "none".to_string();
    }

    stable_tsv_field(&rows.join(";"))
}

pub(super) fn format_join_strategy_rows(rows: &[NativeJoinStrategyMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeJoinStrategyMetricRow::to_stable_kv_row)
}

pub(super) fn format_aggregate_strategy_rows(rows: &[NativeAggregateStrategyMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeAggregateStrategyMetricRow::to_stable_kv_row)
}

pub(super) fn format_graph_runtime_rows(rows: &[NativeGraphRuntimeMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeGraphRuntimeMetricRow::to_stable_kv_row)
}

pub(super) fn format_graph_reservation_rows(rows: &[NativeGraphReservationMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeGraphReservationMetricRow::to_stable_kv_row)
}

pub(super) fn format_task_operator_rows(rows: &[NativeTaskOperatorMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeTaskOperatorMetricRow::to_stable_kv_row)
}

pub(super) fn format_task_runtime_rows(rows: &[NativeTaskRuntimeMetricRow]) -> String {
    format_stable_nested_metric_rows(rows, NativeTaskRuntimeMetricRow::to_stable_kv_row)
}

pub(super) fn write_workload_dataset_metadata(output: &mut String, metadata: Option<&WorkloadDatasetMetadata>) {
    let Some(metadata) = metadata else {
        return;
    };

    let _ = writeln!(output, "# workload={}", metadata.workload);
    let _ = writeln!(output, "# dataset_root={}", metadata.root.display());
    let _ =
        writeln!(output, "# dataset_scale_factor={}", metadata.scale_factor.as_deref().map_or("none", |value| value));
    let _ = writeln!(output, "# dataset_provenance={}", metadata_value(&metadata.provenance));
    let _ = writeln!(output, "# dataset_generation_command={}", metadata_value(&metadata.generation_command));
    let _ = writeln!(output, "# dataset_table_count={}", metadata.table_count);
    let _ = writeln!(output, "# dataset_file_count={}", metadata.file_count);
    let _ = writeln!(output, "# dataset_total_rows={}", metadata.total_rows);
    let _ = writeln!(output, "# dataset_total_bytes={}", metadata.total_bytes);
    let _ = writeln!(output, "# dataset_metadata_fingerprint={}", metadata.metadata_fingerprint);
    let _ = writeln!(
        output,
        "# dataset_content_fingerprint={}",
        metadata.content_fingerprint.as_deref().unwrap_or("not_computed")
    );
    let _ = writeln!(output, "# dataset_table_file_counts={}", metadata.table_file_counts);
    let _ = writeln!(output, "# dataset_table_row_counts={}", metadata.table_row_counts);
    let _ = writeln!(output, "# dataset_table_byte_sizes={}", metadata.table_byte_sizes);
    let _ = writeln!(output, "# dataset_table_row_group_counts={}", metadata.table_row_group_counts);
}

fn metadata_value(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

pub(super) fn stable_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn stable_nested_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r', '|'], " ")
}

fn format_stable_nested_metric_rows<T>(rows: &[T], stable_row: impl Fn(&T) -> String) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().map(stable_row).collect::<Vec<_>>();
    rows.sort();
    rows.into_iter().map(|row| stable_nested_tsv_field(&row)).collect::<Vec<_>>().join("|")
}

pub(super) fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

pub(super) fn optional_observed_metric_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let observed_name = format!("{name}_observed");
    let observed = metrics.sum_by_name(&observed_name).map(|value| value.as_usize()).unwrap_or(0);
    (observed > 0).then(|| metric_count(metrics, name))
}

pub(super) fn metric_duration_nanos(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let nanos = metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0);
    (nanos > 0).then_some(nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_strategy_rows_format_nonzero_filtered_grouped_evidence() {
        let row = NativeAggregateStrategyMetricRow {
            node_id: "n9".to_string(),
            selected_strategy: "filtered_grouped_segmented_state".to_string(),
            execution_stage: "single".to_string(),
            aggregate_filter_count: 4,
            filtered_input_materialization_strategy: "row_chunked_grouped_segmented_state".to_string(),
            filtered_input_lower_level_unresolved_contracts:
                "cudf_dataframe_merge_temporary_workspace_preflight_unavailable".to_string(),
            filtered_grouped_rows_per_chunk: 2048,
            filtered_grouped_inputs_per_segment: 2,
            filtered_grouped_segment_target_temporary_bytes: 4 * 1024 * 1024,
            filtered_grouped_actual_chunk_count: 7,
            filtered_grouped_actual_segment_count: 14,
            filtered_grouped_observed_temporary_peak_bytes: 123_456,
            filtered_grouped_observed_retained_peak_bytes: 78_901,
            filtered_grouped_state_rows_merged: 512,
            filtered_grouped_high_cardinality_admission: "chunk_local_partial_final_merge".to_string(),
            ..NativeAggregateStrategyMetricRow::default()
        };

        let formatted = format_aggregate_strategy_rows(&[row]);

        assert!(formatted.contains("selected_strategy=filtered_grouped_segmented_state"));
        assert!(formatted.contains("filtered_grouped_actual_chunk_count=7"));
        assert!(formatted.contains("filtered_grouped_actual_segment_count=14"));
        assert!(
            formatted.contains(
                "filtered_input_lower_level_unresolved_contracts=cudf_dataframe_merge_temporary_workspace_preflight_unavailable"
            )
        );
    }

    #[test]
    fn aggregate_strategy_rows_format_strategy_execution_options() {
        let row = NativeAggregateStrategyMetricRow {
            aggregate_strategy_options_version: "1".to_string(),
            aggregate_encoded_key_partial_low_cardinality_threshold_ppm: 175_000,
            aggregate_encoded_key_partial_min_input_rows: 123,
            aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm: 200_000,
            aggregate_source_adjacent_prepared_sort_groupby_max_row_groups: 65,
            ..NativeAggregateStrategyMetricRow::default()
        };

        let formatted = format_aggregate_strategy_rows(&[row]);

        assert!(formatted.contains("aggregate_strategy_options_version=1"));
        assert!(formatted.contains("aggregate_encoded_key_partial_low_cardinality_threshold_ppm=175000"));
        assert!(formatted.contains("aggregate_encoded_key_partial_min_input_rows=123"));
        assert!(formatted.contains("aggregate_prepared_sort_groupby_low_cardinality_threshold_ppm=200000"));
        assert!(formatted.contains("aggregate_source_adjacent_prepared_sort_groupby_max_row_groups=65"));
    }

    #[test]
    fn nested_metric_row_formatters_escape_delimiters_sort_rows_and_cover_empty_rows() {
        assert_eq!(format_graph_runtime_rows(&[]), "none");
        assert_eq!(format_task_operator_rows(&[]), "none");
        assert_eq!(format_task_runtime_rows(&[]), "none");
        assert_eq!(format_dynamic_filter_runtime_rows(&[]), "none");
        assert_eq!(format_source_read_runtime_rows(&[]), "none");
        assert_eq!(format_lakehouse_cache_fill_event_snapshot_rows(&[]), "none");
        assert_eq!(format_aggregate_strategy_rows(&[]), "none");
        assert_eq!(format_join_strategy_rows(&[]), "none");

        let formatted = format_task_operator_rows(&[
            NativeTaskOperatorMetricRow {
                operator_kind: "z|op\twith\nbreak".to_string(),
                task_count: 2,
                ..Default::default()
            },
            NativeTaskOperatorMetricRow { operator_kind: "a".to_string(), task_count: 1, ..Default::default() },
        ]);
        assert!(formatted.starts_with("operator_kind=a,"));
        assert_eq!(formatted.matches('|').count(), 1);
        assert!(formatted.contains("operator_kind=z op with break"));

        assert!(
            !format_graph_runtime_rows(&[NativeGraphRuntimeMetricRow {
                node_id: "g|1".to_string(),
                ..Default::default()
            }])
            .contains("g|1")
        );
        assert!(
            !format_task_runtime_rows(&[NativeTaskRuntimeMetricRow {
                compiled_task_id: "t|1".to_string(),
                ..Default::default()
            }])
            .contains("t|1")
        );
        assert!(
            !format_dynamic_filter_runtime_rows(&[NativeDynamicFilterRuntimeMetricRow {
                dynamic_filter_id: "df|1".to_string(),
                ..Default::default()
            }])
            .contains("df|1")
        );
        assert!(
            !format_source_read_runtime_rows(&[NativeSourceReadRuntimeMetricRow {
                source_node_id: "s|1".to_string(),
                ..Default::default()
            }])
            .contains("s|1")
        );
        assert!(
            !format_lakehouse_cache_fill_event_snapshot_rows(&[NativeLakehouseCacheFillEventMetricRow {
                outcome: "filled|late".to_string(),
                ..Default::default()
            }])
            .contains("filled|late")
        );
        assert!(
            !format_aggregate_strategy_rows(&[NativeAggregateStrategyMetricRow {
                node_id: "agg|1".to_string(),
                ..Default::default()
            }])
            .contains("agg|1")
        );
        assert!(
            !format_join_strategy_rows(&[NativeJoinStrategyMetricRow {
                node_id: "join|1".to_string(),
                ..Default::default()
            }])
            .contains("join|1")
        );
    }
}
