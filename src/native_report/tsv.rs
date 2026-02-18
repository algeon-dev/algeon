use std::fmt::Write;

use cudf_query_engine::{AdaptivePipelineReportRow, StrictBoundedContractGateRow};
use datafusion::physical_plan::metrics::MetricsSet;

use super::*;
use crate::planner::NativeStrictBoundedCompiledGraphDiagnostics;

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
        let _ = write!(
            output,
            "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},scan_limit={},schema_mapping={},rejected_features={}",
            diagnostics.format,
            diagnostics.direct_device_read,
            diagnostics.file_count,
            diagnostics.ranged_file_count,
            known_bytes,
            selected_row_groups,
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
        output.push_str(&stable_tsv_field(&row.to_stable_kv_row()));
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
            stable_tsv_field(&graph.sink_kind),
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
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_native_optimizer_diagnostics(
    diagnostics: &cudf_query_engine::NativeOptimizerDiagnostics,
) -> String {
    if diagnostics.is_empty() {
        return "none".to_string();
    }

    let mut rows = Vec::new();
    if !diagnostics.semijoin_reduction.is_empty() {
        let semijoin = &diagnostics.semijoin_reduction;
        let candidates = if semijoin.candidates.is_empty() {
            "none".to_string()
        } else {
            semijoin.candidates.iter().map(format_semijoin_reduction_candidate).collect::<Vec<_>>().join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::SEMIJOIN_REDUCTION_RULE_NAME,
            semijoin.considered_count,
            semijoin.applied_count,
            candidates
        ));
    }
    if !diagnostics.join_build_side_selection.is_empty() {
        let selection = &diagnostics.join_build_side_selection;
        let candidates = if selection.candidates.is_empty() {
            "none".to_string()
        } else {
            selection.candidates.iter().map(format_join_build_side_selection_candidate).collect::<Vec<_>>().join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
            selection.considered_count,
            selection.applied_count,
            candidates
        ));
    }
    stable_tsv_field(&rows.join(";"))
}

fn format_semijoin_reduction_candidate(candidate: &cudf_query_engine::SemijoinReductionCandidateDiagnostics) -> String {
    let after_root_node_id = candidate.after_root_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let crossed_join_node_ids = if candidate.crossed_join_node_ids.is_empty() {
        "none".to_string()
    } else {
        candidate.crossed_join_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+")
    };
    let target_subtree_id = candidate.target_subtree_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let key_mappings = if candidate.key_mappings.is_empty() {
        "none".to_string()
    } else {
        candidate.key_mappings.iter().map(format_semijoin_key_mapping).collect::<Vec<_>>().join(";")
    };
    let cost_gate = candidate.cost_gate.as_ref().map(format_semijoin_cost_gate).unwrap_or_else(|| "none".to_string());
    let admission_decision = candidate
        .admission_decision
        .as_ref()
        .map(|decision| format!("{decision:?}"))
        .unwrap_or_else(|| "none".to_string());
    let original_admitted =
        candidate.original_admitted.map_or_else(|| "not_evaluated".to_string(), |value| value.to_string());
    let rewritten_admitted =
        candidate.rewritten_admitted.map_or_else(|| "not_evaluated".to_string(), |value| value.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());

    format!(
        "rule={},candidate_node_id={},before_root_node_id={},after_root_node_id={},crossed_join_node_ids={},target_subtree_id={},key_mappings={},cost_gate={},admission_decision={},original_admitted={},rewritten_admitted={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.before_root_node_id,
        after_root_node_id,
        crossed_join_node_ids,
        target_subtree_id,
        key_mappings,
        cost_gate,
        admission_decision,
        original_admitted,
        rewritten_admitted,
        candidate.applied,
        rejected_reason,
    )
}

fn format_join_build_side_selection_candidate(
    candidate: &cudf_query_engine::JoinBuildSideSelectionCandidateDiagnostics,
) -> String {
    let after_build_side = candidate.after_build_side.map_or_else(|| "none".to_string(), |side| side.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},before_build_side={},after_build_side={},left_estimated_rows={},right_estimated_rows={},left_estimated_bytes={},right_estimated_bytes={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.before_build_side,
        after_build_side,
        option_u64_text(candidate.left_estimated_rows),
        option_u64_text(candidate.right_estimated_rows),
        option_u64_text(candidate.left_estimated_bytes),
        option_u64_text(candidate.right_estimated_bytes),
        candidate.applied,
        rejected_reason,
    )
}

fn option_u64_text(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn format_semijoin_key_mapping(mapping: &cudf_query_engine::SemijoinReductionKeyMappingDiagnostics) -> String {
    let after_column = mapping.after_column.as_deref().unwrap_or("none");
    let target_subtree_id = mapping.target_subtree_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    format!("{}->{}@{}", stable_tsv_field(&mapping.before_column), stable_tsv_field(after_column), target_subtree_id)
}

fn format_semijoin_cost_gate(gate: &cudf_query_engine::SemijoinReductionCostGateDiagnostics) -> String {
    format!(
        "signal={},threshold_multiplier={},accepted={}",
        format_semijoin_cost_signal(&gate.signal),
        gate.threshold_multiplier,
        gate.accepted,
    )
}

fn format_semijoin_cost_signal(signal: &cudf_query_engine::SemijoinReductionCostSignal) -> String {
    match signal {
        cudf_query_engine::SemijoinReductionCostSignal::RowEstimate { semi_right_rows, target_rows } => {
            format!("row_estimate:semi_right_rows={semi_right_rows}:target_rows={target_rows}")
        },
        cudf_query_engine::SemijoinReductionCostSignal::ByteEstimate { semi_right_bytes, target_bytes } => {
            format!("byte_estimate:semi_right_bytes={semi_right_bytes}:target_bytes={target_bytes}")
        },
        cudf_query_engine::SemijoinReductionCostSignal::AggregateKeysetShape => "aggregate_keyset_shape".to_string(),
        cudf_query_engine::SemijoinReductionCostSignal::Unavailable => "unavailable".to_string(),
    }
}

pub(super) fn format_join_strategy_rows(rows: &[NativeJoinStrategyMetricRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_task_operator_rows(rows: &[NativeTaskOperatorMetricRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_task_runtime_rows(rows: &[NativeTaskRuntimeMetricRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
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

pub(super) fn metric_count(metrics: &MetricsSet, name: &str) -> usize {
    metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0)
}

pub(super) fn optional_observed_metric_count(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let observed_name = format!("{name}_observed");
    let observed = metrics.sum_by_name(&observed_name).map(|value| value.as_usize()).unwrap_or(0);
    (observed > 0).then(|| metric_count(metrics, name))
}
