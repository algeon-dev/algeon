//! TSV escaping and nested-row formatting helpers for native report rows.
//!
//! All formatters here are `pub(super)` and shape the canonical pipe-delimited
//! key/value text emitted into TSV cells. Tabs, newlines, and carriage returns
//! are replaced with spaces before serialization so they cannot break the
//! outer TSV grid; empty nested rows serialize as `none`.

#![deny(missing_docs)]

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
            "format={},direct_device_read={},file_count={},ranged_file_count={},known_bytes={},selected_row_groups={},total_row_groups={},selected_source_bytes={},total_source_bytes={},selected_rows={},total_rows={},read_columns={},projected_output_columns={},pure_filter_columns={},predicate={},byte_range={},partition_columns={},zero_column_count={},scan_limit={},schema_mapping={},rejected_features={}",
            diagnostics.format,
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

pub(super) fn format_dynamic_filter_runtime_rows(rows: &[NativeDynamicFilterRuntimeMetricRow]) -> String {
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
    if !diagnostics.projection_simplification.is_empty() {
        let simplification = &diagnostics.projection_simplification;
        let candidates = if simplification.candidates.is_empty() {
            "none".to_string()
        } else {
            simplification
                .candidates
                .iter()
                .map(format_projection_simplification_candidate)
                .collect::<Vec<_>>()
                .join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::PROJECTION_SIMPLIFICATION_RULE_NAME,
            simplification.considered_count,
            simplification.applied_count,
            candidates
        ));
    }
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
    if !diagnostics.outer_join_null_rejection.is_empty() {
        let null_rejection = &diagnostics.outer_join_null_rejection;
        let candidates = if null_rejection.candidates.is_empty() {
            "none".to_string()
        } else {
            null_rejection
                .candidates
                .iter()
                .map(format_outer_join_null_rejection_candidate)
                .collect::<Vec<_>>()
                .join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::OUTER_JOIN_NULL_REJECTION_RULE_NAME,
            null_rejection.considered_count,
            null_rejection.applied_count,
            candidates
        ));
    }
    if !diagnostics.join_filter_reassociation.is_empty() {
        let reassociation = &diagnostics.join_filter_reassociation;
        let candidates = if reassociation.candidates.is_empty() {
            "none".to_string()
        } else {
            reassociation
                .candidates
                .iter()
                .map(format_join_filter_reassociation_candidate)
                .collect::<Vec<_>>()
                .join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::JOIN_FILTER_REASSOCIATION_RULE_NAME,
            reassociation.considered_count,
            reassociation.applied_count,
            candidates
        ));
    }
    if !diagnostics.dimension_derived_row_group_pruning.is_empty() {
        let pruning = &diagnostics.dimension_derived_row_group_pruning;
        let candidates = if pruning.candidates.is_empty() {
            "none".to_string()
        } else {
            pruning
                .candidates
                .iter()
                .map(format_dimension_derived_row_group_pruning_candidate)
                .collect::<Vec<_>>()
                .join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
            pruning.considered_count,
            pruning.applied_count,
            candidates
        ));
    }
    if !diagnostics.stream_fanout_replay.is_empty() {
        let replay = &diagnostics.stream_fanout_replay;
        let candidates = if replay.candidates.is_empty() {
            "none".to_string()
        } else {
            replay.candidates.iter().map(format_stream_fanout_replay_candidate).collect::<Vec<_>>().join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::STREAM_FANOUT_REPLAY_RULE_NAME,
            replay.considered_count,
            replay.applied_count,
            candidates
        ));
    }
    if !diagnostics.local_parquet_source_reuse.is_empty() {
        let reuse = &diagnostics.local_parquet_source_reuse;
        let candidates = if reuse.candidates.is_empty() {
            "none".to_string()
        } else {
            reuse.candidates.iter().map(format_local_parquet_source_reuse_candidate).collect::<Vec<_>>().join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},applied_count={},candidates={}",
            cudf_query_engine::LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
            reuse.considered_count,
            reuse.applied_count,
            candidates
        ));
    }
    if !diagnostics.shared_subplan_materialization.is_empty() {
        let materialization = &diagnostics.shared_subplan_materialization;
        let candidates = if materialization.candidates.is_empty() {
            "none".to_string()
        } else {
            materialization
                .candidates
                .iter()
                .map(format_shared_subplan_materialization_candidate)
                .collect::<Vec<_>>()
                .join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},candidate_count={},applied_count={},candidates={}",
            cudf_query_engine::SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
            materialization.considered_count,
            materialization.candidate_count,
            materialization.applied_count,
            candidates
        ));
    }
    if !diagnostics.dynamic_filter_route_discovery.is_empty() {
        let discovery = &diagnostics.dynamic_filter_route_discovery;
        let candidates = if discovery.candidates.is_empty() {
            "none".to_string()
        } else {
            discovery.candidates.iter().map(format_dynamic_filter_route_candidate).collect::<Vec<_>>().join("|")
        };
        rows.push(format!(
            "rule={},considered_count={},route_count={},candidates={}",
            cudf_query_engine::DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
            discovery.considered_count,
            discovery.route_count,
            candidates
        ));
    }
    stable_tsv_field(&rows.join(";"))
}

fn format_projection_simplification_candidate(
    candidate: &cudf_query_engine::ProjectionSimplificationCandidateDiagnostics,
) -> String {
    let replacement_node_id = candidate.replacement_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},replacement_node_id={},applied={},rejected_reason={}",
        candidate.rule_name, candidate.candidate_node_id, replacement_node_id, candidate.applied, rejected_reason,
    )
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

fn format_dimension_derived_row_group_pruning_candidate(
    candidate: &cudf_query_engine::DimensionDerivedRowGroupPruningCandidateDiagnostics,
) -> String {
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},dimension_source_node_id={},target_source_node_id={},dimension_key_column={},target_key_column={},derived_min_key={},derived_max_key={},selected_row_groups_before={},selected_row_groups_after={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.dimension_source_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.target_source_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.dimension_key_column.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
        candidate.target_key_column.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
        option_i64_text(candidate.derived_min_key),
        option_i64_text(candidate.derived_max_key),
        option_usize_text(candidate.selected_row_groups_before),
        option_usize_text(candidate.selected_row_groups_after),
        candidate.applied,
        rejected_reason,
    )
}

fn format_stream_fanout_replay_candidate(
    candidate: &cudf_query_engine::StreamFanoutReplayCandidateDiagnostics,
) -> String {
    let join_probe_consumer_node_ids = if candidate.join_probe_consumer_node_ids.is_empty() {
        "none".to_string()
    } else {
        candidate.join_probe_consumer_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+")
    };
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},fanout_node_id={},consumer_count={},join_probe_consumer_node_ids={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.fanout_node_id,
        candidate.consumer_count,
        join_probe_consumer_node_ids,
        candidate.applied,
        rejected_reason,
    )
}

fn format_local_parquet_source_reuse_candidate(
    candidate: &cudf_query_engine::LocalParquetSourceReuseCandidateDiagnostics,
) -> String {
    let source_node_ids = candidate.source_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+");
    let stable_file_ids = if candidate.stable_file_ids.is_empty() {
        "none".to_string()
    } else {
        candidate.stable_file_ids.iter().map(|value| stable_tsv_field(value)).collect::<Vec<_>>().join("+")
    };
    let superset_read_columns = if candidate.superset_read_columns.is_empty() {
        "none".to_string()
    } else {
        candidate.superset_read_columns.iter().map(|value| stable_tsv_field(value)).collect::<Vec<_>>().join("+")
    };
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},source_node_ids={},reuse_count={},stable_file_ids={},superset_read_columns={},branch_filter_count={},applied={},rejected_reason={}",
        candidate.rule_name,
        source_node_ids,
        candidate.reuse_count,
        stable_file_ids,
        superset_read_columns,
        candidate.branch_filter_count,
        candidate.applied,
        rejected_reason,
    )
}

fn format_shared_subplan_materialization_candidate(
    candidate: &cudf_query_engine::SharedSubplanMaterializationCandidateDiagnostics,
) -> String {
    let aggregate_node_ids = candidate.aggregate_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+");
    let group_exprs = if candidate.group_exprs.is_empty() {
        "none".to_string()
    } else {
        candidate.group_exprs.iter().map(|expr| stable_tsv_field(expr)).collect::<Vec<_>>().join("+")
    };
    let aggregate_exprs = if candidate.aggregate_exprs.is_empty() {
        "none".to_string()
    } else {
        candidate.aggregate_exprs.iter().map(|expr| stable_tsv_field(expr)).collect::<Vec<_>>().join("+")
    };
    let source_fingerprints = if candidate.source_fingerprints.is_empty() {
        "none".to_string()
    } else {
        candidate
            .source_fingerprints
            .iter()
            .map(|fingerprint| stable_tsv_field(fingerprint))
            .collect::<Vec<_>>()
            .join("+")
    };
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},family_key={},aggregate_node_ids={},reuse_count={},group_exprs={},aggregate_exprs={},source_fingerprints={},exact_source_variant_count={},repeated_selected_source_bytes={},duplicate_selected_source_bytes={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.family_key,
        aggregate_node_ids,
        candidate.reuse_count,
        group_exprs,
        aggregate_exprs,
        source_fingerprints,
        candidate.exact_source_variant_count,
        option_u64_text(candidate.repeated_selected_source_bytes),
        option_u64_text(candidate.duplicate_selected_source_bytes),
        candidate.applied,
        rejected_reason,
    )
}

fn format_dynamic_filter_route_candidate(
    candidate: &cudf_query_engine::DynamicFilterRouteCandidateDiagnostics,
) -> String {
    let dynamic_filter_id = candidate.dynamic_filter_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let producer_build_side = candidate.producer_build_side.map_or_else(|| "none".to_string(), |side| side.to_string());
    let producer_input_node_id =
        candidate.producer_input_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let consumer_input_node_id =
        candidate.consumer_input_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let consumer_kind = candidate.consumer_kind.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string());
    let consumer_node_id = candidate.consumer_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let consumer_source_node_id =
        candidate.consumer_source_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let filter_kinds = if candidate.filter_kinds.is_empty() {
        "none".to_string()
    } else {
        candidate.filter_kinds.iter().map(ToString::to_string).collect::<Vec<_>>().join("+")
    };
    let consumer_lowering =
        candidate.consumer_lowering.map_or_else(|| "none".to_string(), |lowering| lowering.to_string());
    let consumer_reduction_stage =
        candidate.consumer_reduction_stage.map_or_else(|| "none".to_string(), |stage| stage.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},dynamic_filter_id={},producer_build_side={},producer_input_node_id={},consumer_input_node_id={},consumer_kind={},consumer_node_id={},consumer_source_node_id={},producer_key_column={},consumer_key_column={},filter_kinds={},consumer_lowering={},consumer_reduction_stage={},selected_source_bytes={},total_source_bytes={},routed={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        dynamic_filter_id,
        producer_build_side,
        producer_input_node_id,
        consumer_input_node_id,
        consumer_kind,
        consumer_node_id,
        consumer_source_node_id,
        candidate.producer_key_column.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
        candidate.consumer_key_column.as_deref().map(stable_tsv_field).unwrap_or_else(|| "none".to_string()),
        filter_kinds,
        consumer_lowering,
        consumer_reduction_stage,
        option_u64_text(candidate.selected_source_bytes),
        option_u64_text(candidate.total_source_bytes),
        candidate.routed,
        rejected_reason,
    )
}

fn format_outer_join_null_rejection_candidate(
    candidate: &cudf_query_engine::OuterJoinNullRejectionCandidateDiagnostics,
) -> String {
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    let rejecting_columns = if candidate.rejecting_columns.is_empty() {
        "none".to_string()
    } else {
        candidate.rejecting_columns.iter().map(|column| stable_tsv_field(column)).collect::<Vec<_>>().join("+")
    };
    format!(
        "rule={},candidate_node_id={},rejecting_node_id={},rejecting_join_side={},rejecting_columns={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.rejecting_node_id,
        candidate.rejecting_join_side,
        rejecting_columns,
        candidate.applied,
        rejected_reason,
    )
}

fn format_join_filter_reassociation_candidate(
    candidate: &cudf_query_engine::JoinFilterReassociationCandidateDiagnostics,
) -> String {
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},crossed_join_node_id={},filtered_child_node_id={},filtering_node_id={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.crossed_join_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.filtered_child_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.filtering_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.applied,
        rejected_reason,
    )
}

fn option_u64_text(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn option_i64_text(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn option_usize_text(value: Option<usize>) -> String {
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

pub(super) fn format_aggregate_strategy_rows(rows: &[NativeAggregateStrategyMetricRow]) -> String {
    if rows.is_empty() {
        return "none".to_string();
    }

    let mut rows = rows.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| row.to_stable_kv_row());
    rows.iter().map(|row| stable_tsv_field(&row.to_stable_kv_row())).collect::<Vec<_>>().join("|")
}

pub(super) fn format_graph_runtime_rows(rows: &[NativeGraphRuntimeMetricRow]) -> String {
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

pub(super) fn metric_duration_nanos(metrics: &MetricsSet, name: &str) -> Option<usize> {
    let nanos = metrics.sum_by_name(name).map(|value| value.as_usize()).unwrap_or(0);
    (nanos > 0).then_some(nanos)
}
