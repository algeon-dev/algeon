use super::*;

impl NativeOptimizerDiagnostics {
    /// Projects native optimizer rule diagnostics into stable key/value rows.
    ///
    /// The DataFusion adapter owns only the outer report escaping and joining.
    /// Rule-specific diagnostic shape lives here with the native optimizer
    /// contracts so adding a native rule cannot silently drop adapter evidence.
    pub fn stable_rule_rows(&self) -> Vec<String> {
        let mut rows = Vec::new();

        if !self.projection_simplification.is_empty() {
            let simplification = &self.projection_simplification;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                PROJECTION_SIMPLIFICATION_RULE_NAME,
                simplification.considered_count,
                simplification.applied_count,
                format_candidates(&simplification.candidates, format_projection_simplification_candidate)
            ));
        }
        if !self.semijoin_reduction.is_empty() {
            let semijoin = &self.semijoin_reduction;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                SEMIJOIN_REDUCTION_RULE_NAME,
                semijoin.considered_count,
                semijoin.applied_count,
                format_candidates(&semijoin.candidates, format_semijoin_reduction_candidate)
            ));
        }
        if !self.join_build_side_selection.is_empty() {
            let selection = &self.join_build_side_selection;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
                selection.considered_count,
                selection.applied_count,
                format_candidates(&selection.candidates, format_join_build_side_selection_candidate)
            ));
        }
        if !self.outer_join_null_rejection.is_empty() {
            let null_rejection = &self.outer_join_null_rejection;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                OUTER_JOIN_NULL_REJECTION_RULE_NAME,
                null_rejection.considered_count,
                null_rejection.applied_count,
                format_candidates(&null_rejection.candidates, format_outer_join_null_rejection_candidate)
            ));
        }
        if !self.join_filter_reassociation.is_empty() {
            let reassociation = &self.join_filter_reassociation;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                JOIN_FILTER_REASSOCIATION_RULE_NAME,
                reassociation.considered_count,
                reassociation.applied_count,
                format_candidates(&reassociation.candidates, format_join_filter_reassociation_candidate)
            ));
        }
        if !self.dimension_derived_row_group_pruning.is_empty() {
            let pruning = &self.dimension_derived_row_group_pruning;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
                pruning.considered_count,
                pruning.applied_count,
                format_candidates(&pruning.candidates, format_dimension_derived_row_group_pruning_candidate)
            ));
        }
        if !self.stream_fanout_replay.is_empty() {
            let replay = &self.stream_fanout_replay;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                STREAM_FANOUT_REPLAY_RULE_NAME,
                replay.considered_count,
                replay.applied_count,
                format_candidates(&replay.candidates, format_stream_fanout_replay_candidate)
            ));
        }
        if !self.local_parquet_source_reuse.is_empty() {
            let reuse = &self.local_parquet_source_reuse;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME,
                reuse.considered_count,
                reuse.applied_count,
                format_candidates(&reuse.candidates, format_local_parquet_source_reuse_candidate)
            ));
        }
        if !self.shared_subplan_materialization.is_empty() {
            let materialization = &self.shared_subplan_materialization;
            rows.push(format!(
                "rule={},considered_count={},candidate_count={},applied_count={},candidates={}",
                SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
                materialization.considered_count,
                materialization.candidate_count,
                materialization.applied_count,
                format_candidates(&materialization.candidates, format_shared_subplan_materialization_candidate)
            ));
        }
        if !self.dynamic_filter_route_discovery.is_empty() {
            let discovery = &self.dynamic_filter_route_discovery;
            rows.push(format!(
                "rule={},considered_count={},route_count={},candidates={}",
                DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
                discovery.considered_count,
                discovery.route_count,
                format_candidates(&discovery.candidates, format_dynamic_filter_route_candidate)
            ));
        }
        if !self.aggregate_join_pushdown.is_empty() {
            let pushdown = &self.aggregate_join_pushdown;
            rows.push(format!(
                "rule={},considered_count={},applied_count={},candidates={}",
                AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                pushdown.considered_count,
                pushdown.applied_count,
                format_candidates(&pushdown.candidates, format_aggregate_join_pushdown_candidate)
            ));
        }

        rows
    }
}

fn format_candidates<T>(candidates: &[T], format: fn(&T) -> String) -> String {
    if candidates.is_empty() { "none".to_string() } else { candidates.iter().map(format).collect::<Vec<_>>().join("|") }
}

fn format_projection_simplification_candidate(candidate: &ProjectionSimplificationCandidateDiagnostics) -> String {
    let replacement_node_id = candidate.replacement_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},replacement_node_id={},applied={},rejected_reason={}",
        candidate.rule_name, candidate.candidate_node_id, replacement_node_id, candidate.applied, rejected_reason,
    )
}

fn format_semijoin_reduction_candidate(candidate: &SemijoinReductionCandidateDiagnostics) -> String {
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

fn format_join_build_side_selection_candidate(candidate: &JoinBuildSideSelectionCandidateDiagnostics) -> String {
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
    candidate: &DimensionDerivedRowGroupPruningCandidateDiagnostics,
) -> String {
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},candidate_node_id={},dimension_source_node_id={},target_source_node_id={},dimension_key_column={},target_key_column={},derived_min_key={},derived_max_key={},selected_row_groups_before={},selected_row_groups_after={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.candidate_node_id,
        candidate.dimension_source_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.target_source_node_id.map_or_else(|| "none".to_string(), |id| id.to_string()),
        candidate.dimension_key_column.as_deref().map(stable_diagnostic_field).unwrap_or_else(|| "none".to_string()),
        candidate.target_key_column.as_deref().map(stable_diagnostic_field).unwrap_or_else(|| "none".to_string()),
        option_i64_text(candidate.derived_min_key),
        option_i64_text(candidate.derived_max_key),
        option_usize_text(candidate.selected_row_groups_before),
        option_usize_text(candidate.selected_row_groups_after),
        candidate.applied,
        rejected_reason,
    )
}

fn format_stream_fanout_replay_candidate(candidate: &StreamFanoutReplayCandidateDiagnostics) -> String {
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

fn format_local_parquet_source_reuse_candidate(candidate: &LocalParquetSourceReuseCandidateDiagnostics) -> String {
    let source_node_ids = candidate.source_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+");
    let stable_file_ids = if candidate.stable_file_ids.is_empty() {
        "none".to_string()
    } else {
        candidate.stable_file_ids.iter().map(|value| stable_diagnostic_field(value)).collect::<Vec<_>>().join("+")
    };
    let superset_read_columns = if candidate.superset_read_columns.is_empty() {
        "none".to_string()
    } else {
        candidate.superset_read_columns.iter().map(|value| stable_diagnostic_field(value)).collect::<Vec<_>>().join("+")
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
    candidate: &SharedSubplanMaterializationCandidateDiagnostics,
) -> String {
    let aggregate_node_ids = candidate.aggregate_node_ids.iter().map(ToString::to_string).collect::<Vec<_>>().join("+");
    let group_exprs = if candidate.group_exprs.is_empty() {
        "none".to_string()
    } else {
        candidate.group_exprs.iter().map(|expr| stable_diagnostic_field(expr)).collect::<Vec<_>>().join("+")
    };
    let aggregate_exprs = if candidate.aggregate_exprs.is_empty() {
        "none".to_string()
    } else {
        candidate.aggregate_exprs.iter().map(|expr| stable_diagnostic_field(expr)).collect::<Vec<_>>().join("+")
    };
    let source_fingerprints = if candidate.source_fingerprints.is_empty() {
        "none".to_string()
    } else {
        candidate
            .source_fingerprints
            .iter()
            .map(|fingerprint| stable_diagnostic_field(fingerprint))
            .collect::<Vec<_>>()
            .join("+")
    };
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},family_key={},aggregate_node_ids={},reuse_count={},group_exprs={},aggregate_exprs={},filtered_aggregate_count={},filtered_aggregate_memory_shape={},filtered_aggregate_estimation_input_rows={},filtered_aggregate_estimated_materialization_bytes={},filtered_aggregate_memory_budget_bytes={},filtered_aggregate_memory_threshold_bytes={},filtered_aggregate_memory_threshold_policy={},source_fingerprints={},exact_source_variant_count={},repeated_selected_source_bytes={},duplicate_selected_source_bytes={},applied={},rejected_reason={}",
        candidate.rule_name,
        candidate.family_key,
        aggregate_node_ids,
        candidate.reuse_count,
        group_exprs,
        aggregate_exprs,
        candidate.filtered_aggregate_count,
        candidate.filtered_aggregate_memory_shape,
        option_u64_text(candidate.filtered_aggregate_estimation_input_rows),
        option_u64_text(candidate.filtered_aggregate_estimated_materialization_bytes),
        option_u64_text(candidate.filtered_aggregate_memory_budget_bytes),
        option_u64_text(candidate.filtered_aggregate_memory_threshold_bytes),
        candidate.filtered_aggregate_memory_threshold_policy,
        source_fingerprints,
        candidate.exact_source_variant_count,
        option_u64_text(candidate.repeated_selected_source_bytes),
        option_u64_text(candidate.duplicate_selected_source_bytes),
        candidate.applied,
        rejected_reason,
    )
}

fn format_dynamic_filter_route_candidate(candidate: &DynamicFilterRouteCandidateDiagnostics) -> String {
    let dynamic_filter_id = candidate.dynamic_filter_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let producer_build_side = candidate.producer_build_side.map_or_else(|| "none".to_string(), |side| side.to_string());
    let producer_input_node_id =
        candidate.producer_input_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let consumer_input_node_id =
        candidate.consumer_input_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let consumer_kind =
        candidate.consumer_kind.as_deref().map(stable_diagnostic_field).unwrap_or_else(|| "none".to_string());
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
        candidate.producer_key_column.as_deref().map(stable_diagnostic_field).unwrap_or_else(|| "none".to_string()),
        candidate.consumer_key_column.as_deref().map(stable_diagnostic_field).unwrap_or_else(|| "none".to_string()),
        filter_kinds,
        consumer_lowering,
        consumer_reduction_stage,
        option_u64_text(candidate.selected_source_bytes),
        option_u64_text(candidate.total_source_bytes),
        candidate.routed,
        rejected_reason,
    )
}

fn format_outer_join_null_rejection_candidate(candidate: &OuterJoinNullRejectionCandidateDiagnostics) -> String {
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    let rejecting_columns = if candidate.rejecting_columns.is_empty() {
        "none".to_string()
    } else {
        candidate.rejecting_columns.iter().map(|column| stable_diagnostic_field(column)).collect::<Vec<_>>().join("+")
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

fn format_join_filter_reassociation_candidate(candidate: &JoinFilterReassociationCandidateDiagnostics) -> String {
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

fn format_aggregate_join_pushdown_candidate(candidate: &AggregateJoinPushdownCandidateDiagnostics) -> String {
    let join_node_id = candidate.join_node_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    let rejected_reason =
        candidate.rejected_reason.as_ref().map(|reason| format!("{reason:?}")).unwrap_or_else(|| "none".to_string());
    format!(
        "rule={},aggregate_node_id={},join_node_id={},applied={},rejected_reason={}",
        candidate.rule_name, candidate.aggregate_node_id, join_node_id, candidate.applied, rejected_reason,
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

fn format_semijoin_key_mapping(mapping: &SemijoinReductionKeyMappingDiagnostics) -> String {
    let after_column = mapping.after_column.as_deref().unwrap_or("none");
    let target_subtree_id = mapping.target_subtree_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    format!(
        "{}->{}@{}",
        stable_diagnostic_field(&mapping.before_column),
        stable_diagnostic_field(after_column),
        target_subtree_id
    )
}

fn format_semijoin_cost_gate(gate: &SemijoinReductionCostGateDiagnostics) -> String {
    format!(
        "signal={},threshold_multiplier={},accepted={}",
        format_semijoin_cost_signal(&gate.signal),
        gate.threshold_multiplier,
        gate.accepted,
    )
}

fn format_semijoin_cost_signal(signal: &SemijoinReductionCostSignal) -> String {
    match signal {
        SemijoinReductionCostSignal::RowEstimate { semi_right_rows, target_rows } => {
            format!("row_estimate:semi_right_rows={semi_right_rows}:target_rows={target_rows}")
        },
        SemijoinReductionCostSignal::ByteEstimate { semi_right_bytes, target_bytes } => {
            format!("byte_estimate:semi_right_bytes={semi_right_bytes}:target_bytes={target_bytes}")
        },
        SemijoinReductionCostSignal::AggregateKeysetShape => "aggregate_keyset_shape".to_string(),
        SemijoinReductionCostSignal::FilteredSourceKeysetShape {
            filtered_source_count,
            semi_right_source_bytes,
            target_rows,
            target_bytes,
            crossed_join_count,
        } => format!(
            "filtered_source_keyset_shape:filtered_source_count={filtered_source_count}:semi_right_source_bytes={}:target_rows={}:target_bytes={}:crossed_join_count={crossed_join_count}",
            optional_semijoin_u64(*semi_right_source_bytes),
            optional_semijoin_u64(*target_rows),
            optional_semijoin_u64(*target_bytes),
        ),
        SemijoinReductionCostSignal::Unavailable => "unavailable".to_string(),
    }
}

fn optional_semijoin_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

fn stable_diagnostic_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
