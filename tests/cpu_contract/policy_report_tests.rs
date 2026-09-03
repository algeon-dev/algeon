use datafusion_nexus::{
    capability::CapabilityReason,
    report::{CandidateOutcome, CandidateOutcomeKind, GpuCandidate, PlanningReport},
};
use nexus_query_engine::{
    plan::NativePlanCostReason,
    triage::{TriageCode, TriageVerdict},
};

#[test]
fn capability_reason_codes_are_stable_manifest_values() {
    let codes = [
        (CapabilityReason::UnsupportedOperator, "unsupported_operator"),
        (CapabilityReason::UnsupportedExpression, "unsupported_expression"),
        (CapabilityReason::UnsupportedType, "unsupported_type"),
        (CapabilityReason::NativeSortPreservingMergeUnsupported, "native_sort_preserving_merge_unsupported"),
        (CapabilityReason::NativeDistributedAggregateUnsupported, "native_distributed_aggregate_unsupported"),
        (CapabilityReason::NativePartitionPreservingSortUnsupported, "native_partition_preserving_sort_unsupported"),
        (CapabilityReason::NativeBoundedJoinUnsupported, "native_bounded_join_unsupported"),
        (CapabilityReason::NativeOrderedOutputUnsupported, "native_ordered_output_unsupported"),
        (CapabilityReason::NativeSourceNonLocalObjectStore, "native_source_non_local_object_store"),
        (CapabilityReason::NativeSourceExprAdapter, "native_source_expr_adapter"),
        (CapabilityReason::NativeSourceParquetFileReaderFactory, "native_source_parquet_file_reader_factory"),
        (CapabilityReason::NativeSourcePreserveOrder, "native_source_preserve_order"),
        (CapabilityReason::NativeSourceOutputOrdering, "native_source_output_ordering"),
        (CapabilityReason::NativeSourcePushedDownParquetPredicate, "native_source_pushed_down_parquet_predicate"),
        (CapabilityReason::NativeSourceEmpty, "native_source_empty"),
        (CapabilityReason::NativeSourcePartitionValues, "native_source_partition_values"),
        (CapabilityReason::NativeSourceByteRange, "native_source_byte_range"),
        (CapabilityReason::NativeSourceRowSelection, "native_source_row_selection"),
        (CapabilityReason::NativeSourceSchemaMapping, "native_source_schema_mapping"),
        (CapabilityReason::TableFormatUnsupported, "table_format_unsupported"),
        (CapabilityReason::TableFormatPlanningFailed, "table_format_planning_failed"),
        (CapabilityReason::IcebergNonParquetDataFile, "iceberg_non_parquet_data_file"),
        (CapabilityReason::IcebergDeleteFilesUnsupported, "iceberg_delete_files_unsupported"),
        (CapabilityReason::IcebergFieldIdMappingUnsupported, "iceberg_field_id_mapping_unsupported"),
        (CapabilityReason::IcebergNameMappingUnsupported, "iceberg_name_mapping_unsupported"),
        (CapabilityReason::IcebergNestedProjectionUnsupported, "iceberg_nested_projection_unsupported"),
        (CapabilityReason::IcebergPartitionTransformUnsupported, "iceberg_partition_transform_unsupported"),
        (CapabilityReason::IcebergResidualPredicateUnsupported, "iceberg_residual_predicate_unsupported"),
        (CapabilityReason::IcebergRowSelectionUnsupported, "iceberg_row_selection_unsupported"),
        (CapabilityReason::IcebergObjectStoreReadUnsupported, "iceberg_object_store_read_unsupported"),
        (CapabilityReason::IcebergFileIoSourceAccessUnverified, "iceberg_file_io_source_access_unverified"),
    ];

    for (reason, expected) in codes {
        assert_eq!(reason.code(), expected);
    }
}

#[test]
fn cost_reason_codes_are_stable_manifest_values() {
    let codes = [
        (NativePlanCostReason::SmallSelectiveTopK, "native_small_selective_topk_not_selected_by_cost"),
        (NativePlanCostReason::SmallSelectiveResidualJoin, "native_small_selective_residual_join_not_selected_by_cost"),
        (NativePlanCostReason::SmallSelectiveFullSort, "native_small_selective_full_sort_not_selected_by_cost"),
    ];

    for (reason, expected) in codes {
        assert_eq!(reason.code(), expected);
    }
}

#[test]
fn cost_decision_has_a_shared_triage_code() {
    assert_eq!(TriageCode::NotSelectedByCost.as_str(), "not_selected_by_cost");
}

#[test]
fn capability_reasons_map_to_shared_triage_codes() {
    assert_eq!(CapabilityReason::UnsupportedExpression.triage_code(), TriageCode::UnsupportedExpression);
    assert_eq!(CapabilityReason::NativeBoundedJoinUnsupported.triage_code(), TriageCode::CapabilityEnvelopeRejected);
    assert_eq!(CapabilityReason::NativeSourceNonLocalObjectStore.triage_code(), TriageCode::UnsupportedSource);
    assert_eq!(CapabilityReason::IcebergObjectStoreReadUnsupported.triage_code(), TriageCode::UnsupportedSource);
    assert_eq!(CapabilityReason::IcebergFileIoSourceAccessUnverified.triage_code(), TriageCode::UnsupportedSource);
}

#[test]
fn planning_report_keeps_selected_and_not_supported_candidates_in_order() {
    let mut report = PlanningReport::new();
    let filter = GpuCandidate::new("FilterExec", 2);
    let projection_filter = GpuCandidate::new("ProjectionExec(FilterExec(child))", 3);

    report.push_not_supported(filter.clone(), CapabilityReason::UnsupportedExpression);
    report.push_selected(projection_filter.clone());

    assert_eq!(report.not_selected_count(), 1);
    assert_eq!(report.selected_count(), 1);
    assert_eq!(
        report.outcomes(),
        &[
            CandidateOutcome::NotSupported {
                candidate: filter,
                capability_reason: CapabilityReason::UnsupportedExpression
            },
            CandidateOutcome::Selected { candidate: projection_filter },
        ],
    );
    assert_eq!(
        report.outcomes().iter().map(CandidateOutcome::kind).collect::<Vec<_>>(),
        vec![CandidateOutcomeKind::NotSupported, CandidateOutcomeKind::Selected],
    );
}

#[test]
fn planning_report_carries_optional_correlation_ids() {
    let default_report = PlanningReport::from_outcomes([]);
    assert!(default_report.query_id().is_none());
    assert!(default_report.trace_id().is_none());

    let report = PlanningReport::from_outcomes([CandidateOutcome::NotSupported {
        candidate: GpuCandidate::new("FilterExec", 2).with_plan_node_id("n12"),
        capability_reason: CapabilityReason::UnsupportedExpression,
    }])
    .with_query_id("query-42")
    .with_trace_id("trace-7");

    assert_eq!(report.query_id().map(|id| id.as_str()), Some("query-42"));
    assert_eq!(report.trace_id().map(|id| id.as_str()), Some("trace-7"));
}

#[test]
fn planning_report_json_includes_triage_summary() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::NotSupported {
        candidate: GpuCandidate::new("FilterExec", 2).with_plan_node_id("n12"),
        capability_reason: CapabilityReason::UnsupportedExpression,
    }])
    .with_query_id("query-42");

    let summary = report.triage_summary();
    assert_eq!(summary.verdict, TriageVerdict::NotSelected);
    assert_eq!(summary.signals[0].code, TriageCode::UnsupportedExpression);

    let json = report.to_json();
    assert_eq!(json["schema"], "datafusion_nexus.planning_report.v2");
    assert_eq!(json["triage"]["verdict"], "not_selected");
    assert_eq!(json["triage"]["signals"][0]["code"], "unsupported_expression");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "query-42");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n12");
}

#[test]
fn planning_report_json_includes_data_layout_fixable_headline_fact() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::NotSupported {
        candidate: GpuCandidate::new("AggregateExec(LocalParquetScan)", 2),
        capability_reason: CapabilityReason::NativeOrderedOutputUnsupported,
    }]);

    let json = report.to_json();
    assert_eq!(json["headline_facts"]["data_layout_fixable_not_supported"], true);
    assert_eq!(json["headline_facts"]["data_layout_fixable_reason_code"], "native_ordered_output_unsupported");
    assert_eq!(json["headline_facts"]["data_layout_fixable_property"], "declared_output_ordering");
    assert_eq!(json["headline_facts"]["data_layout_fixable_candidate_shape"], "AggregateExec(LocalParquetScan)");
    assert_eq!(json["headline_facts"]["data_layout_fixable_outcome_index"], 0);
}
