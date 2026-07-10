use datafusion_nexus::{CandidateOutcome, FallbackReason, GpuCandidate, GpuFallbackPolicy, PlanningReport};
use nexus_query_engine::{TriageCode, TriageVerdict};

#[test]
fn gpu_preferred_is_default_development_policy() {
    assert_eq!(GpuFallbackPolicy::default(), GpuFallbackPolicy::GpuPreferred);
}

#[test]
fn fallback_reason_codes_are_stable_manifest_values() {
    let codes = [
        (FallbackReason::UnsupportedOperator, "unsupported_operator"),
        (FallbackReason::UnsupportedExpression, "unsupported_expression"),
        (FallbackReason::UnsupportedType, "unsupported_type"),
        (FallbackReason::ExplicitCpuPolicy, "explicit_cpu_policy"),
        (FallbackReason::NativeSortPreservingMergeUnsupported, "native_sort_preserving_merge_unsupported"),
        (FallbackReason::NativeDistributedAggregateUnsupported, "native_distributed_aggregate_unsupported"),
        (FallbackReason::NativePartitionPreservingSortUnsupported, "native_partition_preserving_sort_unsupported"),
        (FallbackReason::NativeBoundedJoinUnsupported, "native_bounded_join_unsupported"),
        (FallbackReason::NativeAdaptivePipelineUnsupported, "native_adaptive_pipeline_unsupported"),
        (FallbackReason::NativeSmallSelectiveTopkCpuPreferred, "native_small_selective_topk_cpu_preferred"),
        (
            FallbackReason::NativeSmallSelectiveResidualJoinCpuPreferred,
            "native_small_selective_residual_join_cpu_preferred",
        ),
        (FallbackReason::NativeSmallSelectiveFullSortCpuPreferred, "native_small_selective_full_sort_cpu_preferred"),
        (FallbackReason::NativeOrderedOutputUnsupported, "native_ordered_output_unsupported"),
        (FallbackReason::NativeSourceNonLocalObjectStore, "native_source_non_local_object_store"),
        (FallbackReason::NativeSourcePartitionColumns, "native_source_partition_columns"),
        (FallbackReason::NativeSourceExprAdapter, "native_source_expr_adapter"),
        (FallbackReason::NativeSourceParquetFileReaderFactory, "native_source_parquet_file_reader_factory"),
        (FallbackReason::NativeSourceScanLimit, "native_source_scan_limit"),
        (FallbackReason::NativeSourcePreserveOrder, "native_source_preserve_order"),
        (FallbackReason::NativeSourceOutputOrdering, "native_source_output_ordering"),
        (FallbackReason::NativeSourcePushedDownParquetPredicate, "native_source_pushed_down_parquet_predicate"),
        (FallbackReason::NativeSourceEmpty, "native_source_empty"),
        (FallbackReason::NativeSourcePartitionValues, "native_source_partition_values"),
        (FallbackReason::NativeSourceByteRange, "native_source_byte_range"),
        (FallbackReason::NativeSourceFileOrdering, "native_source_file_ordering"),
        (FallbackReason::NativeSourceRowSelection, "native_source_row_selection"),
        (FallbackReason::NativeSourceSchemaMapping, "native_source_schema_mapping"),
        (FallbackReason::TableFormatUnsupported, "table_format_unsupported"),
        (FallbackReason::TableFormatPlanningFailed, "table_format_planning_failed"),
        (FallbackReason::IcebergNonParquetDataFile, "iceberg_non_parquet_data_file"),
        (FallbackReason::IcebergDeleteFilesUnsupported, "iceberg_delete_files_unsupported"),
        (FallbackReason::IcebergFieldIdMappingUnsupported, "iceberg_field_id_mapping_unsupported"),
        (FallbackReason::IcebergNameMappingUnsupported, "iceberg_name_mapping_unsupported"),
        (FallbackReason::IcebergNestedProjectionUnsupported, "iceberg_nested_projection_unsupported"),
        (FallbackReason::IcebergPartitionTransformUnsupported, "iceberg_partition_transform_unsupported"),
        (FallbackReason::IcebergResidualPredicateUnsupported, "iceberg_residual_predicate_unsupported"),
        (FallbackReason::IcebergRowSelectionUnsupported, "iceberg_row_selection_unsupported"),
        (FallbackReason::IcebergObjectStoreReadUnsupported, "iceberg_object_store_read_unsupported"),
        (FallbackReason::IcebergSnapshotUnsupported, "iceberg_snapshot_unsupported"),
        (FallbackReason::TableFormatGpuOnlyLeafNotRewritten, "table_format_gpu_only_leaf_not_rewritten"),
    ];

    for (reason, expected) in codes {
        assert_eq!(reason.code(), expected);
    }
}

#[test]
fn fallback_reasons_map_to_shared_triage_codes() {
    assert_eq!(FallbackReason::UnsupportedExpression.triage_code(), TriageCode::UnsupportedExpression);
    assert_eq!(FallbackReason::NativeBoundedJoinUnsupported.triage_code(), TriageCode::StrictBoundedRejected);
    assert_eq!(FallbackReason::NativeAdaptivePipelineUnsupported.triage_code(), TriageCode::AdaptivePipelineRejected);
    assert_eq!(FallbackReason::NativeSourceNonLocalObjectStore.triage_code(), TriageCode::UnsupportedSource);
    assert_eq!(FallbackReason::IcebergObjectStoreReadUnsupported.triage_code(), TriageCode::UnsupportedSource);
}

#[test]
fn planning_report_keeps_rewritten_and_rejected_candidates_in_order() {
    let mut report = PlanningReport::new();
    let filter = GpuCandidate::new("FilterExec", 2);
    let projection_filter = GpuCandidate::new("ProjectionExec(FilterExec(child))", 3);

    report.push_rejected(filter.clone(), FallbackReason::UnsupportedExpression);
    report.push_rewritten(projection_filter.clone());

    assert_eq!(report.rejected_count(), 1);
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(
        report.outcomes(),
        &[
            CandidateOutcome::Rejected { candidate: filter, reason: FallbackReason::UnsupportedExpression },
            CandidateOutcome::Rewritten { candidate: projection_filter },
        ],
    );
}

#[test]
fn planning_report_carries_optional_correlation_ids() {
    let default_report = PlanningReport::from_outcomes([]);
    assert!(default_report.query_id().is_none());
    assert!(default_report.trace_id().is_none());

    let report = PlanningReport::from_outcomes([CandidateOutcome::Rejected {
        candidate: GpuCandidate::new("FilterExec", 2).with_plan_node_id("n12"),
        reason: FallbackReason::UnsupportedExpression,
    }])
    .with_query_id("query-42")
    .with_trace_id("trace-7");

    assert_eq!(report.query_id().map(|id| id.as_str()), Some("query-42"));
    assert_eq!(report.trace_id().map(|id| id.as_str()), Some("trace-7"));
}

#[test]
fn planning_report_json_includes_triage_summary() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::Rejected {
        candidate: GpuCandidate::new("FilterExec", 2).with_plan_node_id("n12"),
        reason: FallbackReason::UnsupportedExpression,
    }])
    .with_query_id("query-42");

    let summary = report.triage_summary();
    assert_eq!(summary.verdict, TriageVerdict::FellBack);
    assert_eq!(summary.signals[0].code, TriageCode::UnsupportedExpression);

    let json = report.to_json();
    assert_eq!(json["triage"]["verdict"], "fell_back");
    assert_eq!(json["triage"]["signals"][0]["code"], "unsupported_expression");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["query_id"], "query-42");
    assert_eq!(json["triage"]["signals"][0]["evidence"]["plan_node_id"], "n12");
}

#[test]
fn planning_report_json_includes_data_layout_fixable_headline_fact() {
    let report = PlanningReport::from_outcomes([CandidateOutcome::Rejected {
        candidate: GpuCandidate::new("AggregateExec(LocalParquetScan)", 2),
        reason: FallbackReason::NativeOrderedOutputUnsupported,
    }]);

    let json = report.to_json();
    assert_eq!(json["headline_facts"]["data_layout_fixable_rejection"], true);
    assert_eq!(json["headline_facts"]["data_layout_fixable_reason_code"], "native_ordered_output_unsupported");
    assert_eq!(json["headline_facts"]["data_layout_fixable_property"], "declared_output_ordering");
    assert_eq!(json["headline_facts"]["data_layout_fixable_candidate_shape"], "AggregateExec(LocalParquetScan)");
    assert_eq!(json["headline_facts"]["data_layout_fixable_outcome_index"], 0);
}
