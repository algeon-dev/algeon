use datafusion_cudf::{CandidateOutcome, FallbackReason, GpuCandidate, GpuFallbackPolicy, PlanningReport};

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
    ];

    for (reason, expected) in codes {
        assert_eq!(reason.code(), expected);
    }
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
