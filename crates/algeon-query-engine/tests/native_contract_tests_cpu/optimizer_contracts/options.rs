use super::*;

#[test]
fn native_optimizer_entry_point_reports_skipped_shared_resident_candidate() {
    let plan = single_source_plan();

    let result = optimize_query_plan(
        plan.clone(),
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(!result.changed());
    assert_eq!(result.diagnostics.shared_resident_extraction.considered_count, 1);
    assert_eq!(result.diagnostics.shared_resident_extraction.extracted_count, 0);
    assert_eq!(result.diagnostics.shared_resident_extraction.candidates.len(), 1);
    assert_eq!(
        result.diagnostics.shared_resident_extraction.candidates[0].decision,
        SharedResidentExtractionDecision::Skipped
    );
    assert_eq!(
        result.diagnostics.shared_resident_extraction.candidates[0].reason,
        SharedResidentExtractionReason::NotRepeated
    );
    assert_eq!(result.diagnostics.semijoin_reduction.considered_count, 0);
    assert_eq!(result.diagnostics.semijoin_reduction.applied_count, 0);
    assert!(result.diagnostics.semijoin_reduction.candidates.is_empty());
}

#[test]
fn native_optimizer_options_can_disable_semijoin_reduction_pass() {
    let plan = single_source_plan();
    let options = NativeOptimizerOptions::disabled();

    let result = optimize_query_plan(
        plan.clone(),
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert!(!options.projection_simplification.enabled);
    assert!(!options.semijoin_reduction.enabled);
    assert!(!options.count_non_null_column_pruning.enabled);
    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.diagnostics.shared_resident_extraction.considered_count, 1);
    assert_eq!(result.diagnostics.shared_resident_extraction.extracted_count, 0);
}

#[test]
fn native_optimizer_options_validate_expert_optimizer_thresholds() {
    NativeOptimizerOptions::default().validate().unwrap();
    NativeOptimizerOptions::disabled().validate().unwrap();

    type InvalidateOption = fn(&mut NativeOptimizerOptions);
    let cases: &[(&str, InvalidateOption)] = &[
        ("cost_threshold_multiplier", |options| options.semijoin_reduction.cost_threshold_multiplier = 0),
        ("small_build_side_bytes", |options| options.join_build_side_selection.small_build_side_bytes = 0),
        ("small_build_side_rows", |options| options.join_build_side_selection.small_build_side_rows = 0),
        ("max_dimension_rows", |options| options.dimension_derived_row_group_pruning.max_dimension_rows = 0),
        ("max_variant_selected_bytes", |options| options.local_parquet_source_reuse.max_variant_selected_bytes = 0),
    ];
    for &(field, invalidate) in cases {
        let mut options = NativeOptimizerOptions::default();
        invalidate(&mut options);
        let error = options.validate().unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidConfiguration, "field={field}");
        assert_eq!(error.fact_value("field"), Some(field));
        assert_eq!(error.fact_value("reason"), Some("zero"));
    }
}
