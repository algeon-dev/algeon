use super::*;

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_left_semi_join_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-left-semi-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_semi_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 3);
    let mut rows = (0..batches[0].num_rows())
        .map(|row| {
            (
                int32_array(&batches[0], 0).value(row),
                int32_array(&batches[0], 1).value(row),
                int64_array(&batches[0], 2).value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 5, 100), (1, 30, 300), (2, 20, 200)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_left_anti_join_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-left-anti-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_left_anti_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));
    assert_eq!(rule.last_source_diagnostics().len(), 2);

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 0);
    assert_eq!(batches[0].num_columns(), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_join_aggregate_pipeline() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-join-aggregate");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(batches[0].num_columns(), 2);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(&batches[0], 1).values().as_ref(), &[303, 202]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_bounded_q11_aggregate_over_join_subplan_inputs() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-q11-admission");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").expect("TPC-H Q11 query must exist");
    let plan = tpch_sql_plan(&dir, q11.sql);
    let lowering = lower_physical_plan_to_native(Arc::clone(&plan));
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert!(
        lowering.plans[0].has_bounded_executable_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline(),
        "direct Q11 lowering should be bounded executable:\n{}",
        lowering.plans[0]
    );
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert!(
        optimized.as_any().downcast_ref::<NativeCudfExec>().is_some(),
        "Q11 should optimize to NativeCudfExec; report={report:?}; diagnostics={:?}",
        rule.last_planning_diagnostics()
    );
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_enabled").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_execution_status").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_lower_level_blockers").map(String::as_str),
        Some("")
    );
    assert!(
        diagnostics.bounded_join_reservation_facts.get("bounded_join_execution_families").is_some_and(|families| {
            families.contains("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
        }),
        "Q11 bounded planner facts must name the aggregate-over-join subplan inputs family"
    );
    assert_eq!(
        diagnostics
            .bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability")
            .map(String::as_str),
        Some("keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline")
    );
    assert_eq!(
        diagnostics
            .bounded_join_reservation_facts
            .get("bounded_join_keyless_conditional_join_aggregate_over_join_subplan_inputs_pipeline_capability_detail")
            .map(String::as_str),
        Some(
            "root_join=keyless_conditional_inner;left_input=projection_wrapped_grouped_aggregate_over_bounded_join_subplan;right_input=projection_wrapped_scalar_global_aggregate_over_bounded_join_subplan;root_output=left_only;post_root=projection_sort"
        )
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_optimizer_rule_rewrites_and_executes_bounded_root_hash_join() {
    let dir = unique_temp_dir("datafusion-cudf-native-exec-bounded-root-join");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_hash_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    assert!(optimized.as_any().downcast_ref::<NativeCudfExec>().is_some());
    let report = rule.last_report().expect("native optimizer should record a planning report");
    assert_eq!(report.rewritten_count(), 1);
    assert_eq!(report.rejected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Rewritten { .. }));

    let batches = futures::executor::block_on(collect(optimized, Arc::new(TaskContext::default()))).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    assert_eq!(batches[0].num_columns(), 3);
    let mut rows = (0..batches[0].num_rows())
        .map(|row| {
            (
                int32_array(&batches[0], 0).value(row),
                int64_array(&batches[0], 1).value(row),
                int64_array(&batches[0], 2).value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(rows, vec![(1, 100, 1), (1, 300, 1), (2, 200, 2)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_exec_preserves_bounded_join_rejection_facts_from_native_runtime() {
    let native_plan = bounded_left_build_left_semi_join_host_sink_plan();
    let exec = Arc::new(NativeCudfExec::try_new(native_plan, bounded_config()).unwrap()) as Arc<dyn ExecutionPlan>;

    let error = futures::executor::block_on(collect(exec, Arc::new(TaskContext::default()))).unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("native_code"), Some("unsupported_plan_shape"));
    assert_eq!(adapter_error.fact_value("native_operation"), Some("cudf_query_engine.bounded.plan"));
    assert_eq!(adapter_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(adapter_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        adapter_error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(adapter_error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(adapter_error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_eq!(adapter_error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(adapter_error.fact_value("bounded_join_capability_rejection_scope"), Some("node"));
    assert_eq!(
        adapter_error.fact_value("bounded_join_capability_rejection_reason"),
        Some("unsupported_join_node_capability")
    );
    assert_eq!(
        adapter_error.fact_value("modeled_reservations"),
        Some(
            "source_read,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization"
        )
    );
    assert_eq!(adapter_error.fact_value("bounded_join_lower_level_blockers"), Some(""));
    assert_eq!(
        adapter_error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_hash_table_observability"),
        Some("filtered_join_device_allocated_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_output_size_observability"),
        Some("filtered_join_probe_state_output_size")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_state_contract"),
        Some("filtered_join_retained_probe_state")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_state_observability"),
        Some("filtered_join_probe_device_allocated_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_materialization_contract"),
        Some("filtered_join_probe_state_materialize_indices")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(adapter_error.fact_value("bounded_join_execution_family_alias_compatibility"), None);
    assert_eq!(
        adapter_error.fact_value("cudf_rs_mixed_join_size_data_alias_compatibility"),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(
        adapter_error.fact_value("bounded_join_mixed_keyed_residual_capability_detail"),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );
    assert_eq!(adapter_error.fact_value("path"), None);
    assert_eq!(adapter_error.fact_value("file_index"), None);
    assert_eq!(adapter_error.fact_value("row_groups"), None);
    assert_eq!(adapter_error.fact_value("chunk_index"), None);
}

#[test]
fn bounded_join_facts_cover_planner_supported_and_runtime_rejected_paths() {
    let dir = unique_temp_dir("datafusion-cudf-bounded-join-rejection-split");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let plan = root_nested_loop_join_plan(&left, &right);
    let rule = CudfNativePhysicalOptimizerRule::new(native_optimizer_config());
    let _ = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let diagnostics = rule.last_planning_diagnostics().expect("native optimizer should retain diagnostics");
    let planner_facts = &diagnostics.bounded_join_reservation_facts;

    assert_eq!(planner_facts.get("bounded_join_enabled").map(String::as_str), Some("true"));
    assert_eq!(planner_facts.get("bounded_join_execution_status").map(String::as_str), Some("enabled"));
    let expected_unsupported_shape = BoundedJoinReservationContract.supported_shape_requirement();
    assert_eq!(planner_facts.get("bounded_join_unsupported_shape").map(String::as_str), None);
    assert_eq!(
        planner_facts.get("modeled_reservations").map(String::as_str),
        Some(
            "source_read,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization"
        )
    );
    assert_eq!(planner_facts.get("blocked_reservations").map(String::as_str), Some(""));
    assert_eq!(planner_facts.get("bounded_join_execution_family_alias_compatibility").map(String::as_str), None);
    assert_eq!(
        planner_facts.get("cudf_rs_mixed_join_size_data_alias_compatibility").map(String::as_str),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(
        planner_facts.get("bounded_join_mixed_keyed_residual_capability_detail").map(String::as_str),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );

    fs::remove_dir_all(dir).unwrap();

    // --- Runtime path ---
    let native_plan = bounded_left_build_left_semi_join_host_sink_plan();
    let exec = Arc::new(NativeCudfExec::try_new(native_plan, bounded_config()).unwrap()) as Arc<dyn ExecutionPlan>;
    let error = futures::executor::block_on(collect(exec, Arc::new(TaskContext::default()))).unwrap_err();
    let runtime_error = adapter_error(&error);

    assert_eq!(runtime_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_status"), Some("unsupported_plan_shape"));
    assert_eq!(runtime_error.fact_value("bounded_join_unsupported_shape"), Some(expected_unsupported_shape.as_str()));
    let expected_execution_families = cudf_query_engine::BoundedJoinReservationContract.execution_families_csv();
    assert_eq!(runtime_error.fact_value("bounded_join_execution_families"), Some(expected_execution_families.as_str()));
    assert_eq!(runtime_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(
        runtime_error.fact_value("modeled_reservations"),
        Some(
            "source_read,build_side,hash_table,probe_side,filtered_probe_state,mixed_size_data,gather_map,join_output,materialization"
        )
    );
    assert_eq!(runtime_error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_family_alias_compatibility"), None);
    assert_eq!(
        runtime_error.fact_value("cudf_rs_mixed_join_size_data_alias_compatibility"),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(
        runtime_error.fact_value("bounded_join_mixed_keyed_residual_capability_detail"),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );

    assert_eq!(
        planner_facts.get("modeled_reservations").map(String::as_str),
        runtime_error.fact_value("modeled_reservations")
    );
    assert_eq!(
        planner_facts.get("blocked_reservations").map(String::as_str),
        runtime_error.fact_value("blocked_reservations")
    );
    assert_eq!(
        planner_facts.get("bounded_join_execution_family_alias_compatibility").map(String::as_str),
        runtime_error.fact_value("bounded_join_execution_family_alias_compatibility")
    );
    assert_eq!(
        planner_facts.get("cudf_rs_mixed_join_size_data_alias_compatibility").map(String::as_str),
        runtime_error.fact_value("cudf_rs_mixed_join_size_data_alias_compatibility")
    );
    assert_eq!(
        planner_facts.get("bounded_join_mixed_keyed_residual_capability_detail").map(String::as_str),
        runtime_error.fact_value("bounded_join_mixed_keyed_residual_capability_detail")
    );
}
