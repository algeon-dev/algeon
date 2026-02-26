use super::*;

#[test]
fn phase_e2_native_pipeline_modes_are_explicit_and_report_construction_stays_strict_only() {
    assert_eq!(NativePipelineExecutionMode::StrictBounded.as_str(), "strict_bounded");
    assert_eq!(NativePipelineExecutionMode::Adaptive.as_str(), "adaptive");
    assert_eq!(
        StrictBoundedExecutionMode::UnsupportedStrictBounded.native_mode(),
        NativePipelineExecutionMode::StrictBounded
    );
    assert_eq!(AdaptiveExecutionMode::Adaptive.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert_ne!(AdaptiveExecutionMode::Adaptive.as_str(), StrictBoundedExecutionMode::StrictBounded.as_str());
    assert_ne!(
        AdaptiveExecutionMode::UnsupportedAdaptive.as_str(),
        StrictBoundedExecutionMode::UnsupportedStrictBounded.as_str()
    );

    let _: fn(
        StrictBoundedExecutionMode,
        Vec<StrictBoundedContractGateRow>,
    ) -> nexus_query_engine::Result<StrictBoundedContractReport> = StrictBoundedContractReport::from_gate_rows;
}

#[test]
fn phase_e2_strict_execution_uses_strict_admission_evidence() {
    let plan = root_join_plan();
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("root join should produce strict admission evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(admission.analysis().admits_strict_bounded_join());

    let report = StrictBoundedContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("strict admission evidence should build strict report rows");
    assert_eq!(report.execution_mode().native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(report.gate_rows().iter().all(|row| {
        row.execution_mode.native_mode() == NativePipelineExecutionMode::StrictBounded
            && row.execution_mode.as_str() != AdaptiveExecutionMode::Adaptive.as_str()
            && row.execution_mode.as_str() != AdaptiveExecutionMode::UnsupportedAdaptive.as_str()
    }));
}

#[test]
fn phase_e3_adaptive_admission_reports_use_adaptive_rows_not_strict_contract_rows() {
    let plan = root_join_plan();
    let admission = adaptive_bounded_pipeline_admission(&plan)
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert!(admission.pipeline().is_some());
    let report = admission.report();
    assert_eq!(report.execution_mode(), AdaptiveExecutionMode::Adaptive);
    let row = report.row();
    assert_eq!(row.execution_mode, AdaptiveExecutionMode::Adaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Admitted);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::ConservativeCompiledPipeline);
    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);
    assert_eq!(row.final_sink_boundary.materialization_mode, QueryOutputMode::MaterializeToHost);
    assert!(
        row.missing_strict_contracts
            .iter()
            .all(|missing| missing.contract_status != ContractStatus::RejectedByCapability)
    );

    let rendered = report.stable_kv_row();
    assert!(rendered.contains("execution_mode=adaptive"));
    assert!(rendered.contains("adaptive_strategy=conservative_compiled_pipeline"));
    assert!(rendered.contains("retry_count=0"));
    assert!(rendered.contains("spill_count=0"));
    assert!(rendered.contains("downgrade_count=0"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("unsupported_strict_bounded"));
}

#[test]
fn phase_e3_unsupported_adaptive_rows_reference_strict_gaps_without_strict_evidence() {
    let plan = unsupported_cross_key_join_plan();
    let admission = adaptive_bounded_pipeline_admission(&plan)
        .expect("adaptive admission should not fail")
        .expect("join plan should produce adaptive report evidence");

    assert_eq!(admission.native_mode(), NativePipelineExecutionMode::Adaptive);
    assert!(admission.pipeline().is_none());
    let row = admission.report().row();
    assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "none");
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);
    assert_eq!(row.rejection_reason, Some("missing_strict_contracts"));
    assert!(row.missing_strict_contracts.iter().any(|missing| {
        missing.required_contract == "unsupported_join_node_capability"
            && missing.contract_status == ContractStatus::RejectedByCapability
    }));

    let rendered = admission.report().stable_kv_row();
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(rendered.contains("required_contract=unsupported_join_node_capability"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("execution_mode=unsupported_strict_bounded"));
}

#[test]
fn phase_e6_adaptive_memory_history_lookup_miss_is_adaptive_evidence_only() {
    let plan = root_join_plan();
    let memory_history = AdaptiveMemoryHistory::new();
    let admission = adaptive_bounded_pipeline_admission_with_memory_history(&plan, Some(&memory_history))
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");
    let row = admission.report().row();

    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "memory_history_lookup");
    assert_eq!(row.memory_history_evidence, AdaptiveMemoryHistoryEvidence::LookupMiss);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Admitted);
    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("strict admission should ignore adaptive memory history");
    assert_eq!(strict_admission.native_mode(), NativePipelineExecutionMode::StrictBounded);

    let strict_report =
        StrictBoundedContractReport::from_compiled_pipeline(strict_admission.analysis(), strict_admission.pipeline())
            .expect("strict report should not read adaptive history");
    assert!(strict_report.gate_rows().iter().all(|row| {
        let rendered = row.to_stable_kv_row();
        !rendered.contains("memory_history") && !rendered.contains("lookup_miss")
    }));
}

#[test]
fn phase_e6_adaptive_memory_history_hit_reports_observed_summary_without_changing_strict_admission() {
    let plan = root_join_plan();
    let memory_history = AdaptiveMemoryHistory::new();
    let key = AdaptiveMemoryHistoryKey::for_query_plan(&plan);
    assert!(!key.as_str().contains("compiled-d1-left"));
    assert!(!key.as_str().contains("left_id"));
    memory_history.record_observation(
        key,
        AdaptiveMemoryObservation {
            resident_dataframe_peak_bytes: Some(4096),
            reservation_granted_bytes: 8192,
            source_chunks_committed: 3,
        },
    );

    let adaptive_admission = adaptive_bounded_pipeline_admission_with_memory_history(&plan, Some(&memory_history))
        .expect("adaptive admission should not fail")
        .expect("root join should produce adaptive admission evidence");
    let row = adaptive_admission.report().row();

    assert_eq!(row.runtime_actions_allowed.to_stable_value(), "memory_history_lookup");
    let AdaptiveMemoryHistoryEvidence::Observed(summary) = &row.memory_history_evidence else {
        panic!("memory history should be reported as observed adaptive evidence");
    };
    assert_eq!(summary.observation_count, 1);
    assert_eq!(summary.max_resident_dataframe_peak_bytes, Some(4096));
    assert_eq!(summary.max_reservation_granted_bytes, 8192);
    assert_eq!(summary.max_source_chunks_committed, 3);
    assert_eq!(row.runtime_action_counts.retry_count, 0);
    assert_eq!(row.runtime_action_counts.spill_count, 0);
    assert_eq!(row.runtime_action_counts.downgrade_count, 0);

    let rendered = adaptive_admission.report().stable_kv_row();
    assert!(rendered.contains("memory_history_evidence=observed:observation_count=1"));
    assert!(rendered.contains("resident_dataframe_peak_bytes=4096"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));

    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("strict admission must ignore adaptive memory history");
    assert_eq!(strict_admission.native_mode(), NativePipelineExecutionMode::StrictBounded);
    assert!(strict_admission.analysis().admits_strict_bounded_join());
}

#[test]
fn phase_e8_adaptive_diagnostic_failure_rows_are_visible_without_strict_labels() {
    let plan = root_join_plan();
    let report = AdaptivePipelineReport::diagnostic_failure(&plan, ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON);
    let row = report.row();

    assert_eq!(row.execution_mode, AdaptiveExecutionMode::UnsupportedAdaptive);
    assert_eq!(row.candidate_status, AdaptiveCandidateStatus::Rejected);
    assert_eq!(row.adaptive_strategy, AdaptiveStrategy::Unsupported);
    assert_eq!(row.rejection_reason, Some(ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON));

    let rendered = report.stable_kv_row();
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(rendered.contains("rejection_reason=adaptive_diagnostic_construction_failed"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("unsupported_strict_bounded"));
}

#[test]
fn phase_e8_adaptive_memory_history_key_distinguishes_memory_shape_factors() {
    let int_sum_gt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Sum, BinaryOp::Gt);
    let decimal_sum_gt =
        adaptive_memory_history_key_probe_plan(DataType::Decimal128(20, 2), AggregateFunction::Sum, BinaryOp::Gt);
    let int_count_gt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Count, BinaryOp::Gt);
    let int_sum_lt = adaptive_memory_history_key_probe_plan(DataType::Int32, AggregateFunction::Sum, BinaryOp::Lt);

    let int_sum_gt_key = AdaptiveMemoryHistoryKey::for_query_plan(&int_sum_gt);
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&decimal_sum_gt));
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&int_count_gt));
    assert_ne!(int_sum_gt_key, AdaptiveMemoryHistoryKey::for_query_plan(&int_sum_lt));

    assert!(!int_sum_gt_key.as_str().contains("/tmp/"));
    assert!(!int_sum_gt_key.as_str().contains("adaptive_key"));
    assert!(!int_sum_gt_key.as_str().contains("secret_literal"));
}

#[test]
fn workstream5_adaptive_task_memory_history_records_compiled_task_observations() {
    let plan = root_join_plan();
    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should compile")
        .expect("root join should compile");
    let pipeline = strict_admission.pipeline();
    let task = pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::SourceRead).unwrap();
    let task_key = AdaptiveTaskMemoryHistoryKey::for_plan_task(&plan, task, 96);

    let mut metrics = QueryMetrics::default();
    metrics.record_compiled_pipeline_task_execution(task.id(), task.kind().as_str());
    let request = MemoryReservationRequest::new(ReservationKind::SourceRead, 96, 1024)
        .with_compiled_task_id(task.id())
        .with_memory_boundary("source_read");
    metrics.record_reservation_granted_timeline_event(&request, MemoryReservationEventAction::Reserve, 96, 96, 96);
    metrics.record_reservation_timeline_event(nexus_query_engine::MemoryReservationEvent::from_request(
        &request,
        nexus_query_engine::MemoryReservationEventOutcome::Applied,
        MemoryReservationEventAction::Release,
        0,
        -96,
        0,
    ));

    let history = AdaptiveMemoryHistory::new();
    history.record_task_observations_from_query_metrics(&plan, pipeline, &metrics);
    let summary = history.lookup_task(&task_key).expect("task observation should be keyed by structural task shape");

    assert_eq!(summary.observation_count, 1);
    assert_eq!(summary.max_reservation_requested_bytes, 96);
    assert_eq!(summary.max_reservation_granted_bytes, 96);
    assert_eq!(summary.max_reservation_released_bytes, 96);
    assert!(!task_key.as_str().contains("compiled-d1-left"));
    assert!(!task_key.as_str().contains("left_id"));
}

#[test]
fn workstream5_adaptive_task_memory_history_recommends_lower_reservation_with_headroom() {
    let plan = root_join_plan();
    let strict_admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should compile")
        .expect("root join should compile");
    let pipeline = strict_admission.pipeline();
    let task = pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::SourceRead).unwrap();
    let history = AdaptiveMemoryHistory::new();
    let task_key = AdaptiveTaskMemoryHistoryKey::for_plan_task(&plan, task, 1024);

    history.record_task_observation(
        task_key,
        nexus_query_engine::AdaptiveTaskMemoryObservation {
            reservation_requested_bytes: 1024,
            reservation_granted_bytes: 1024,
            observed_peak_bytes: Some(128),
            retained_bytes: Some(96),
            reservation_released_bytes: 1024,
        },
    );

    let recommended = history
        .recommended_task_reservation_bytes_for_plan_task(&plan, task, 1024, 1024)
        .expect("observed task peak should produce an adaptive reservation recommendation");

    assert_eq!(recommended, 144);
    assert!(recommended < 1024);
    assert!(recommended >= 128);
    assert_eq!(
        history.recommended_task_reservation_bytes_for_plan_task(&plan, task, 2048, 2048),
        None,
        "different input-basis bytes should not reuse an unrelated task observation"
    );
}
