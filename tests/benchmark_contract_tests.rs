use std::{sync::Arc, time::Duration};

use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};
use datafusion_cudf::{
    BenchmarkGateReason, BenchmarkGateStatus, CandidateOutcome, CudfOptimizerConfig, FallbackReason, GpuCandidate,
    GpuFallbackPolicy, LargerIslandBenchmarkConfig, LargerIslandBenchmarkDecisionCriteria,
    LargerIslandBenchmarkMetrics, LargerIslandBenchmarkPlanning, PhaseCRuntimeGateStatus, TimedBenchmarkRun,
    build_larger_island_benchmark_plan, optimize_plan,
};

#[test]
fn larger_island_benchmark_config_rejects_zero_sized_inputs() {
    let err = LargerIslandBenchmarkConfig::default().with_input_rows(0).validate().unwrap_err();
    assert!(err.to_string().contains("input_rows"));

    let err = LargerIslandBenchmarkConfig::default().with_batch_size(0).validate().unwrap_err();
    assert!(err.to_string().contains("batch_size"));

    let err = LargerIslandBenchmarkConfig::default().with_group_count(0).validate().unwrap_err();
    assert!(err.to_string().contains("group_count"));

    let err = LargerIslandBenchmarkConfig::default().with_iterations(0).validate().unwrap_err();
    assert!(err.to_string().contains("iterations"));
}

#[test]
fn larger_island_benchmark_plan_is_one_fused_gpu_candidate() {
    let config = LargerIslandBenchmarkConfig::default()
        .with_input_rows(1024)
        .with_batch_size(128)
        .with_group_count(32)
        .with_limit(8);
    let plan: Arc<dyn ExecutionPlan> = build_larger_island_benchmark_plan(&config).unwrap();

    let optimizer = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_min_operators_per_island(2);
    let result = optimize_plan(plan, &optimizer).unwrap();

    assert_eq!(result.plan.name(), "CudfGpuIslandExec");
    assert_eq!(
        result.report.outcomes(),
        &[CandidateOutcome::Rewritten {
            candidate: GpuCandidate::with_operator_count(
                "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(child)))))",
                2,
                5,
            ),
        }],
    );
}

#[test]
fn larger_island_benchmark_report_keeps_required_measurements() {
    let planning = LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::from_outcomes([
        CandidateOutcome::Rejected {
            candidate: GpuCandidate::with_operator_count("FilterExec(child)", 3, 1),
            reason: FallbackReason::InsufficientIslandSize,
        },
        CandidateOutcome::Rewritten {
            candidate: GpuCandidate::with_operator_count(
                "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(child)))))",
                2,
                5,
            ),
        },
    ]));

    let metrics = LargerIslandBenchmarkMetrics::from_metrics(&MetricsSet::new());
    let report = datafusion_cudf::LargerIslandBenchmarkReport {
        config: LargerIslandBenchmarkConfig::default().with_input_rows(10).with_batch_size(5),
        cpu_baseline: TimedBenchmarkRun::from_samples([Duration::from_millis(12), Duration::from_millis(10)]),
        gpu_island: TimedBenchmarkRun::from_samples([Duration::from_millis(9), Duration::from_millis(8)]),
        gpu_metrics: metrics,
        planning,
        output_batches: 1,
        output_rows: 2,
        peak_device_memory_bytes: None,
    };

    assert_eq!(report.cpu_baseline.iterations, 2);
    assert_eq!(report.gpu_island.min, Duration::from_millis(8));
    assert_eq!(report.planning.candidate_count, 2);
    assert_eq!(report.planning.rewritten_count, 1);
    assert_eq!(report.planning.rejected_count, 1);
    assert_eq!(report.planning.rejected_reasons, vec![FallbackReason::InsufficientIslandSize]);
    assert_eq!(report.gpu_metrics.host_to_device_uploads, 0);
    assert_eq!(report.peak_device_memory_bytes, None);

    let text = report.to_text();
    assert!(text.contains("input_batches"));
    assert!(text.contains("cpu_baseline_wall_ms"));
    assert!(text.contains("gpu_island_wall_ms"));
    assert!(text.contains("cudf_operator_ms"));
    assert!(text.contains("phase_c_memory_control_policy=unbounded"));
    assert!(text.contains("phase_c_runtime_gate=not_evaluated"));
    assert!(text.contains("source_chunks_planned=0"));
    assert!(text.contains("reservation_denied_count=0"));
    assert!(text.contains("host_to_device_uploads"));
    assert!(text.contains("device_to_host_materializations"));
    assert!(text.contains("fallback_rejected_count"));
    assert!(text.contains("fallback_rejected_reasons=insufficient_island_size"));
    assert!(text.contains("benchmark_gate_status="));
    assert!(text.contains("benchmark_gate_reasons="));
}

#[test]
fn benchmark_gate_allows_phase_4_only_when_gpu_beats_cpu_without_boundary_or_fallback_pressure() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(80), 1, 1, 0, None);

    let assessment = report.assess_phase4_readiness(&LargerIslandBenchmarkDecisionCriteria::default());

    assert_eq!(assessment.status, BenchmarkGateStatus::ProceedToPhase4);
    assert_eq!(assessment.reasons, Vec::<BenchmarkGateReason>::new());
    assert_eq!(assessment.host_device_boundary_count, 2);
    assert!(assessment.gpu_speedup > 1.0);
}

#[test]
fn benchmark_gate_stops_native_engine_expansion_when_gpu_is_not_faster_than_cpu() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(125), 1, 1, 0, None);

    let assessment = report.assess_phase4_readiness(&LargerIslandBenchmarkDecisionCriteria::default());

    assert_eq!(assessment.status, BenchmarkGateStatus::DoNotExpandNativeEngine);
    assert_eq!(assessment.reasons, vec![BenchmarkGateReason::GpuNotFasterThanCpu]);
}

#[test]
fn benchmark_gate_routes_fallback_pressure_to_coverage_work_before_phase_4() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(70), 1, 1, 2, None);

    let assessment = report.assess_phase4_readiness(&LargerIslandBenchmarkDecisionCriteria::default());

    assert_eq!(assessment.status, BenchmarkGateStatus::ImproveCoverageBeforeNativeEngine);
    assert_eq!(assessment.reasons, vec![BenchmarkGateReason::RejectedFallbackCandidates]);
}

#[test]
fn benchmark_gate_reports_extra_materialization_boundaries_as_datafusion_boundary_pressure() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(70), 1, 3, 0, None);

    let assessment = report.assess_phase4_readiness(&LargerIslandBenchmarkDecisionCriteria::default());

    assert_eq!(assessment.status, BenchmarkGateStatus::KeepExtensionAdapter);
    assert_eq!(assessment.reasons, vec![BenchmarkGateReason::TooManyMaterializationBoundaries]);
}

#[test]
fn benchmark_gate_can_require_peak_device_memory_before_phase_4() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(70), 1, 1, 0, None);
    let criteria = LargerIslandBenchmarkDecisionCriteria::default().with_peak_device_memory_required(true);

    let assessment = report.assess_phase4_readiness(&criteria);

    assert_eq!(assessment.status, BenchmarkGateStatus::NeedMoreMeasurements);
    assert_eq!(assessment.reasons, vec![BenchmarkGateReason::PeakDeviceMemoryUnavailable]);
}

#[test]
fn benchmark_gate_accepts_recorded_peak_device_memory_when_required() {
    let report = benchmark_report(Duration::from_millis(100), Duration::from_millis(70), 1, 1, 0, Some(256));
    let criteria = LargerIslandBenchmarkDecisionCriteria::default().with_peak_device_memory_required(true);

    let assessment = report.assess_phase4_readiness(&criteria);

    assert_eq!(assessment.status, BenchmarkGateStatus::ProceedToPhase4);
    assert_eq!(assessment.reasons, Vec::<BenchmarkGateReason>::new());
    assert_eq!(report.peak_device_memory_bytes, Some(256));
    assert_eq!(report.gpu_metrics.peak_device_memory_bytes, Some(256));
}

#[test]
fn phase_c_runtime_gate_passes_only_for_bounded_zero_upload_chunked_native_metrics() {
    let mut metrics = benchmark_metrics("bounded", 1);
    metrics.host_to_device_uploads = 0;
    metrics.host_to_device_uploaded_rows = 0;
    metrics.host_to_device_uploaded_bytes = 0;
    metrics.device_to_host_materializations = 1;
    metrics.source_chunks_planned = 2;
    metrics.source_chunks_committed = 2;
    metrics.row_groups_planned = 2;
    metrics.row_groups_committed = 2;
    metrics.resident_dataframe_peak_bytes = Some(1024);

    assert_eq!(metrics.assess_phase_c_runtime_gate(true), PhaseCRuntimeGateStatus::Pass);

    metrics.host_to_device_uploads = 1;
    assert_eq!(metrics.assess_phase_c_runtime_gate(true), PhaseCRuntimeGateStatus::Fail);

    metrics.phase_c_memory_control_policy = "unbounded";
    assert_eq!(metrics.assess_phase_c_runtime_gate(true), PhaseCRuntimeGateStatus::NotEvaluated);
}

fn benchmark_report(
    cpu_wall_time: Duration,
    gpu_wall_time: Duration,
    rewritten_count: usize,
    materializations: usize,
    rejected_count: usize,
    peak_device_memory_bytes: Option<usize>,
) -> datafusion_cudf::LargerIslandBenchmarkReport {
    let mut outcomes = Vec::new();
    for _ in 0..rewritten_count {
        outcomes.push(CandidateOutcome::Rewritten {
            candidate: GpuCandidate::with_operator_count(
                "LocalLimitExec(SortExec(AggregateExec(ProjectionExec(FilterExec(child)))))",
                2,
                5,
            ),
        });
    }
    for _ in 0..rejected_count {
        outcomes.push(CandidateOutcome::Rejected {
            candidate: GpuCandidate::with_operator_count("FilterExec(child)", 3, 1),
            reason: FallbackReason::UnsupportedExpression,
        });
    }

    datafusion_cudf::LargerIslandBenchmarkReport {
        config: LargerIslandBenchmarkConfig::default().with_input_rows(10).with_batch_size(5),
        cpu_baseline: TimedBenchmarkRun::from_samples([cpu_wall_time]),
        gpu_island: TimedBenchmarkRun::from_samples([gpu_wall_time]),
        gpu_metrics: LargerIslandBenchmarkMetrics {
            peak_device_memory_bytes,
            resident_dataframe_peak_bytes: peak_device_memory_bytes,
            device_to_host_materializations: materializations,
            elapsed_compute: Some(gpu_wall_time),
            ..benchmark_metrics("unbounded", 1)
        },
        planning: LargerIslandBenchmarkPlanning::from_report(&datafusion_cudf::PlanningReport::from_outcomes(outcomes)),
        output_batches: 1,
        output_rows: 2,
        peak_device_memory_bytes,
    }
}

fn benchmark_metrics(
    phase_c_memory_control_policy: &'static str,
    host_to_device_uploads: usize,
) -> LargerIslandBenchmarkMetrics {
    LargerIslandBenchmarkMetrics {
        phase_c_memory_control_policy,
        phase_c_runtime_gate: "not_evaluated",
        input_batches: 1,
        input_rows: 10,
        resident_dataframe_peak_bytes: None,
        allocator_high_water_bytes: None,
        reservation_requested_bytes: 0,
        reservation_granted_bytes: 0,
        reservation_denied_count: 0,
        source_chunks_planned: 0,
        source_chunks_attempted: 0,
        source_chunks_committed: 0,
        source_chunks_retried: 0,
        row_groups_planned: 0,
        row_groups_committed: 0,
        partial_aggregate_count: 0,
        partial_aggregate_rows: 0,
        final_merge_attempts: 0,
        final_merge_levels: 0,
        retry_count: 0,
        retry_resource_exhausted_count: 0,
        retry_reservation_failure_count: 0,
        peak_device_memory_bytes: None,
        host_to_device_uploads,
        host_to_device_uploaded_rows: host_to_device_uploads.saturating_mul(10),
        host_to_device_uploaded_bytes: host_to_device_uploads.saturating_mul(120),
        device_to_host_materializations: 1,
        device_to_host_materialized_rows: 2,
        device_to_host_materialized_bytes: 24,
        parquet_files: 0,
        parquet_bytes_known: 0,
        row_groups_selected: 0,
        read_columns: 0,
        parquet_read_time: Duration::ZERO,
        record_batch_upload_time: Duration::from_millis(10),
        cudf_operator_time: Duration::from_millis(30),
        record_batch_materialization_time: Duration::from_millis(5),
        elapsed_compute: None,
        output_rows: Some(2),
    }
}
