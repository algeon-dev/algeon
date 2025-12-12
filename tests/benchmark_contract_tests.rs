use std::{sync::Arc, time::Duration};

use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};
use datafusion_cudf::{
    CandidateOutcome, CudfOptimizerConfig, FallbackReason, GpuCandidate, GpuFallbackPolicy,
    LargerIslandBenchmarkConfig, LargerIslandBenchmarkMetrics, LargerIslandBenchmarkPlanning, TimedBenchmarkRun,
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
    assert!(text.contains("cpu_baseline_wall_ms"));
    assert!(text.contains("gpu_island_wall_ms"));
    assert!(text.contains("cudf_operator_ms"));
    assert!(text.contains("host_to_device_uploads"));
    assert!(text.contains("device_to_host_materializations"));
    assert!(text.contains("fallback_rejected_count"));
    assert!(text.contains("fallback_rejected_reasons=insufficient_island_size"));
}
