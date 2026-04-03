use nexus_query_engine::{
    admission_contracts, ir, metrics_contracts, optimizer_contracts, pipeline_contracts, prelude, runtime_contracts,
    triage_contracts,
};

#[test]
fn facade_modules_expose_intentional_contract_groups() {
    let _mode = prelude::NativePipelineExecutionMode::StrictBounded;
    let _expr = ir::Expr::column("c0");
    let _metrics = metrics_contracts::QueryMetrics::default();
    let _runtime = runtime_contracts::ExecutionContext::builder();
    let _optimizer = optimizer_contracts::NativeOptimizerOptions::default();
    let _pipeline_kind = pipeline_contracts::PipelineTaskKind::SourceRead;
    let _admission_status = admission_contracts::AdaptiveCandidateStatus::Rejected;
    let _triage = triage_contracts::TriageSummary::healthy();
}
