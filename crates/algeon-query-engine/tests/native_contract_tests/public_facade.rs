use arrow::datatypes::SchemaRef;
use query_engine::{
    admission::{AdmissionDecision, admit},
    backend::AnalysisContext,
    capability::AggregateStrategyExecutionOptions,
    error::Result,
    expr::Expr,
    metrics::QueryMetrics,
    mode::StatisticalAggregateExecutionMode,
    optimizer::NativeOptimizerOptions,
    pipeline::PipelineTaskKind,
    plan::QueryPlan,
    policy::PlanningPolicy,
    runtime::{
        ExecutionContext, GpuBackendControllerDraft, GpuBackendCore, GpuBackendCoreDraft, GpuBackendDeviceConfig,
        GpuBackendDeviceProfile, GpuMemoryOwnership, QueryExecutionConfig, QueueBounds,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{LocalParquetSource, validate_local_parquet_source},
    triage::TriageSummary,
};

#[test]
fn canonical_engine_paths_expose_intentional_contract_groups() {
    let _expr = Expr::column("c0");
    let _metrics = QueryMetrics::default();
    let controller = GpuBackendControllerDraft::new(QueryExecutionConfig::default(), QueueBounds::default());
    let draft = GpuBackendCoreDraft::new(
        vec![GpuBackendDeviceConfig::new(GpuBackendDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)],
        controller,
    );
    let _supported_backend_builder = GpuBackendCore::builder(draft);
    let _resident_input_map = GpuResidentInputMap::new();
    let _gpu_frame_new: fn(cudf::DataFrame) -> GpuDataFrame = GpuDataFrame::new;
    let _gpu_handle_new: fn(&ExecutionContext, GpuDataFrame, SchemaRef) -> Result<GpuResultHandle> =
        GpuResultHandle::new;
    let _optimizer = NativeOptimizerOptions::default();
    let _pipeline_kind = PipelineTaskKind::SourceRead;
    let _aggregate = AggregateStrategyExecutionOptions::default();
    let _triage = TriageSummary::healthy();
    let _bounded_decision: fn(
        &QueryPlan,
        StatisticalAggregateExecutionMode,
        PlanningPolicy,
        AnalysisContext<'_>,
    ) -> Result<AdmissionDecision> = admit;
    let _source_validation: fn(&LocalParquetSource, AnalysisContext<'_>) -> Result<()> = validate_local_parquet_source;
}
