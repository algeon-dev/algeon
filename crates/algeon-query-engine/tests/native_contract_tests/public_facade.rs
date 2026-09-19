use query_engine::{
    admission::{BoundedPipelineDecision, bounded_pipeline_decision},
    backend::AnalysisContext,
    capability::AggregateStrategyExecutionOptions,
    error::Result,
    expr::Expr,
    metrics::QueryMetrics,
    optimizer::NativeOptimizerOptions,
    pipeline::PipelineTaskKind,
    plan::QueryPlan,
    runtime::{
        DeploymentIoCapability, ExecutionContext, GdsDriverProbeStatus, GpuBackendControllerDraft, GpuBackendCore,
        GpuBackendCoreDraft, GpuBackendDeviceConfig, GpuBackendDeviceProfile, GpuMemoryOwnership,
        KvikioCompatModeStatus, NvidiaFsProbeStatus, QueryExecutionConfig, QueueBounds,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::{LocalParquetSource, validate_local_parquet_source},
    triage::TriageSummary,
};

#[test]
fn canonical_engine_paths_expose_intentional_contract_groups() {
    let _expr = Expr::column("c0");
    let _metrics = QueryMetrics::default();
    let deployment = DeploymentIoCapability::from_probe_inputs(
        NvidiaFsProbeStatus::DriverMissing,
        KvikioCompatModeStatus::Unset,
        GdsDriverProbeStatus::ProbeUnimplemented,
    );
    let controller = GpuBackendControllerDraft::new(QueryExecutionConfig::default(), QueueBounds::default());
    let draft = GpuBackendCoreDraft::new(
        vec![GpuBackendDeviceConfig::new(GpuBackendDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)],
        controller,
        deployment,
    );
    let _supported_backend_builder = GpuBackendCore::builder(draft);
    let _resident_input_map = GpuResidentInputMap::new();
    let _gpu_frame_new: fn(cudf::DataFrame) -> GpuDataFrame = GpuDataFrame::new;
    let _gpu_handle_new: fn(&ExecutionContext, GpuDataFrame, arrow::datatypes::SchemaRef) -> Result<GpuResultHandle> =
        GpuResultHandle::new;
    let _optimizer = NativeOptimizerOptions::default();
    let _pipeline_kind = PipelineTaskKind::SourceRead;
    let _aggregate = AggregateStrategyExecutionOptions::default();
    let _triage = TriageSummary::healthy();
    let _bounded_decision: fn(
        &QueryPlan,
        query_engine::mode::StatisticalAggregateExecutionMode,
        AnalysisContext<'_>,
    ) -> Result<BoundedPipelineDecision> = bounded_pipeline_decision;
    let _source_validation: fn(&LocalParquetSource, AnalysisContext<'_>) -> Result<()> = validate_local_parquet_source;
}
