use arrow::datatypes::SchemaRef;
use query_planning::{
    admission::{AdmissionDecision, admit},
    backend::AnalysisContext,
    capability::AggregateStrategyExecutionOptions,
    error::Result,
    expr::Expr,
    mode::StatisticalAggregateExecutionMode,
    optimizer::NativeOptimizerOptions,
    pipeline::PipelineTaskKind,
    plan::QueryPlan,
    policy::PlanningPolicy,
    source::{LocalParquetSource, validate_local_parquet_source},
    triage::TriageSummary,
};
use query_runtime::{
    metrics::QueryMetrics,
    runtime::{
        ExecutionContext, GpuBackendConfig, GpuBackendCore, GpuBackendDeviceConfig, GpuBackendDeviceProfile,
        GpuMemoryOwnership, QueryExecutionConfig, QueueBounds, ValidatedGpuBackendConfig,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
};

fn canonical_engine_paths_expose_intentional_contract_groups() {
    let _expr = Expr::column("c0");
    let _metrics = QueryMetrics::default();
    let _validate: fn(GpuBackendConfig) -> Result<ValidatedGpuBackendConfig> = GpuBackendConfig::validate;
    let _construct: fn(ValidatedGpuBackendConfig) -> Result<GpuBackendCore> = GpuBackendCore::new;
    let _config = GpuBackendConfig::new(
        vec![GpuBackendDeviceConfig::new(GpuBackendDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive)],
        QueryExecutionConfig::default(),
        QueueBounds::default(),
    );
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

const _: fn() = canonical_engine_paths_expose_intentional_contract_groups;
