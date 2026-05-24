//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-cudf`.
//!
//! Public APIs fall into four groups:
//!
//! - [`plan`] and [`expr`] define the DataFusion-free query IR.
//! - [`capability`] and [`pipeline`] prove and compile bounded-execution
//!   contracts without naming legacy execution families.
//! - [`runtime`] owns device, memory, boundary, and scheduler policy.
//! - [`exec`] executes a validated [`QueryPlan`] and returns
//!   either host Arrow output or GPU-resident handles according to its
//!   [`ResultSink`].
//!
//! Byte counts in native reports and runtime contracts are device bytes unless
//! explicitly labeled as host or Arrow bytes. Native ids and cache keys are
//! local identities for one plan or runtime; callers should not treat them as
//! stable global handles.

pub mod adaptive_report;
pub mod adaptive_runtime;
pub(crate) mod bounded;
/// Capability-analysis model and admission helpers for native execution.
pub mod capability;
pub mod chunk;
pub mod contract_report;
/// Arrow-to-cuDF datatype support and cast policy helpers.
pub mod datatypes;
pub mod error;
pub mod exec;
/// DataFusion-free expression IR used by native query plans.
pub mod expr;
/// Runtime metrics, counters, and memory-reservation event records.
pub mod metrics;
pub mod mode;
pub mod optimizer;
pub mod pipeline;
/// DataFusion-free query plan IR and graph operators.
pub mod plan;
pub mod runtime;
pub mod source;

pub use adaptive_report::{
    ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveAdmissionRuntimeEvidence, AdaptiveCandidateStatus,
    AdaptiveFinalSinkBoundary, AdaptiveMemoryHistoryEvidence, AdaptiveMissingStrictContract, AdaptivePipelineReport,
    AdaptivePipelineReportRow, AdaptiveRuntimeActionCounts, AdaptiveRuntimeActionsAllowed, AdaptiveSinkKind,
    AdaptiveStrategy,
};
pub use adaptive_runtime::{
    AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryHistorySummary, AdaptiveMemoryObservation,
    AdaptiveTaskMemoryHistoryKey, AdaptiveTaskMemoryHistoryShape, AdaptiveTaskMemoryHistorySummary,
    AdaptiveTaskMemoryObservation,
};
pub use chunk::{
    SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceMetadataCache, SourceTaskMemoryModel,
    plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_metadata_cache,
};
pub use contract_report::{
    StrictBoundedContractGateId, StrictBoundedContractGateRow, StrictBoundedContractOwner, StrictBoundedContractReport,
};
pub use datatypes::{
    arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type, is_supported_arrow_cast,
    is_supported_arrow_implicit_cast, is_supported_arrow_type, timestamp_unit_to_cudf_resolution,
};
pub use error::{Facts, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, Result};
pub use exec::{
    AdaptivePipelineAdmission, GpuDataFrame, GpuHashJoinBuildCache, GpuHashJoinBuildHandle, GpuLocalParquetSourceCache,
    GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap, GpuResultHandle,
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape, QueryExecutionResult,
    QueryOutput, QueryOutputMode, STRING_OFFSET_BYTES_CONSERVATIVE, StrictBoundedPipelineAdmission,
    adaptive_bounded_pipeline_admission, adaptive_bounded_pipeline_admission_with_memory_history, execute_query_plan,
    execute_query_plan_with_gpu_inputs, join_output_upper_bound_bytes, materialize_gpu_result_to_host,
    partial_aggregate_upper_bound_bytes, read_gpu_resident_source, read_local_parquet_source,
    strict_bounded_pipeline_admission, validate_local_parquet_source,
};
pub use expr::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnQualifier, ColumnRef, DatePart, Expr,
    GroupingSetsSpec, LimitSpec, LiteralValue, SortExpr, SortSpec,
};
pub use metrics::{
    BoundaryMetrics, BudgetedExecutionEvent, GraphAlgorithmRuntimeMetrics, JoinOperatorMetrics,
    JoinStrategyDecisionMetrics, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
    OperatorMetrics, QueryMetrics, TaskOperatorMetrics, TaskRuntimeMetrics,
};
pub use mode::{AdaptiveExecutionMode, NativePipelineExecutionMode, StrictBoundedExecutionMode};
pub use optimizer::{
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
    JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics,
    JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason, JoinFilterReassociationCandidateDiagnostics,
    JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
    NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, NativeOptimizerResult,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionCandidateDiagnostics, OuterJoinNullRejectionDiagnostics,
    OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason, SEMIJOIN_REDUCTION_RULE_NAME,
    SemijoinReductionAdmissionDecision, SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics,
    SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
    SemijoinReductionResult, optimize_query_plan, optimize_semijoin_reduction,
};
pub use pipeline::{
    CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
    PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
    ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
    compile_pipeline,
};
pub use plan::{
    AggregateNode, DataPlacement, FilterNode, GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS,
    GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY, GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT,
    GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec,
    GraphAlgorithmStrictBoundedLowerLevelContract, GraphBfsParams, GraphConstructionOptions, GraphConstructionPolicy,
    GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeColumnBindings,
    GraphEigenvectorCentralityParams, GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams,
    GraphKatzCentralityParams, GraphLayoutRequirement, GraphLeidenParams, GraphLouvainParams, GraphMstParams,
    GraphOperationDefinition, GraphOutDegreesAllParams, GraphPageRankParams, GraphSccParams,
    GraphSpectralClusteringParams, GraphSsspParams, GraphTriangleCountAllParams, GraphWccParams,
    HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
    LimitNode, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, QueryPlanBuilder,
    ResultSink, SchemaBehavior, ShiftDirection, SortNode, WindowFrame, WindowFunction, WindowNode, WindowSpec,
    stable_graph_result_schema,
};
pub use runtime::{
    BoundedJoinReservationContract, BoundedJoinReservationRequirement, BoundedJoinReservationStatus,
    BuildMaterializationBoundary, DevicePlacement, ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext,
    ExecutionContextBuilder, MemoryControlPolicy, MemoryReservationRequest, MemoryResourcePolicy, MemorySpaceId,
    ProbeExecutionBoundary, ReservationKind, ResultMaterializationBoundary, RetryFacts, SourceReadBoundary,
    SourceSchedulerPolicy,
};
pub use source::{
    ByteRangeDisposition, GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn,
    ParquetSourceLocation, PartitionColumn, PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature,
    ScanLimitDisposition, SchemaMappingDisposition, SourceColumnRef, SourceDiagnostics, SourceFormat, SourceNode,
    ZERO_COLUMN_COUNT_COLUMN, ZeroColumnCountDisposition,
};
