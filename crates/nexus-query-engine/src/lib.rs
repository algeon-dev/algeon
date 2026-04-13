//! Native cudf-rs query engine contracts.
//!
//! This crate deliberately does not depend on DataFusion. DataFusion-specific
//! matching, fallback reporting, and adapter wrappers belong in
//! `datafusion-nexus`.
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
/// DataFusion-free dynamic-filter producer/consumer route IR.
pub mod dynamic_filter;
pub mod error;
pub mod exec;
/// DataFusion-free expression IR used by native query plans.
pub mod expr;
/// Approximate cuGraph memory pricing outside strict bounded admission.
pub mod graph_memory;
/// Stable helper primitives shared by report and benchmark harnesses.
pub mod harness;
/// Runtime metrics, counters, and memory-reservation event records.
pub mod metrics;
pub mod mode;
/// Stable name-generation helpers shared across native planning code.
pub mod naming;
pub mod optimizer;
pub(crate) mod parquet_metadata;
/// Shared Parquet footer/statistics helpers for native pruning and sizing.
pub mod parquet_stats;
pub mod pipeline;
/// DataFusion-free query plan IR and graph operators.
pub mod plan;
pub mod runtime;
pub mod source;
pub mod triage;
pub(crate) mod util;

/// Common native-engine imports for constructing and executing a plan.
pub mod prelude {
    pub use crate::{
        ExecutionContext, ExecutionContextBuilder, Expr, LocalParquetSource, NativePipelineExecutionMode, QueryOutput,
        QueryOutputMode, QueryPlan, QueryPlanBuilder, ResultSink, execute_query_plan,
    };
}

/// DataFusion-free native IR contracts.
pub mod ir {
    pub use crate::{
        datatypes::{
            arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type, diagnostic_arrow_type,
            is_supported_arrow_cast, is_supported_arrow_implicit_cast, is_supported_arrow_type,
            is_supported_list_carry_type, is_timestamp_with_timezone, timestamp_unit_to_cudf_resolution,
        },
        expr::{
            AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnQualifier, ColumnRef, DatePart,
            Expr, GroupingSetsSpec, LimitSpec, LiteralValue, SortExpr, SortSpec,
        },
        plan::{
            AggregateNode, DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, DataPlacement, FilterNode, GraphAlgorithm,
            GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams,
            GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs,
            GraphConstructionOptions, GraphConstructionPolicy, GraphCoreNumberParams, GraphDegreesAllParams,
            GraphEcgParams, GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings, GraphEdgePredicateInput,
            GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams, GraphInDegreesAllParams,
            GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLayoutRequirement,
            GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOperationDefinition, GraphOutDegreesAllParams,
            GraphOutputSchemaKind, GraphPageRankParams, GraphPageRankPersonalizationInput,
            GraphPersonalizedPageRankParams, GraphPredicateInput, GraphSccParams, GraphSourceVertexRequirement,
            GraphSpectralClusteringParams, GraphSsspParams, GraphTriangleCountAllParams, GraphVertexIdTypeSupport,
            GraphWccParams, GraphWeightPolicy, HashJoinBuildArtifactId, JoinKey, JoinNode, JoinNullEquality,
            JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitNode, NativePlanCpuPreferenceReason,
            NativePlanPreferenceDecision, NativePlanPreferenceFacts, NativePlanPreferenceOptions, NodeId, PlanFacts,
            PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan, QueryPlanBuilder, ResultSink,
            SchemaBehavior, ShiftDirection, SortNode, WindowFrame, WindowFunction, WindowNode, WindowSpec,
            native_plan_preference_decision, stable_graph_result_schema,
        },
        source::{
            ByteRangeDisposition, CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode,
            LakehouseCacheStatus, LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn,
            ParquetEqualityDeleteFile, ParquetFieldIdReadBinding, ParquetPositionDeleteFile, ParquetReadStrategy,
            ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, PartitionColumn,
            PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature, RemoteParquetReadMode,
            ResolvedParquetSource, ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats,
            SourceColumnRef, SourceDecodeInput, SourceDiagnostics, SourceFormat, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
            ZeroColumnCountDisposition,
        },
    };
}

/// Capability admission, strict bounded, and adaptive admission contracts.
pub mod admission_contracts {
    pub use crate::{
        adaptive_report::{
            ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveAdmissionRuntimeEvidence, AdaptiveCandidateStatus,
            AdaptiveFinalSinkBoundary, AdaptiveMemoryHistoryEvidence, AdaptiveMissingStrictContract,
            AdaptivePipelineReport, AdaptivePipelineReportRow, AdaptiveRuntimeActionCounts,
            AdaptiveRuntimeActionsAllowed, AdaptiveSinkKind, AdaptiveStrategy,
        },
        capability::*,
        contract_report::{
            StrictBoundedContractGateId, StrictBoundedContractGateRow, StrictBoundedContractOwner,
            StrictBoundedContractReport,
        },
        exec::{
            AdaptivePipelineAdmission, StrictBoundedPipelineAdmission, adaptive_bounded_pipeline_admission,
            adaptive_bounded_pipeline_admission_with_memory_history, strict_bounded_pipeline_admission,
            strict_bounded_pipeline_admission_with_statistical_aggregate_mode,
            strict_bounded_pipeline_rejection_projection,
            strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode,
        },
        mode::{
            AdaptiveExecutionMode, NativePipelineExecutionMode, StatisticalAggregateExecutionMode,
            StatisticalAggregateToleranceContract, StddevSampExecutionMode, StrictBoundedExecutionMode,
        },
    };
}

/// Runtime execution, resource, and source IO contracts.
pub mod runtime_contracts {
    pub use crate::{
        adaptive_runtime::{
            AdaptiveGraphMemoryHistoryKey, AdaptiveGraphMemoryHistorySummary, AdaptiveGraphMemoryObservation,
            AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryHistorySummary, AdaptiveMemoryObservation,
            AdaptiveTaskMemoryHistoryKey, AdaptiveTaskMemoryHistoryShape, AdaptiveTaskMemoryHistorySummary,
            AdaptiveTaskMemoryObservation,
        },
        chunk::{
            SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, SourceMetadataCache, SourceTaskMemoryModel,
            plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_execution_context,
            plan_local_parquet_source_chunks_with_metadata_cache,
        },
        exec::{
            GpuDataFrame, GpuHashJoinBuildCache, GpuHashJoinBuildHandle, GpuLocalParquetSourceCache,
            GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap, GpuResultHandle,
            JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape,
            QueryExecutionResult, QueryOutput, QueryOutputMode, ResolvedRemoteParquetDatasourceProbe,
            STRING_OFFSET_BYTES_CONSERVATIVE, execute_query_plan, execute_query_plan_with_gpu_inputs,
            join_output_upper_bound_bytes, materialize_gpu_result_to_host, partial_aggregate_upper_bound_bytes,
            probe_resolved_remote_parquet_datasource, read_gpu_resident_source, read_local_parquet_source,
            validate_local_parquet_source,
        },
        graph_memory::{
            GRAPH_MEMORY_BOUNDARY, GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES,
            GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON, GRAPH_VERTEX_ESTIMATE_BOUNDARY,
            GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES, GraphCalibrationSource, GraphMemoryCalibrationOverride,
            GraphMemoryClass, GraphMemoryEstimate, GraphMemoryModel, GraphMemoryModelInput, GraphModelErrorRatio,
            GraphPlanMemoryEstimate, GraphReservationOutcome, GraphReservationReport, GraphSeedMode, GraphStatistics,
            GraphVertexEstimate, GraphVertexEstimateSource, graph_edge_list_output_rows_for_plan, graph_memory_class,
            graph_memory_estimate_for_edge_list, graph_memory_estimates_for_plan, graph_memory_model_input,
            graph_memory_reservation_required, graph_runtime_statistics_probe_bytes, graph_vertex_estimate_probe_bytes,
            triangle_count_peak_chunk_min_degree_pricing_term,
        },
        runtime::{
            BestFitPlacement, BoundedJoinReservationContract, BoundedJoinReservationRequirement,
            BoundedJoinReservationStatus, BuildMaterializationBoundary, CudfInventory, DEFAULT_GDS_LOCAL_MOUNT_ROOT,
            DEFAULT_GDS_MAX_CONCURRENT_OPS, DEFAULT_GDS_MIN_CHUNK_BYTES, DeploymentIoCapability, DeviceContext,
            DeviceFacts, DeviceMemoryFailureReport, DevicePlacement, DeviceRegistry, DeviceRegistryConfig,
            EvictableMemoryObserver, EvictableMemoryPressure, ExecutionBoundaryPolicy, ExecutionCancellation,
            ExecutionContext, ExecutionContextBuilder, GdsAvailabilityStatus, GdsDriverProbeStatus, GpuComms,
            GpuInventory, GrantRequest, KVIKIO_COMPAT_MODE_ENV, KvikioCompatModeStatus,
            LakehouseCacheFillEventSnapshot, LeastLoadedPlacement, MemoryControlPolicy, MemoryGovernor,
            MemoryGovernorGrantRow, MemoryGovernorReport, MemoryGrant, MemoryReservationRequest, MemoryResourcePolicy,
            MemorySpaceId, NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH, NvidiaFsProbeStatus, PinnedDevicePlacement,
            Placement, PlacementPolicy, PlacementRequest, Priority, ProbeExecutionBoundary, QueryAdmissionWaiter,
            QueryGrantBand, QueryGrantRequest, QueryHandle, QueryPlacementPolicy, QueryQos, QueryService,
            QueryServiceAdmissionSnapshot, QueryServiceConfig, QueryServiceTenantConfig, ReservationKind,
            ResultMaterializationBoundary, RetryFacts, S3SourceStorageCredential, S3SourceStorageProfile,
            ServiceStreamPool, SingleDeviceComms, SlotGuarantee, SourceChunkPlanningMode, SourceFetchConfig,
            SourceFetchStrategy, SourceFetchStrategyDecision, SourceFetchStrategyReason, SourceFetchSubstrate,
            SourceGdsEligibility, SourceReadBoundary, SourceSchedulerPolicy, SourceStorageProfile, StreamLease,
            TenantId, TenantQueryContext, TenantQueueDepth, device_budget_bytes_from_facts, source_gds_eligibility,
            source_gds_eligibility_with_default_mounts,
        },
    };
}

/// Metrics and report evidence contracts.
pub mod metrics_contracts {
    pub use crate::metrics::{
        AggregateStrategyDecisionMetrics, BoundaryMetrics, BudgetedExecutionEvent, CompiledTaskId,
        DynamicFilterRuntimeMetrics, DynamicFilterSourceSnapshot, GraphAlgorithmRuntimeMetrics, JoinOperatorMetrics,
        JoinStrategyDecisionMetrics, MemoryReservationEvent, MemoryReservationEventAction,
        MemoryReservationEventOutcome, MetricAggregation, MetricDescriptor, MetricDirection, MetricUnit,
        OperatorMetrics, PlanNodeId, PlanTaskMapping, QueryId, QueryMetrics, SourceReadParquetEvidenceMetrics,
        SourceReadRuntimeMetrics, TaskOperatorMetrics, TaskRuntimeMetrics, TraceId,
    };
}

/// Compact triage verdict and signal contracts.
pub mod triage_contracts {
    pub use crate::triage::{
        TriageCode, TriageEvidenceRef, TriageSeverity, TriageSignal, TriageSummary, TriageVerdict,
    };
}

/// Native optimizer rule contracts.
pub mod optimizer_contracts {
    pub use crate::{
        dynamic_filter::{
            DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage,
            DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterKind, DynamicFilterPayload,
            DynamicFilterProducer, DynamicFilterReadiness, DynamicFilterRoute, DynamicFilterSet,
            LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON,
        },
        optimizer::{
            AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics,
            AggregateJoinPushdownDiagnostics, AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
            COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME, CountNonNullColumnPruningCandidateDiagnostics,
            CountNonNullColumnPruningDiagnostics, CountNonNullColumnPruningOptions,
            CountNonNullColumnPruningRejectedReason, DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER,
            DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, DIMENSION_KEY_FILTER_MAX_RUNS,
            DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedApplyResult, DimensionDerivedPruningRequest,
            DimensionDerivedRowGroupPruningCandidateDiagnostics, DimensionDerivedRowGroupPruningDiagnostics,
            DimensionDerivedRowGroupPruningOptions, DimensionDerivedRowGroupPruningRejectedReason,
            DimensionKeyAccumulator, DimensionKeyFilter, DynamicFilterRouteCandidateDiagnostics,
            DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions,
            DynamicFilterRouteRejectedReason, JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME,
            JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics,
            JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
            JoinFilterReassociationCandidateDiagnostics, JoinFilterReassociationDiagnostics,
            JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
            KeyRange as DimensionDerivedKeyRange, LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetRowGroupSelection,
            LocalParquetRowGroupSelectionApplyResult, LocalParquetSourceReuseCandidateDiagnostics,
            LocalParquetSourceReuseDiagnostics, LocalParquetSourceReuseOptions, LocalParquetSourceReuseRejectedReason,
            NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, NativeOptimizerResult,
            OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionCandidateDiagnostics,
            OuterJoinNullRejectionDiagnostics, OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason,
            PROJECTION_SIMPLIFICATION_RULE_NAME, ProjectionSimplificationCandidateDiagnostics,
            ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions,
            ProjectionSimplificationRejectedReason, RowGroupOverlapSelection, SEMIJOIN_REDUCTION_RULE_NAME,
            SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME,
            SemijoinReductionAdmissionDecision, SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics,
            SemijoinReductionCostGateDiagnostics, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
            SemijoinReductionKeyMappingDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
            SemijoinReductionResult, SharedSubplanMaterializationCandidateDiagnostics,
            SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions,
            SharedSubplanMaterializationRejectedReason, StreamFanoutReplayCandidateDiagnostics,
            StreamFanoutReplayDiagnostics, StreamFanoutReplayOptions, StreamFanoutReplayRejectedReason,
            apply_dimension_derived_pruning_with_stats, apply_local_parquet_row_group_selections,
            dimension_key_filter_from_batch, discover_dimension_derived_pruning_requests, optimize_query_plan,
            optimize_semijoin_reduction, seed_dimension_derived_reader_predicates, select_overlapping_row_groups,
        },
    };
}

/// Compiled pipeline graph contracts.
pub mod pipeline_contracts {
    pub use crate::pipeline::{
        CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
        PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
        ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
        compile_pipeline,
    };
}

pub use adaptive_report::{
    ADAPTIVE_DIAGNOSTIC_CONSTRUCTION_FAILED_REASON, AdaptiveAdmissionRuntimeEvidence, AdaptiveCandidateStatus,
    AdaptiveFinalSinkBoundary, AdaptiveMemoryHistoryEvidence, AdaptiveMissingStrictContract, AdaptivePipelineReport,
    AdaptivePipelineReportRow, AdaptiveRuntimeActionCounts, AdaptiveRuntimeActionsAllowed, AdaptiveSinkKind,
    AdaptiveStrategy,
};
pub use adaptive_runtime::{
    AdaptiveGraphMemoryHistoryKey, AdaptiveGraphMemoryHistorySummary, AdaptiveGraphMemoryObservation,
    AdaptiveMemoryHistory, AdaptiveMemoryHistoryKey, AdaptiveMemoryHistorySummary, AdaptiveMemoryObservation,
    AdaptiveTaskMemoryHistoryKey, AdaptiveTaskMemoryHistoryShape, AdaptiveTaskMemoryHistorySummary,
    AdaptiveTaskMemoryObservation,
};
pub use capability::{AggregateStrategyExecutionOptions, KeyedHashJoinStrategyOptions};
pub use chunk::{
    SourceChunk, SourceChunkPlan, SourceChunkPlanningOptions, SourceChunkPlanningPolicy, SourceMetadataCache,
    SourceTaskMemoryModel, plan_local_parquet_source_chunks, plan_local_parquet_source_chunks_with_execution_context,
    plan_local_parquet_source_chunks_with_metadata_cache,
};
pub use contract_report::{
    StrictBoundedContractGateId, StrictBoundedContractGateRow, StrictBoundedContractOwner, StrictBoundedContractReport,
};
pub use datatypes::{
    arrow_decimal_scale_to_cudf_scale, cudf_dtype_from_arrow_type, diagnostic_arrow_type, is_supported_arrow_cast,
    is_supported_arrow_implicit_cast, is_supported_arrow_type, is_supported_list_carry_type,
    is_timestamp_with_timezone, timestamp_unit_to_cudf_resolution,
};
pub use dynamic_filter::{
    DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage,
    DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterKind, DynamicFilterPayload, DynamicFilterProducer,
    DynamicFilterReadiness, DynamicFilterRoute, DynamicFilterSet, LARGE_KEYSET_DIMENSION_SOURCE_SKIP_REASON,
};
pub use error::{Facts, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus, Result};
pub use exec::{
    AdaptivePipelineAdmission, GpuDataFrame, GpuHashJoinBuildCache, GpuHashJoinBuildHandle, GpuLocalParquetSourceCache,
    GpuLocalParquetSourceCacheInsert, GpuLocalParquetSourceCacheKey, GpuResidentInputMap, GpuResultHandle,
    JoinOutputColumnUpperBoundInput, PER_COLUMN_ALIGNMENT_PADDING_BYTES, PartialAggregateShape, QueryExecutionResult,
    QueryOutput, QueryOutputMode, ResolvedRemoteParquetDatasourceProbe, STRING_OFFSET_BYTES_CONSERVATIVE,
    StrictBoundedPipelineAdmission, adaptive_bounded_pipeline_admission,
    adaptive_bounded_pipeline_admission_with_memory_history, execute_query_plan, execute_query_plan_with_gpu_inputs,
    join_output_upper_bound_bytes, materialize_gpu_result_to_host, partial_aggregate_upper_bound_bytes,
    probe_resolved_remote_parquet_datasource, read_gpu_resident_source, read_local_parquet_source,
    strict_bounded_pipeline_admission, strict_bounded_pipeline_admission_with_statistical_aggregate_mode,
    strict_bounded_pipeline_rejection_projection,
    strict_bounded_pipeline_rejection_projection_with_statistical_aggregate_mode, validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub use exec::{
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test,
    force_next_graph_statistics_optional_resource_exhausted_for_test,
};
pub use expr::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnQualifier, ColumnRef, DatePart, Expr,
    GroupingSetsSpec, LimitSpec, LiteralValue, SortExpr, SortSpec,
};
pub use graph_memory::{
    GRAPH_MEMORY_BOUNDARY, GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES, GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON,
    GRAPH_VERTEX_ESTIMATE_BOUNDARY, GRAPH_VERTEX_ESTIMATE_PROBE_MIN_BYTES, GraphCalibrationSource,
    GraphMemoryCalibrationOverride, GraphMemoryClass, GraphMemoryEstimate, GraphMemoryModel, GraphMemoryModelInput,
    GraphModelErrorRatio, GraphPlanMemoryEstimate, GraphReservationOutcome, GraphReservationReport, GraphSeedMode,
    GraphStatistics, GraphVertexEstimate, GraphVertexEstimateSource, graph_edge_list_output_rows_for_plan,
    graph_memory_class, graph_memory_estimate_for_edge_list, graph_memory_estimates_for_plan, graph_memory_model_input,
    graph_memory_reservation_required, graph_runtime_statistics_probe_bytes, graph_vertex_estimate_probe_bytes,
    triangle_count_peak_chunk_min_degree_pricing_term,
};
pub use metrics::{
    AggregateStrategyDecisionMetrics, BoundaryMetrics, BudgetedExecutionEvent, CompiledTaskId,
    DynamicFilterRuntimeMetrics, DynamicFilterSourceSnapshot, GraphAlgorithmRuntimeMetrics, JoinOperatorMetrics,
    JoinStrategyDecisionMetrics, MemoryReservationEvent, MemoryReservationEventAction, MemoryReservationEventOutcome,
    MetricAggregation, MetricDescriptor, MetricDirection, MetricUnit, OperatorMetrics, PlanNodeId, PlanTaskMapping,
    QueryId, QueryMetrics, SourceReadParquetEvidenceMetrics, SourceReadRuntimeMetrics, TaskOperatorMetrics,
    TaskRuntimeMetrics, TraceId,
};
pub use mode::{
    AdaptiveExecutionMode, NativePipelineExecutionMode, StatisticalAggregateExecutionMode,
    StatisticalAggregateToleranceContract, StddevSampExecutionMode, StrictBoundedExecutionMode,
};
pub use optimizer::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason, COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
    CountNonNullColumnPruningCandidateDiagnostics, CountNonNullColumnPruningDiagnostics,
    CountNonNullColumnPruningOptions, CountNonNullColumnPruningRejectedReason,
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
    DIMENSION_KEY_FILTER_MAX_RUNS, DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedApplyResult,
    DimensionDerivedPruningRequest, DimensionDerivedRowGroupPruningCandidateDiagnostics,
    DimensionDerivedRowGroupPruningDiagnostics, DimensionDerivedRowGroupPruningOptions,
    DimensionDerivedRowGroupPruningRejectedReason, DimensionKeyAccumulator, DimensionKeyFilter,
    DynamicFilterRouteCandidateDiagnostics, DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions,
    DynamicFilterRouteRejectedReason, JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME,
    JoinBuildSideSelectionCandidateDiagnostics, JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions,
    JoinBuildSideSelectionRejectedReason, JoinFilterReassociationCandidateDiagnostics,
    JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
    KeyRange as DimensionDerivedKeyRange, LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetRowGroupSelection,
    LocalParquetRowGroupSelectionApplyResult, LocalParquetSourceReuseCandidateDiagnostics,
    LocalParquetSourceReuseDiagnostics, LocalParquetSourceReuseOptions, LocalParquetSourceReuseRejectedReason,
    NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, NativeOptimizerResult,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionCandidateDiagnostics, OuterJoinNullRejectionDiagnostics,
    OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason, PROJECTION_SIMPLIFICATION_RULE_NAME,
    ProjectionSimplificationCandidateDiagnostics, ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions,
    ProjectionSimplificationRejectedReason, RowGroupOverlapSelection, SEMIJOIN_REDUCTION_RULE_NAME,
    SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME, SemijoinReductionAdmissionDecision,
    SemijoinReductionAdmissionMode, SemijoinReductionCandidateDiagnostics, SemijoinReductionCostGateDiagnostics,
    SemijoinReductionCostSignal, SemijoinReductionDiagnostics, SemijoinReductionKeyMappingDiagnostics,
    SemijoinReductionOptions, SemijoinReductionRejectedReason, SemijoinReductionResult,
    SharedSubplanMaterializationCandidateDiagnostics, SharedSubplanMaterializationDiagnostics,
    SharedSubplanMaterializationOptions, SharedSubplanMaterializationRejectedReason,
    StreamFanoutReplayCandidateDiagnostics, StreamFanoutReplayDiagnostics, StreamFanoutReplayOptions,
    StreamFanoutReplayRejectedReason, apply_dimension_derived_pruning_with_stats,
    apply_local_parquet_row_group_selections, dimension_key_filter_from_batch,
    discover_dimension_derived_pruning_requests, optimize_query_plan, optimize_semijoin_reduction,
    seed_dimension_derived_reader_predicates, select_overlapping_row_groups,
};
pub use pipeline::{
    CompiledPipeline, PipelineBarrier, PipelineBarrierId, PipelineBarrierKind, PipelineEdge, PipelineEdgeId,
    PipelineEdgeKind, PipelineTask, PipelineTaskId, PipelineTaskKind, ReservationGate, ReservationGateId,
    ReservationGateOwner, ResidentLifetime, ResidentResource, ResidentResourceId, SinkBoundary, SinkBoundaryKind,
    compile_pipeline,
};
pub use plan::{
    AggregateNode, DEFAULT_BETWEENNESS_EXACT_VERTEX_THRESHOLD, DataPlacement, FilterNode,
    GRAPH_ALGORITHM_STRICT_BOUNDED_LOWER_LEVEL_CONTRACTS, GRAPH_ALGORITHM_STRICT_BOUNDED_MEMORY_BOUNDARY,
    GRAPH_ALGORITHM_STRICT_BOUNDED_REQUIRED_CONTRACT, GRAPH_OPERATION_REGISTRY, GraphAlgorithm, GraphAlgorithmNode,
    GraphAlgorithmParams, GraphAlgorithmSpec, GraphAlgorithmStrictBoundedLowerLevelContract,
    GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
    GraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy, GraphCoreNumberParams,
    GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams, GraphEdgeColumnBindings,
    GraphEdgePredicateInput, GraphEigenvectorCentralityParams, GraphForceAtlas2Params, GraphHitsParams,
    GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams, GraphLayoutRequirement,
    GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOperationDefinition, GraphOutDegreesAllParams,
    GraphOutputSchemaKind, GraphPageRankParams, GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams,
    GraphPredicateInput, GraphSccParams, GraphSourceVertexRequirement, GraphSpectralClusteringParams, GraphSsspParams,
    GraphTriangleCountAllParams, GraphVertexIdTypeSupport, GraphWccParams, GraphWeightPolicy, HashJoinBuildArtifactId,
    JoinKey, JoinNode, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitNode,
    NativePlanCpuPreferenceReason, NativePlanPreferenceDecision, NativePlanPreferenceFacts,
    NativePlanPreferenceOptions, NodeId, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput, QueryNode, QueryPlan,
    QueryPlanBuilder, ResultSink, SchemaBehavior, ShiftDirection, SortNode, WindowFrame, WindowFunction, WindowNode,
    WindowSpec, native_plan_preference_decision, stable_graph_result_schema,
};
pub use runtime::{
    BestFitPlacement, BoundedJoinReservationContract, BoundedJoinReservationRequirement, BoundedJoinReservationStatus,
    BuildMaterializationBoundary, CudfInventory, DEFAULT_GDS_LOCAL_MOUNT_ROOT, DEFAULT_GDS_MAX_CONCURRENT_OPS,
    DEFAULT_GDS_MIN_CHUNK_BYTES, DeploymentIoCapability, DeviceContext, DeviceFacts, DeviceMemoryFailureReport,
    DevicePlacement, DeviceRegistry, DeviceRegistryConfig, EvictableMemoryObserver, EvictableMemoryPressure,
    ExecutionBoundaryPolicy, ExecutionCancellation, ExecutionContext, ExecutionContextBuilder, GdsAvailabilityStatus,
    GdsDriverProbeStatus, GpuComms, GpuInventory, GrantRequest, KVIKIO_COMPAT_MODE_ENV, KvikioCompatModeStatus,
    LakehouseCacheFillEventSnapshot, LeastLoadedPlacement, MemoryControlPolicy, MemoryGovernor, MemoryGovernorGrantRow,
    MemoryGovernorReport, MemoryGrant, MemoryReservationRequest, MemoryResourcePolicy, MemorySpaceId,
    NVIDIA_FS_PROC_PATH, NVIDIA_FS_SYSFS_PATH, NvidiaFsProbeStatus, PinnedDevicePlacement, Placement, PlacementPolicy,
    PlacementRequest, Priority, ProbeExecutionBoundary, QueryAdmissionWaiter, QueryGrantBand, QueryGrantRequest,
    QueryHandle, QueryPlacementPolicy, QueryQos, QueryService, QueryServiceAdmissionSnapshot, QueryServiceConfig,
    QueryServiceTenantConfig, ReservationKind, ResultMaterializationBoundary, RetryFacts, S3SourceStorageCredential,
    S3SourceStorageProfile, ServiceStreamPool, SingleDeviceComms, SlotGuarantee, SourceChunkPlanningMode,
    SourceFetchConfig, SourceFetchStrategy, SourceFetchStrategyDecision, SourceFetchStrategyReason,
    SourceFetchSubstrate, SourceGdsEligibility, SourceReadBoundary, SourceSchedulerPolicy, SourceStorageProfile,
    StreamLease, TenantId, TenantQueryContext, TenantQueueDepth, device_budget_bytes_from_facts,
    source_gds_eligibility, source_gds_eligibility_with_default_mounts,
};
pub use source::{
    ByteRangeDisposition, CredentialMode, GpuResidentSource, LakehouseCacheFillMode, LakehouseCacheReadMode,
    LakehouseCacheStatus, LiteralBatchSource, LocalParquetFile, LocalParquetSource, MissingColumn,
    ParquetEqualityDeleteFile, ParquetFieldIdReadBinding, ParquetPositionDeleteFile, ParquetReadStrategy,
    ParquetReaderDecimalType, ParquetReaderDecimalTypeId, ParquetSourceLocation, PartitionColumn,
    PartitionColumnDisposition, PredicateDisposition, RejectedSourceFeature, RemoteParquetReadMode,
    ResolvedParquetSource, ScanLimitDisposition, SchemaMappingDisposition, SelectedRowGroupStats, SourceColumnRef,
    SourceDecodeInput, SourceDiagnostics, SourceFormat, SourceNode, ZERO_COLUMN_COUNT_COLUMN,
    ZeroColumnCountDisposition,
};
pub use triage::{TriageCode, TriageEvidenceRef, TriageSeverity, TriageSignal, TriageSummary, TriageVerdict};
