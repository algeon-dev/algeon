use nexus_query_engine::{
    capability::KeyedHashJoinStrategyOptions,
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage,
        DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterKind, DynamicFilterProducer, DynamicFilterRoute,
    },
    error::QueryErrorCode,
    metrics::DynamicFilterSourceSnapshot,
    optimizer::{
        AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
        AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason, COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
        CountNonNullColumnPruningCandidateDiagnostics, CountNonNullColumnPruningDiagnostics,
        CountNonNullColumnPruningOptions, CountNonNullColumnPruningRejectedReason,
        DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
        DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedRowGroupPruningDiagnostics,
        DimensionDerivedRowGroupPruningOptions, DimensionDerivedRowGroupPruningRejectedReason,
        DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions, DynamicFilterRouteRejectedReason,
        JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinBuildSideSelectionBasis,
        JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
        JoinFilterReassociationDiagnostics, JoinFilterReassociationRejectedReason,
        LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetSourceReuseCandidateDiagnostics,
        LocalParquetSourceReuseDiagnostics, LocalParquetSourceReuseOptions, LocalParquetSourceReuseRejectedReason,
        NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, OUTER_JOIN_NULL_REJECTION_RULE_NAME,
        OuterJoinNullRejectionDiagnostics, OuterJoinNullRejectionRejectedReason, PROJECTION_SIMPLIFICATION_RULE_NAME,
        ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions, SEMIJOIN_REDUCTION_RULE_NAME,
        SHARED_RESIDENT_EXTRACTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME,
        SemijoinReductionAdmissionMode, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
        SemijoinReductionOptions, SemijoinReductionRejectedReason, SharedResidentExtractionDecision,
        SharedResidentExtractionDiagnostics, SharedResidentExtractionReason, SharedResidentPlan,
        SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions, StreamFanoutReplayDiagnostics,
        StreamFanoutReplayOptions, apply_dimension_derived_pruning_with_stats,
        discover_dimension_derived_pruning_requests, local_parquet_source,
        optimize_dimension_derived_row_group_pruning, optimize_query_plan, optimize_semijoin_reduction,
        optimize_shared_resident_extraction, seed_dimension_derived_reader_predicates,
    },
    plan::{
        AdvisoryOutputStatistics, AdvisoryStatistic, AdvisoryStatisticsProvenance, GraphAlgorithm,
        GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs,
        GraphConstructionOptions, GraphEdgeColumnBindings, GraphPredicateInput, NodeId,
    },
    rapids_backend::rapids_analysis_context,
    source::LocalParquetSource,
};

use super::*;
// Optimizer contract tests are split by optimizer pass while staying in this
// integration-test module via include! so existing test names remain stable.
include!("optimizer_contracts/support.rs");
include!("optimizer_contracts/options.rs");
include!("optimizer_contracts/diagnostics.rs");
include!("optimizer_contracts/aggregate_join_pushdown.rs");
include!("optimizer_contracts/count_non_null_pruning.rs");
include!("optimizer_contracts/shared_subplan_materialization.rs");
include!("optimizer_contracts/dynamic_filter.rs");
include!("optimizer_contracts/dimension_pruning.rs");
include!("optimizer_contracts/local_parquet_source_reuse.rs");
include!("optimizer_contracts/stream_fanout_replay.rs");
include!("optimizer_contracts/outer_join_null_rejection.rs");
include!("optimizer_contracts/join_filter_reassociation.rs");
include!("optimizer_contracts/join_build_side.rs");
include!("optimizer_contracts/semijoin_reduction.rs");
include!("optimizer_contracts/shared_resident_extraction.rs");
