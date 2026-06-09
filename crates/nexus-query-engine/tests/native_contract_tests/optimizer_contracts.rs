use nexus_query_engine::{
    AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
    AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason,
    DEFAULT_SEMIJOIN_REDUCTION_COST_THRESHOLD_MULTIPLIER, DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME,
    DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME, DimensionDerivedRowGroupPruningDiagnostics,
    DimensionDerivedRowGroupPruningOptions, DimensionDerivedRowGroupPruningRejectedReason, DynamicFilter,
    DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage, DynamicFilterKind,
    DynamicFilterProducer, DynamicFilterRouteDiscoveryDiagnostics, DynamicFilterRouteDiscoveryOptions,
    DynamicFilterRouteRejectedReason, DynamicFilterSourceSnapshot, GraphAlgorithm, GraphAlgorithmParams,
    GraphAlgorithmSpec, GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions, GraphEdgeColumnBindings,
    GraphPredicateInput, JOIN_BUILD_SIDE_SELECTION_RULE_NAME, JOIN_FILTER_REASSOCIATION_RULE_NAME,
    JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions, JoinFilterReassociationDiagnostics,
    LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetSource, LocalParquetSourceReuseDiagnostics,
    LocalParquetSourceReuseOptions, NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, NodeId,
    OUTER_JOIN_NULL_REJECTION_RULE_NAME, OuterJoinNullRejectionDiagnostics, OuterJoinNullRejectionRejectedReason,
    PROJECTION_SIMPLIFICATION_RULE_NAME, ProjectionSimplificationDiagnostics, ProjectionSimplificationOptions,
    SEMIJOIN_REDUCTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME,
    SemijoinReductionAdmissionMode, SemijoinReductionCostSignal, SemijoinReductionDiagnostics,
    SemijoinReductionOptions, SemijoinReductionRejectedReason, SharedSubplanMaterializationDiagnostics,
    SharedSubplanMaterializationOptions, StreamFanoutReplayDiagnostics, StreamFanoutReplayOptions,
    apply_dimension_derived_pruning_with_stats, discover_dimension_derived_pruning_requests, optimize_query_plan,
    seed_dimension_derived_reader_predicates,
};

use super::*;
// Optimizer contract tests are split by optimizer pass while staying in this
// integration-test module via include! so existing test names remain stable.
include!("optimizer_contracts/support.rs");
include!("optimizer_contracts/options.rs");
include!("optimizer_contracts/diagnostics.rs");
include!("optimizer_contracts/aggregate_join_pushdown.rs");
include!("optimizer_contracts/shared_subplan_materialization.rs");
include!("optimizer_contracts/dynamic_filter.rs");
include!("optimizer_contracts/dimension_pruning.rs");
include!("optimizer_contracts/local_parquet_source_reuse.rs");
include!("optimizer_contracts/stream_fanout_replay.rs");
include!("optimizer_contracts/outer_join_null_rejection.rs");
include!("optimizer_contracts/join_filter_reassociation.rs");
include!("optimizer_contracts/join_build_side.rs");
include!("optimizer_contracts/semijoin_reduction.rs");
