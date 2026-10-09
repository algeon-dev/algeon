use std::collections::BTreeMap;

use arrow::array::{Array, Date32Array, Float64Array, Int64Array, StringArray, StringViewArray};
use dimension_pruning::{unique_contract_temp_dir, write_i32_record_batch_parquet, write_record_batch_parquet};
use join_build_side::{advisory_rows, root_join_build_side};
use local_parquet_source_reuse::{
    assert_source_input, cache_policy_customer_address_contract, gpu_source, local_parquet_source_count_for_stable_id,
    remote_direct_customer_address_contract,
};
use query_engine::{
    dimension_pruning::local_dimension_pruning_resolver,
    exec::{admit_query_plan, preflight_query_plan},
    rapids_backend::rapids_analysis_context,
};
use query_planning::{
    dynamic_filter::{
        DynamicFilter, DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage,
        DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterKind, DynamicFilterProducer, DynamicFilterRoute,
    },
    error::QueryErrorCode,
    expr::GroupingSetsSpec,
    mode::StatisticalAggregateExecutionMode,
    optimizer::{
        AGGREGATE_JOIN_PUSHDOWN_RULE_NAME, AggregateJoinPushdownCandidateDiagnostics, AggregateJoinPushdownDiagnostics,
        AggregateJoinPushdownOptions, AggregateJoinPushdownRejectedReason, COUNT_NON_NULL_COLUMN_PRUNING_RULE_NAME,
        CountNonNullColumnPruningCandidateDiagnostics, CountNonNullColumnPruningDiagnostics,
        CountNonNullColumnPruningOptions, CountNonNullColumnPruningRejectedReason,
        DIMENSION_DERIVED_ROW_GROUP_PRUNING_RULE_NAME, DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
        DimensionDerivedRowGroupPruningDiagnostics, DimensionDerivedRowGroupPruningOptions,
        DimensionDerivedRowGroupPruningRejectedReason, DimensionKeyFilter, DynamicFilterRouteDiscoveryDiagnostics,
        DynamicFilterRouteDiscoveryOptions, DynamicFilterRouteRejectedReason, JOIN_BUILD_SIDE_SELECTION_RULE_NAME,
        JOIN_FILTER_REASSOCIATION_RULE_NAME, JoinBuildSideSelectionBasis, JoinBuildSideSelectionCandidateDiagnostics,
        JoinBuildSideSelectionDiagnostics, JoinBuildSideSelectionOptions, JoinBuildSideSelectionRejectedReason,
        JoinFilterReassociationDiagnostics, JoinFilterReassociationOptions, JoinFilterReassociationRejectedReason,
        KeyRange, LOCAL_PARQUET_SOURCE_REUSE_RULE_NAME, LocalParquetSourceReuseCandidateDiagnostics,
        LocalParquetSourceReuseDiagnostics, LocalParquetSourceReuseOptions, LocalParquetSourceReuseRejectedReason,
        NativeOptimizerChange, NativeOptimizerDiagnostics, NativeOptimizerOptions, OUTER_JOIN_NULL_REJECTION_RULE_NAME,
        OuterJoinNullRejectionDiagnostics, OuterJoinNullRejectionOptions, OuterJoinNullRejectionRejectedReason,
        PROJECTION_SIMPLIFICATION_RULE_NAME, ProjectionSimplificationDiagnostics, SEMIJOIN_REDUCTION_RULE_NAME,
        SHARED_RESIDENT_EXTRACTION_RULE_NAME, SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
        SINGLE_DISTINCT_AGGREGATE_SPLIT_RULE_NAME, STREAM_FANOUT_REPLAY_RULE_NAME, SemijoinReductionCostSignal,
        SemijoinReductionDiagnostics, SemijoinReductionOptions, SemijoinReductionRejectedReason,
        SharedResidentExtractionDecision, SharedResidentExtractionDiagnostics, SharedResidentExtractionReason,
        SharedResidentPlan, SharedSubplanMaterializationDiagnostics, SharedSubplanMaterializationOptions,
        SharedSubplanMaterializationRejectedReason, SingleDistinctAggregateSplitDiagnostics,
        SingleDistinctAggregateSplitOptions, StreamFanoutReplayDiagnostics, apply_dimension_derived_pruning_with_stats,
        dimension_key_filter_from_batch, discover_dimension_derived_pruning_requests, local_parquet_source,
        optimize_query_plan, seed_dimension_derived_reader_predicates,
    },
    plan::{
        AdvisoryOutputStatistics, AdvisoryStatistic, GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec,
        GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs, GraphConstructionOptions,
        GraphEdgeColumnBindings, GraphPredicateInput, NodeId, QueryPlan,
    },
    policy::PlanningPolicy,
    source::LocalParquetSource,
};
use query_runtime::metrics::DynamicFilterSourceSnapshot;
use shared_subplan_materialization::{
    assert_no_non_decimal_binary_output_types, estimated_sales_source_with_quarter_predicate,
    nested_reader_predicate_with_declared_boolean_output_types,
};
use support::{
    aggregate_keyset_source, assert_all_node_inputs_exist, between_i32, estimated_inner_join_plan,
    estimated_inner_residual_join_plan, estimated_local_source, estimated_local_source_with_fields,
    estimated_local_source_with_row_group_count, estimated_local_source_with_stable_id,
    estimated_local_source_with_stable_id_and_file_indices, estimated_local_source_with_stable_id_and_predicate,
    estimated_source_with_resolved_contract, expr_contains_int32_equality, left_projection,
    max_direct_join_probe_consumer_count, plan_has_output_name_prefix, schema, single_source_plan, source_input_name,
};

use super::*;

mod aggregate_join_pushdown;
mod count_non_null_pruning;
mod diagnostics;
mod dimension_pruning;
mod dynamic_filter;
mod join_build_side;
mod join_filter_reassociation;
mod local_parquet_source_reuse;
mod options;
mod outer_join_null_rejection;
mod semijoin_reduction;
mod shared_resident_extraction;
mod shared_subplan_materialization;
mod single_distinct_aggregate_split;
mod stream_fanout_replay;
mod support;
