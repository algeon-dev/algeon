//! One rejection vocabulary for capability analysis, compiled pipeline
//! contracts, plan validation, and admission.
//!
//! The reason alone decides a rejection's category and, for pipeline,
//! validation, and execution-mode refusals, its boundary; the owner decides
//! its scope. Facts are diagnostics only: adding one never reclassifies a
//! rejection.

use std::{collections::BTreeMap, fmt};

use super::{CapabilityRejectionCategory, CapabilityScope, EdgeConsumer, InvariantId, NodeId, ResultSinkKind};
use crate::error::QueryErrorCode;

/// The part of an operator the backend cannot execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnsupportedPart {
    Operator,
    DataType,
    Expression,
}

impl UnsupportedPart {
    /// The part a validator's typed refusal names.
    pub const fn from_code(code: QueryErrorCode) -> Self {
        match code {
            QueryErrorCode::UnsupportedType => Self::DataType,
            QueryErrorCode::UnsupportedExpression => Self::Expression,
            _ => Self::Operator,
        }
    }

    const fn category(self) -> CapabilityRejectionCategory {
        match self {
            Self::Operator => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::DataType => CapabilityRejectionCategory::UnsupportedArrowDtype,
            Self::Expression => CapabilityRejectionCategory::UnsupportedExpressionFamily,
        }
    }
}

/// Why a capability, edge, invariant, compiled contract, or admission gate
/// refuses a plan. `as_str` is the report label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    UnsupportedAggregateGroupingSetsCapability,
    UnsupportedAggregateNodeCapability,
    UnsupportedGpuResidentSourceCapability,
    UnsupportedJoinNodeCapability,
    UnsupportedLimitNodeCapability,
    UnsupportedProjectionNodeCapability,
    UnsupportedSortNodeCapability,
    UnsupportedUnionNodeCapability,
    UnsupportedWindowNodeCapability,
    PostJoinResidualFilterRequiresOneShotEquiJoin,
    LargeKeyedHashJoinStrategyUnsupported,
    UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart),
    UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart),
    UnsupportedVectorAlgorithmNodeCapability(UnsupportedPart),
    UnsupportedMlAlgorithmNodeCapability(UnsupportedPart),
    UnsupportedAggregateExpressionCapability,
    UnsupportedFilterExpressionCapability,
    UnsupportedProjectionExpressionCapability,
    UnsupportedAggregateGroupExpressionCapability,
    UnsupportedJoinResidualExpressionCapability,
    UnsupportedSortExpressionCapability,
    UnsupportedWindowAggregateExpressionCapability,
    UnsupportedWindowLagLeadExpressionCapability,
    UnsupportedWindowOrderExpressionCapability,
    UnsupportedWindowPartitionExpressionCapability,
    UnsupportedLiteralBatchSourceCapability,
    UnsupportedLocalParquetSourceCapability,
    UnsupportedJoinOutputDtype,
    LeftAntiBuildRequiresDedupAggregate,
    LeftAntiProbeRequiresDedupProvenInput,
    MissingEdgeProducer,
    MissingEdgeRequiredProperty,
    AggregateSharedProducerConsumerNotAggregate,
    AggregateSharedProducerInputLineageUnresolved,
    AggregateSharedProducerMissingAggregate,
    AggregateSharedProducerMissingNode,
    AggregateSharedProducerRequiresDataflowIdentity,
    AggregateSharedProducerRequiresInputLineage,
    AggregateSharedProducerRequiresResidentFrame,
    BuildBeforeProbeCycle,
    ExpressionLineageRequiresResolvedInputColumn,
    OutputLineageAggregateFieldIndexOutOfRange,
    OutputLineageFieldIndexOutOfRange,
    OutputLineageJoinOutputColumnResolution,
    OutputLineageJoinProjectionIndexOutOfRange,
    OutputLineageMissingNode,
    OutputLineageProjectionColumnResolution,
    OutputLineageProjectionFieldIndexOutOfRange,
    OutputLineageWindowMissingInput,
    ProjectionTransparentResidencyConsumerNotProjection,
    ProjectionTransparentResidencyMissingProducer,
    ProjectionTransparentResidencyMissingProjection,
    ProjectionTransparentResidencyRequiresResidentFrames,
    ReachableSubgraphCycle,
    ReachableSubgraphMissingNode,
    ResidentLifetimeEdgeRejected,
    ResidentLifetimeMissingNode,
    ResidentLifetimeRequiresResidentFrame,
    ResidentUnaryChainMissingNode,
    ResidentUnaryChainMissingProducer,
    ResidentUnaryChainRequiresConnectedNodes,
    ResidentUnaryChainRequiresFilterProjectionNodes,
    ResidentUnaryChainRequiresResidentNodes,
    ResidentUnaryChainRequiresResidentProducer,
    ResidualDoesNotReferenceBothInputs,
    SharedResidentSubplanEdgeRejected,
    SharedResidentSubplanMissingEdge,
    SharedResidentSubplanMissingNode,
    SharedResidentSubplanRequiresResidentFrame,
    TopologicalScheduleCycle,
    TopologicalScheduleMissingNode,
    CompiledPipelineRequiresJoinTaskGroup,
    CompiledPipelineJoinTaskGroupIncomplete,
    CompiledPipelineSharedProducerTaskKindUnsupported,
    CompiledPipelineStageTaskPositionUnsupported,
    CompiledPipelineSharedProducerLineageUnproven,
    CompiledPipelineJoinTaskGroupPlanNodeMismatch,
    CompiledPipelineJoinExecutionStrategyUnavailable,
    CompiledTaskKindOutsideContract,
    CompiledPipelineGraphNotConstructible,
    ConsumerRequiresExclusiveInput,
    CompiledPipelineTaskIdentityInvalid,
    CompiledPipelineEdgeIdentityInvalid,
    CompiledPipelineBarrierIdentityInvalid,
    CompiledPipelineResourceIdentityInvalid,
    CompiledPipelineTaskDependencyMissing,
    CompiledPipelineDependencyCycle,
    CompiledPipelineResourceReferenceMissing,
    CompiledPipelineResourceFlowInvalid,
    CompiledPipelineSinkOutputLifetimeInvalid,
    CompiledPipelineBarrierDominanceInvalid,
    CompiledPipelineTaskResourceArityInvalid,
    CompiledPipelineStageBoundaryIncomplete,
    CompiledPipelineJoinBuildBeforeProbeMissing,
    CompiledPipelineFusedTopkLimitBindingInvalid,
    CompiledPipelineSinkInputTaskMissing,
    WindowInputOrderingNotEstablished,
    PlanValidation(UnsupportedPart),
    DatafusionExactStatisticalAggregateUnsupported(UnsupportedPart),
    DatafusionExactStatisticalGroupKeyUnsupported,
    DatafusionExactUngroupedFloatSumUnsupported,
}

impl Reason {
    /// Stable report label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedAggregateGroupingSetsCapability => "unsupported_aggregate_grouping_sets_capability",
            Self::UnsupportedAggregateNodeCapability => "unsupported_aggregate_node_capability",
            Self::UnsupportedGpuResidentSourceCapability => "unsupported_gpu_resident_source_capability",
            Self::UnsupportedJoinNodeCapability => "unsupported_join_node_capability",
            Self::UnsupportedLimitNodeCapability => "unsupported_limit_node_capability",
            Self::UnsupportedProjectionNodeCapability => "unsupported_projection_node_capability",
            Self::UnsupportedSortNodeCapability => "unsupported_sort_node_capability",
            Self::UnsupportedUnionNodeCapability => "unsupported_union_node_capability",
            Self::UnsupportedWindowNodeCapability => "unsupported_window_node_capability",
            Self::PostJoinResidualFilterRequiresOneShotEquiJoin => {
                "post_join_residual_filter_requires_one_shot_equi_join"
            },
            Self::LargeKeyedHashJoinStrategyUnsupported => "large_keyed_hash_join_strategy_unsupported",
            Self::UnsupportedEdgeNormalizeNodeCapability(_) => "unsupported_edge_normalize_node_capability",
            Self::UnsupportedGraphAlgorithmNodeCapability(_) => "unsupported_graph_algorithm_node_capability",
            Self::UnsupportedVectorAlgorithmNodeCapability(_) => "unsupported_vector_algorithm_node_capability",
            Self::UnsupportedMlAlgorithmNodeCapability(_) => "unsupported_ml_algorithm_node_capability",
            Self::UnsupportedAggregateExpressionCapability => "unsupported_aggregate_expression_capability",
            Self::UnsupportedFilterExpressionCapability => "unsupported_filter_expression_capability",
            Self::UnsupportedProjectionExpressionCapability => "unsupported_projection_expression_capability",
            Self::UnsupportedAggregateGroupExpressionCapability => "unsupported_aggregate_group_expression_capability",
            Self::UnsupportedJoinResidualExpressionCapability => "unsupported_join_residual_expression_capability",
            Self::UnsupportedSortExpressionCapability => "unsupported_sort_expression_capability",
            Self::UnsupportedWindowAggregateExpressionCapability => {
                "unsupported_window_aggregate_expression_capability"
            },
            Self::UnsupportedWindowLagLeadExpressionCapability => "unsupported_window_lag_lead_expression_capability",
            Self::UnsupportedWindowOrderExpressionCapability => "unsupported_window_order_expression_capability",
            Self::UnsupportedWindowPartitionExpressionCapability => {
                "unsupported_window_partition_expression_capability"
            },
            Self::UnsupportedLiteralBatchSourceCapability => "unsupported_literal_batch_source_capability",
            Self::UnsupportedLocalParquetSourceCapability => "unsupported_local_parquet_source_capability",
            Self::UnsupportedJoinOutputDtype => "unsupported_join_output_dtype",
            Self::LeftAntiBuildRequiresDedupAggregate => "left_anti_build_requires_dedup_aggregate",
            Self::LeftAntiProbeRequiresDedupProvenInput => "left_anti_probe_requires_dedup_proven_input",
            Self::MissingEdgeProducer => "missing_edge_producer",
            Self::MissingEdgeRequiredProperty => "missing_edge_required_property",
            Self::AggregateSharedProducerConsumerNotAggregate => "aggregate_shared_producer_consumer_not_aggregate",
            Self::AggregateSharedProducerInputLineageUnresolved => "aggregate_shared_producer_input_lineage_unresolved",
            Self::AggregateSharedProducerMissingAggregate => "aggregate_shared_producer_missing_aggregate",
            Self::AggregateSharedProducerMissingNode => "aggregate_shared_producer_missing_node",
            Self::AggregateSharedProducerRequiresDataflowIdentity => {
                "aggregate_shared_producer_requires_dataflow_identity"
            },
            Self::AggregateSharedProducerRequiresInputLineage => "aggregate_shared_producer_requires_input_lineage",
            Self::AggregateSharedProducerRequiresResidentFrame => "aggregate_shared_producer_requires_resident_frame",
            Self::BuildBeforeProbeCycle => "build_before_probe_cycle",
            Self::ExpressionLineageRequiresResolvedInputColumn => "expression_lineage_requires_resolved_input_column",
            Self::OutputLineageAggregateFieldIndexOutOfRange => "output_lineage_aggregate_field_index_out_of_range",
            Self::OutputLineageFieldIndexOutOfRange => "output_lineage_field_index_out_of_range",
            Self::OutputLineageJoinOutputColumnResolution => "output_lineage_join_output_column_resolution",
            Self::OutputLineageJoinProjectionIndexOutOfRange => "output_lineage_join_projection_index_out_of_range",
            Self::OutputLineageMissingNode => "output_lineage_missing_node",
            Self::OutputLineageProjectionColumnResolution => "output_lineage_projection_column_resolution",
            Self::OutputLineageProjectionFieldIndexOutOfRange => "output_lineage_projection_field_index_out_of_range",
            Self::OutputLineageWindowMissingInput => "output_lineage_window_missing_input",
            Self::ProjectionTransparentResidencyConsumerNotProjection => {
                "projection_transparent_residency_consumer_not_projection"
            },
            Self::ProjectionTransparentResidencyMissingProducer => "projection_transparent_residency_missing_producer",
            Self::ProjectionTransparentResidencyMissingProjection => {
                "projection_transparent_residency_missing_projection"
            },
            Self::ProjectionTransparentResidencyRequiresResidentFrames => {
                "projection_transparent_residency_requires_resident_frames"
            },
            Self::ReachableSubgraphCycle => "reachable_subgraph_cycle",
            Self::ReachableSubgraphMissingNode => "reachable_subgraph_missing_node",
            Self::ResidentLifetimeEdgeRejected => "resident_lifetime_edge_rejected",
            Self::ResidentLifetimeMissingNode => "resident_lifetime_missing_node",
            Self::ResidentLifetimeRequiresResidentFrame => "resident_lifetime_requires_resident_frame",
            Self::ResidentUnaryChainMissingNode => "resident_unary_chain_missing_node",
            Self::ResidentUnaryChainMissingProducer => "resident_unary_chain_missing_producer",
            Self::ResidentUnaryChainRequiresConnectedNodes => "resident_unary_chain_requires_connected_nodes",
            Self::ResidentUnaryChainRequiresFilterProjectionNodes => {
                "resident_unary_chain_requires_filter_projection_nodes"
            },
            Self::ResidentUnaryChainRequiresResidentNodes => "resident_unary_chain_requires_resident_nodes",
            Self::ResidentUnaryChainRequiresResidentProducer => "resident_unary_chain_requires_resident_producer",
            Self::ResidualDoesNotReferenceBothInputs => "residual_does_not_reference_both_inputs",
            Self::SharedResidentSubplanEdgeRejected => "shared_resident_subplan_edge_rejected",
            Self::SharedResidentSubplanMissingEdge => "shared_resident_subplan_missing_edge",
            Self::SharedResidentSubplanMissingNode => "shared_resident_subplan_missing_node",
            Self::SharedResidentSubplanRequiresResidentFrame => "shared_resident_subplan_requires_resident_frame",
            Self::TopologicalScheduleCycle => "topological_schedule_cycle",
            Self::TopologicalScheduleMissingNode => "topological_schedule_missing_node",
            Self::CompiledPipelineRequiresJoinTaskGroup => "compiled_pipeline_requires_join_task_group",
            Self::CompiledPipelineJoinTaskGroupIncomplete => "compiled_pipeline_join_task_group_incomplete",
            Self::CompiledPipelineSharedProducerTaskKindUnsupported => {
                "compiled_pipeline_shared_producer_task_kind_unsupported"
            },
            Self::CompiledPipelineStageTaskPositionUnsupported => "compiled_pipeline_stage_task_position_unsupported",
            Self::CompiledPipelineSharedProducerLineageUnproven => "compiled_pipeline_shared_producer_lineage_unproven",
            Self::CompiledPipelineJoinTaskGroupPlanNodeMismatch => {
                "compiled_pipeline_join_task_group_plan_node_mismatch"
            },
            Self::CompiledPipelineJoinExecutionStrategyUnavailable => {
                "compiled_pipeline_join_execution_strategy_unavailable"
            },
            Self::CompiledTaskKindOutsideContract => "compiled_task_kind_outside_contract",
            Self::CompiledPipelineGraphNotConstructible => "compiled_pipeline_graph_not_constructible",
            Self::ConsumerRequiresExclusiveInput => "consumer_requires_exclusive_input",
            Self::CompiledPipelineTaskIdentityInvalid => "compiled_pipeline_task_identity_invalid",
            Self::CompiledPipelineEdgeIdentityInvalid => "compiled_pipeline_edge_identity_invalid",
            Self::CompiledPipelineBarrierIdentityInvalid => "compiled_pipeline_barrier_identity_invalid",
            Self::CompiledPipelineResourceIdentityInvalid => "compiled_pipeline_resource_identity_invalid",
            Self::CompiledPipelineTaskDependencyMissing => "compiled_pipeline_task_dependency_missing",
            Self::CompiledPipelineDependencyCycle => "compiled_pipeline_dependency_cycle",
            Self::CompiledPipelineResourceReferenceMissing => "compiled_pipeline_resource_reference_missing",
            Self::CompiledPipelineResourceFlowInvalid => "compiled_pipeline_resource_flow_invalid",
            Self::CompiledPipelineSinkOutputLifetimeInvalid => "compiled_pipeline_sink_output_lifetime_invalid",
            Self::CompiledPipelineBarrierDominanceInvalid => "compiled_pipeline_barrier_dominance_invalid",
            Self::CompiledPipelineTaskResourceArityInvalid => "compiled_pipeline_task_resource_arity_invalid",
            Self::CompiledPipelineStageBoundaryIncomplete => "compiled_pipeline_stage_boundary_incomplete",
            Self::CompiledPipelineJoinBuildBeforeProbeMissing => "compiled_pipeline_join_build_before_probe_missing",
            Self::CompiledPipelineFusedTopkLimitBindingInvalid => "compiled_pipeline_fused_topk_limit_binding_invalid",
            Self::CompiledPipelineSinkInputTaskMissing => "compiled_pipeline_sink_input_task_missing",
            Self::WindowInputOrderingNotEstablished => "window_input_ordering_not_established",
            Self::PlanValidation(_) => "native_plan_validation_unsupported",
            Self::DatafusionExactStatisticalAggregateUnsupported(_) => {
                "datafusion_exact_statistical_aggregate_unsupported"
            },
            Self::DatafusionExactStatisticalGroupKeyUnsupported => "datafusion_exact_statistical_group_key_unsupported",
            Self::DatafusionExactUngroupedFloatSumUnsupported => "datafusion_exact_ungrouped_float_sum_unsupported",
        }
    }

    /// Report category, decided by the reason alone.
    pub const fn category(self) -> CapabilityRejectionCategory {
        match self {
            Self::UnsupportedAggregateGroupingSetsCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedAggregateNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedGpuResidentSourceCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedJoinNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedLimitNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedProjectionNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedSortNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedUnionNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedWindowNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::PostJoinResidualFilterRequiresOneShotEquiJoin => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::LargeKeyedHashJoinStrategyUnsupported => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedEdgeNormalizeNodeCapability(part) => part.category(),
            Self::UnsupportedGraphAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedVectorAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedMlAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedAggregateExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedFilterExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedProjectionExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedAggregateGroupExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedJoinResidualExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedSortExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedWindowAggregateExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedWindowLagLeadExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedWindowOrderExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedWindowPartitionExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedLiteralBatchSourceCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedLocalParquetSourceCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedJoinOutputDtype => CapabilityRejectionCategory::UnsupportedArrowDtype,
            Self::LeftAntiBuildRequiresDedupAggregate => CapabilityRejectionCategory::MissingEdgeContract,
            Self::LeftAntiProbeRequiresDedupProvenInput => CapabilityRejectionCategory::MissingEdgeContract,
            Self::MissingEdgeProducer => CapabilityRejectionCategory::MissingEdgeContract,
            Self::MissingEdgeRequiredProperty => CapabilityRejectionCategory::MissingEdgeContract,
            Self::AggregateSharedProducerConsumerNotAggregate => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::AggregateSharedProducerInputLineageUnresolved => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::AggregateSharedProducerMissingAggregate => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::AggregateSharedProducerMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::AggregateSharedProducerRequiresDataflowIdentity => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::AggregateSharedProducerRequiresInputLineage => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::AggregateSharedProducerRequiresResidentFrame => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::BuildBeforeProbeCycle => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ExpressionLineageRequiresResolvedInputColumn => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageAggregateFieldIndexOutOfRange => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageFieldIndexOutOfRange => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageJoinOutputColumnResolution => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageJoinProjectionIndexOutOfRange => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageProjectionColumnResolution => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageProjectionFieldIndexOutOfRange => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::OutputLineageWindowMissingInput => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ProjectionTransparentResidencyConsumerNotProjection => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ProjectionTransparentResidencyMissingProducer => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ProjectionTransparentResidencyMissingProjection => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ProjectionTransparentResidencyRequiresResidentFrames => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ReachableSubgraphCycle => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ReachableSubgraphMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentLifetimeEdgeRejected => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentLifetimeMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentLifetimeRequiresResidentFrame => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainMissingProducer => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainRequiresConnectedNodes => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainRequiresFilterProjectionNodes => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ResidentUnaryChainRequiresResidentNodes => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainRequiresResidentProducer => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidualDoesNotReferenceBothInputs => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanEdgeRejected => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanMissingEdge => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanRequiresResidentFrame => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::TopologicalScheduleCycle => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::TopologicalScheduleMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::CompiledPipelineRequiresJoinTaskGroup => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineJoinTaskGroupIncomplete => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineSharedProducerTaskKindUnsupported => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineStageTaskPositionUnsupported => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineSharedProducerLineageUnproven => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineJoinTaskGroupPlanNodeMismatch => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineJoinExecutionStrategyUnavailable => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledTaskKindOutsideContract => CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
            Self::CompiledPipelineGraphNotConstructible => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::ConsumerRequiresExclusiveInput => CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
            Self::CompiledPipelineTaskIdentityInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineEdgeIdentityInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineBarrierIdentityInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineResourceIdentityInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineTaskDependencyMissing => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineDependencyCycle => CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope,
            Self::CompiledPipelineResourceReferenceMissing => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineResourceFlowInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineSinkOutputLifetimeInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineBarrierDominanceInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineTaskResourceArityInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineStageBoundaryIncomplete => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineJoinBuildBeforeProbeMissing => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineFusedTopkLimitBindingInvalid => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::CompiledPipelineSinkInputTaskMissing => {
                CapabilityRejectionCategory::UnsupportedCompiledPipelineEnvelope
            },
            Self::WindowInputOrderingNotEstablished => CapabilityRejectionCategory::MissingEdgeContract,
            Self::PlanValidation(part) => part.category(),
            Self::DatafusionExactStatisticalAggregateUnsupported(part) => part.category(),
            Self::DatafusionExactStatisticalGroupKeyUnsupported => CapabilityRejectionCategory::UnsupportedArrowDtype,
            Self::DatafusionExactUngroupedFloatSumUnsupported => CapabilityRejectionCategory::UnsupportedNodeOperator,
        }
    }

    /// The boundary of a pipeline, validation, or execution-mode refusal.
    /// Capability analysis reasons take their boundary from the owner.
    const fn boundary(self) -> Option<&'static str> {
        match self {
            Self::CompiledPipelineRequiresJoinTaskGroup
            | Self::CompiledPipelineJoinTaskGroupIncomplete
            | Self::CompiledPipelineSharedProducerTaskKindUnsupported
            | Self::CompiledPipelineStageTaskPositionUnsupported
            | Self::CompiledPipelineSharedProducerLineageUnproven
            | Self::CompiledPipelineJoinTaskGroupPlanNodeMismatch
            | Self::CompiledPipelineJoinExecutionStrategyUnavailable
            | Self::CompiledTaskKindOutsideContract
            | Self::CompiledPipelineGraphNotConstructible
            | Self::ConsumerRequiresExclusiveInput
            | Self::CompiledPipelineTaskIdentityInvalid
            | Self::CompiledPipelineEdgeIdentityInvalid
            | Self::CompiledPipelineBarrierIdentityInvalid
            | Self::CompiledPipelineResourceIdentityInvalid
            | Self::CompiledPipelineTaskDependencyMissing
            | Self::CompiledPipelineDependencyCycle
            | Self::CompiledPipelineResourceReferenceMissing
            | Self::CompiledPipelineResourceFlowInvalid
            | Self::CompiledPipelineSinkOutputLifetimeInvalid
            | Self::CompiledPipelineBarrierDominanceInvalid
            | Self::CompiledPipelineTaskResourceArityInvalid
            | Self::CompiledPipelineStageBoundaryIncomplete
            | Self::CompiledPipelineJoinBuildBeforeProbeMissing
            | Self::CompiledPipelineFusedTopkLimitBindingInvalid
            | Self::CompiledPipelineSinkInputTaskMissing
            | Self::WindowInputOrderingNotEstablished => Some("compiled_pipeline"),
            Self::PlanValidation(_) => Some("plan_validation"),
            Self::DatafusionExactStatisticalAggregateUnsupported(_)
            | Self::DatafusionExactStatisticalGroupKeyUnsupported
            | Self::DatafusionExactUngroupedFloatSumUnsupported => Some("datafusion_exact_statistical_aggregate"),
            _ => None,
        }
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The graph scope that owns a rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectionOwner {
    /// One native plan node.
    Node(NodeId),
    /// One producer-consumer boundary.
    Edge { producer: NodeId, consumer: EdgeConsumer },
    /// One subgraph invariant.
    Invariant(InvariantId),
    /// The whole plan: compiled pipeline, validation, or admission gates
    /// that no single node, edge, or invariant owns.
    Plan,
}

impl RejectionOwner {
    /// Contract scope reported for this owner.
    pub const fn scope(self) -> CapabilityScope {
        match self {
            Self::Node(_) => CapabilityScope::Node,
            Self::Edge { .. } => CapabilityScope::Edge,
            Self::Invariant(_) | Self::Plan => CapabilityScope::Subgraph,
        }
    }

    const fn boundary(self) -> &'static str {
        match self {
            Self::Node(_) => "node_capability",
            Self::Edge { .. } => "edge_capability",
            Self::Invariant(_) => "subgraph_capability",
            Self::Plan => "plan",
        }
    }
}

impl fmt::Display for RejectionOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Node(node) => write!(f, "node:{node}"),
            Self::Edge { producer, consumer } => match consumer {
                EdgeConsumer::Node(node) => write!(f, "edge:{producer}->node:{node}"),
                EdgeConsumer::Sink(ResultSinkKind::HostArrow) => write!(f, "edge:{producer}->sink:host_arrow"),
                EdgeConsumer::Sink(ResultSinkKind::GpuResident) => write!(f, "edge:{producer}->sink:gpu_resident"),
            },
            Self::Invariant(invariant) => write!(f, "subgraph:i{}", invariant.index()),
            Self::Plan => f.write_str("plan"),
        }
    }
}

/// One rejection: who refused, why, and diagnostic facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub owner: RejectionOwner,
    pub reason: Reason,
    /// Diagnostics only; they never change the category or boundary.
    pub facts: BTreeMap<&'static str, String>,
}

impl Rejection {
    pub fn new(owner: RejectionOwner, reason: Reason, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self { owner, reason, facts: facts.into_iter().collect() }
    }

    pub const fn category(&self) -> CapabilityRejectionCategory {
        self.reason.category()
    }

    pub const fn contract_scope(&self) -> CapabilityScope {
        self.owner.scope()
    }

    /// Stable diagnostic boundary label.
    pub const fn boundary(&self) -> &'static str {
        match self.reason.boundary() {
            Some(boundary) => boundary,
            None => self.owner.boundary(),
        }
    }
}
