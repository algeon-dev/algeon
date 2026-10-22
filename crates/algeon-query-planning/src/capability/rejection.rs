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
    /// The operation or execution strategy is unsupported.
    Operator,
    /// An input/output dtype or physical representation is unsupported.
    DataType,
    /// An expression family or expression shape is unsupported.
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
///
/// Labels are report contracts: planning reports, TSV families and tests match
/// on them, so a variant is renamed only together with every consumer. A new
/// variant needs an `as_str` and a `category` arm, and a `boundary` arm when it
/// is a pipeline, validation, or execution-mode refusal. Capability analysis
/// reasons take their boundary from the owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// The backend cannot execute the requested grouping sets.
    UnsupportedAggregateGroupingSetsCapability,
    /// The aggregate node has no supported native execution strategy.
    UnsupportedAggregateNodeCapability,
    /// The join form cannot be bound to a native strategy.
    UnsupportedJoinNodeCapability,
    /// The sort node cannot be bound to a native strategy.
    UnsupportedSortNodeCapability,
    /// A post-join residual predicate requires the one-shot equijoin path.
    PostJoinResidualFilterRequiresOneShotEquiJoin,
    /// The keyed hash build exceeds the admitted strategy bounds.
    LargeKeyedHashJoinStrategyUnsupported,
    /// Edge normalization was rejected for the supplied operation, type or expression.
    UnsupportedEdgeNormalizeNodeCapability(UnsupportedPart),
    /// A graph algorithm was rejected for the supplied operation, type or expression.
    UnsupportedGraphAlgorithmNodeCapability(UnsupportedPart),
    /// A vector algorithm was rejected for the supplied operation, type or expression.
    UnsupportedVectorAlgorithmNodeCapability(UnsupportedPart),
    /// An ML algorithm was rejected for the supplied operation, type or expression.
    UnsupportedMlAlgorithmNodeCapability(UnsupportedPart),
    /// An aggregate expression cannot be evaluated by the backend.
    UnsupportedAggregateExpressionCapability,
    /// A filter predicate cannot be evaluated by the backend.
    UnsupportedFilterExpressionCapability,
    /// A grouping-key expression cannot be evaluated by the backend.
    UnsupportedAggregateGroupExpressionCapability,
    /// A residual join expression cannot be evaluated by the backend.
    UnsupportedJoinResidualExpressionCapability,
    /// A sort-key expression cannot be evaluated by the backend.
    UnsupportedSortExpressionCapability,
    /// The producer referenced by an execution edge is absent from the plan.
    MissingEdgeProducer,
    /// A shared aggregate producer has a consumer outside the aggregate contract.
    AggregateSharedProducerConsumerNotAggregate,
    /// The shared aggregate producer's input columns lack resolved lineage.
    AggregateSharedProducerInputLineageUnresolved,
    /// The shared-producer invariant references an absent aggregate.
    AggregateSharedProducerMissingAggregate,
    /// A node required by the shared aggregate producer is absent.
    AggregateSharedProducerMissingNode,
    /// Shared aggregate consumers do not have the required dataflow identity.
    AggregateSharedProducerRequiresDataflowIdentity,
    /// The shared aggregate input lacks the required lineage proof.
    AggregateSharedProducerRequiresInputLineage,
    /// An expression references an input column whose lineage cannot be resolved.
    ExpressionLineageRequiresResolvedInputColumn,
    /// An aggregate output field index exceeds its output schema.
    OutputLineageAggregateFieldIndexOutOfRange,
    /// An output field index exceeds the node schema.
    OutputLineageFieldIndexOutOfRange,
    /// A join output column cannot be traced to either input.
    OutputLineageJoinOutputColumnResolution,
    /// A join projection references an output field outside its schema.
    OutputLineageJoinProjectionIndexOutOfRange,
    /// Output lineage traversal references an absent plan node.
    OutputLineageMissingNode,
    /// A projected column cannot be resolved against its input.
    OutputLineageProjectionColumnResolution,
    /// A projection field index exceeds the expression list.
    OutputLineageProjectionFieldIndexOutOfRange,
    /// Window output lineage cannot resolve the window input.
    OutputLineageWindowMissingInput,
    /// A residency-transparent edge has a consumer that is not a projection.
    ProjectionTransparentResidencyConsumerNotProjection,
    /// The producer of a residency-transparent projection is absent.
    ProjectionTransparentResidencyMissingProducer,
    /// The projection named by a residency invariant is absent.
    ProjectionTransparentResidencyMissingProjection,
    /// A producer-consumer edge cannot satisfy the resident lifetime contract.
    ResidentLifetimeEdgeRejected,
    /// A node named by a resident unary-chain invariant is absent.
    ResidentUnaryChainMissingNode,
    /// The producer of a resident unary chain is absent.
    ResidentUnaryChainMissingProducer,
    /// The resident unary-chain nodes do not form the required connected path.
    ResidentUnaryChainRequiresConnectedNodes,
    /// A resident unary chain contains an operation other than filter or projection.
    ResidentUnaryChainRequiresFilterProjectionNodes,
    /// A residual predicate lacks column references to one of the join inputs.
    ResidualDoesNotReferenceBothInputs,
    /// An edge in the shared resident subplan violates its ownership contract.
    SharedResidentSubplanEdgeRejected,
    /// The shared resident subplan lacks an edge required by its invariant.
    SharedResidentSubplanMissingEdge,
    /// A compiled join lacks its required build/probe task group.
    CompiledPipelineRequiresJoinTaskGroup,
    /// A compiled join task group omits a required execution stage.
    CompiledPipelineJoinTaskGroupIncomplete,
    /// A shared producer uses a task kind outside the compiled sharing contract.
    CompiledPipelineSharedProducerTaskKindUnsupported,
    /// A task occurs in a position its compiled stage does not admit.
    CompiledPipelineStageTaskPositionUnsupported,
    /// A compiled shared producer lacks proof of common input lineage.
    CompiledPipelineSharedProducerLineageUnproven,
    /// Tasks grouped as one join refer to different plan nodes.
    CompiledPipelineJoinTaskGroupPlanNodeMismatch,
    /// A scheduled task kind is outside the supported compiled contract.
    CompiledTaskKindOutsideContract,
    /// The plan cannot be represented by the supported task/resource graph.
    CompiledPipelineGraphNotConstructible,
    /// A consumer requires exclusive input ownership that sharing cannot provide.
    ConsumerRequiresExclusiveInput,
    /// A compiled task identity is duplicated or inconsistent with its graph position.
    CompiledPipelineTaskIdentityInvalid,
    /// A compiled edge identity is duplicated or inconsistent with its graph position.
    CompiledPipelineEdgeIdentityInvalid,
    /// A compiled barrier identity is duplicated or inconsistent with its graph position.
    CompiledPipelineBarrierIdentityInvalid,
    /// A compiled resource identity is duplicated or inconsistent with its graph position.
    CompiledPipelineResourceIdentityInvalid,
    /// A task dependency refers to a task absent from the schedule.
    CompiledPipelineTaskDependencyMissing,
    /// Compiled task dependencies contain a cycle.
    CompiledPipelineDependencyCycle,
    /// A task or edge refers to an absent compiled resource.
    CompiledPipelineResourceReferenceMissing,
    /// A resource's producer and consumers violate the compiled dataflow contract.
    CompiledPipelineResourceFlowInvalid,
    /// The sink output does not retain the lifetime required for result delivery.
    CompiledPipelineSinkOutputLifetimeInvalid,
    /// A barrier does not order all tasks required by its completion contract.
    CompiledPipelineBarrierDominanceInvalid,
    /// A task has an unsupported number of input or output resources.
    CompiledPipelineTaskResourceArityInvalid,
    /// A compiled stage boundary omits a required dependency or resource binding.
    CompiledPipelineStageBoundaryIncomplete,
    /// A join probe lacks the required dependency on completion of its build.
    CompiledPipelineJoinBuildBeforeProbeMissing,
    /// The fused top-k stage is not bound to the required limit node.
    CompiledPipelineFusedTopkLimitBindingInvalid,
    /// The result sink has no corresponding input task in the schedule.
    CompiledPipelineSinkInputTaskMissing,
    /// The window input lacks the ordering required by its window expressions.
    WindowInputOrderingNotEstablished,
    /// Plan validation rejected an operation, type or expression before compilation.
    PlanValidation(UnsupportedPart),
    /// The exact statistical aggregate mode cannot cover the indicated contract.
    DatafusionExactStatisticalAggregateUnsupported(UnsupportedPart),
    /// A grouping-key type is outside the exact statistical aggregate contract.
    DatafusionExactStatisticalGroupKeyUnsupported,
    /// Ungrouped floating SUM or AVG cannot preserve the required DataFusion evaluation order.
    DatafusionExactUngroupedFloatSumUnsupported,
}

impl Reason {
    /// Stable report label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UnsupportedAggregateGroupingSetsCapability => "unsupported_aggregate_grouping_sets_capability",
            Self::UnsupportedAggregateNodeCapability => "unsupported_aggregate_node_capability",
            Self::UnsupportedJoinNodeCapability => "unsupported_join_node_capability",
            Self::UnsupportedSortNodeCapability => "unsupported_sort_node_capability",
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
            Self::UnsupportedAggregateGroupExpressionCapability => "unsupported_aggregate_group_expression_capability",
            Self::UnsupportedJoinResidualExpressionCapability => "unsupported_join_residual_expression_capability",
            Self::UnsupportedSortExpressionCapability => "unsupported_sort_expression_capability",
            Self::MissingEdgeProducer => "missing_edge_producer",
            Self::AggregateSharedProducerConsumerNotAggregate => "aggregate_shared_producer_consumer_not_aggregate",
            Self::AggregateSharedProducerInputLineageUnresolved => "aggregate_shared_producer_input_lineage_unresolved",
            Self::AggregateSharedProducerMissingAggregate => "aggregate_shared_producer_missing_aggregate",
            Self::AggregateSharedProducerMissingNode => "aggregate_shared_producer_missing_node",
            Self::AggregateSharedProducerRequiresDataflowIdentity => {
                "aggregate_shared_producer_requires_dataflow_identity"
            },
            Self::AggregateSharedProducerRequiresInputLineage => "aggregate_shared_producer_requires_input_lineage",
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
            Self::ResidentLifetimeEdgeRejected => "resident_lifetime_edge_rejected",
            Self::ResidentUnaryChainMissingNode => "resident_unary_chain_missing_node",
            Self::ResidentUnaryChainMissingProducer => "resident_unary_chain_missing_producer",
            Self::ResidentUnaryChainRequiresConnectedNodes => "resident_unary_chain_requires_connected_nodes",
            Self::ResidentUnaryChainRequiresFilterProjectionNodes => {
                "resident_unary_chain_requires_filter_projection_nodes"
            },
            Self::ResidualDoesNotReferenceBothInputs => "residual_does_not_reference_both_inputs",
            Self::SharedResidentSubplanEdgeRejected => "shared_resident_subplan_edge_rejected",
            Self::SharedResidentSubplanMissingEdge => "shared_resident_subplan_missing_edge",
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
            Self::UnsupportedJoinNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedSortNodeCapability => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::PostJoinResidualFilterRequiresOneShotEquiJoin => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::LargeKeyedHashJoinStrategyUnsupported => CapabilityRejectionCategory::UnsupportedNodeOperator,
            Self::UnsupportedEdgeNormalizeNodeCapability(part) => part.category(),
            Self::UnsupportedGraphAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedVectorAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedMlAlgorithmNodeCapability(part) => part.category(),
            Self::UnsupportedAggregateExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedFilterExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::UnsupportedAggregateGroupExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedJoinResidualExpressionCapability => {
                CapabilityRejectionCategory::UnsupportedExpressionFamily
            },
            Self::UnsupportedSortExpressionCapability => CapabilityRejectionCategory::UnsupportedExpressionFamily,
            Self::MissingEdgeProducer => CapabilityRejectionCategory::MissingEdgeContract,
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
            Self::ResidentLifetimeEdgeRejected => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainMissingNode => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainMissingProducer => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainRequiresConnectedNodes => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::ResidentUnaryChainRequiresFilterProjectionNodes => {
                CapabilityRejectionCategory::MissingSubgraphInvariant
            },
            Self::ResidualDoesNotReferenceBothInputs => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanEdgeRejected => CapabilityRejectionCategory::MissingSubgraphInvariant,
            Self::SharedResidentSubplanMissingEdge => CapabilityRejectionCategory::MissingSubgraphInvariant,
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
    Edge {
        /// Node supplying the rejected edge.
        producer: NodeId,
        /// Node or result sink consuming the edge.
        consumer: EdgeConsumer,
    },
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
    /// Scope that refused; decides the reported scope and, for capability
    /// analysis reasons, the boundary.
    pub owner: RejectionOwner,
    /// Decides the report label and category.
    pub reason: Reason,
    /// Diagnostics only; they never change the category or boundary.
    pub facts: BTreeMap<&'static str, String>,
}

impl Rejection {
    /// Collects diagnostic facts for one typed rejection. Duplicate fact names
    /// retain the last value; facts do not change its classification.
    pub fn new(owner: RejectionOwner, reason: Reason, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self { owner, reason, facts: facts.into_iter().collect() }
    }

    /// Classification determined by the rejection reason.
    pub const fn category(&self) -> CapabilityRejectionCategory {
        self.reason.category()
    }

    pub(crate) const fn contract_scope(&self) -> CapabilityScope {
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
