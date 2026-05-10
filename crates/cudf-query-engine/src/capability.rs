use std::collections::{BTreeMap, BTreeSet};

use crate::{
    bounded::{
        BoundedJoinNodeCapability, BoundedJoinPlanSupport, BoundedKeylessConditionalJoinType,
        analyze_bounded_join_plan_support, bounded_join_node_capability, bounded_multi_join_node_capability,
        bounded_root_filtered_join_compatible, bounded_root_hash_join_compatible,
    },
    error::QueryError,
    exec::validate_local_parquet_source,
    expr::{AggregateExpr, AggregateFunction, BinaryOp, ColumnQualifier, Expr, LiteralValue},
    plan::{JoinNode, JoinSide, JoinType, LimitNode, NodeId, PlanNode, QueryNode, QueryPlan},
    runtime::ReservationKind,
    source::SourceNode,
};

/// Whether the engine currently proves a bounded contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractStatus {
    Modeled,
    BlockedByLowerLevel,
    DeferredUntilPrerequisite,
}

/// How precise a modeled bounded contract is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContractQuality {
    ExactPreflight,
    MetadataBound,
    HardBudgetedReportWithUnresolvedPreflight,
    ObservedRetainedState,
    QueryEngineOwnedOrdering,
    QueryEngineOwnedBoundary,
}

/// The most local graph scope that owns a reservation requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReservationRequirementScope {
    Node,
    Edge,
    Subgraph,
}

impl ReservationRequirementScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Node => "node",
            Self::Edge => "edge",
            Self::Subgraph => "subgraph",
        }
    }
}

/// One memory or execution-boundary requirement attached to the most local
/// capability owner that can state it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservationRequirement {
    pub kind: ReservationKind,
    pub scope: ReservationRequirementScope,
    pub memory_boundary: &'static str,
    pub required_contract: &'static str,
    pub status: ContractStatus,
    pub quality: ContractQuality,
}

impl ReservationRequirement {
    pub fn modeled(
        kind: ReservationKind,
        memory_boundary: &'static str,
        required_contract: &'static str,
        quality: ContractQuality,
    ) -> Self {
        Self {
            kind,
            scope: reservation_requirement_scope(kind),
            memory_boundary,
            required_contract,
            status: ContractStatus::Modeled,
            quality,
        }
    }

    pub fn compatibility_modeled(kind: ReservationKind) -> Self {
        let contract = compatibility_reservation_contract(kind);
        Self::modeled(kind, contract.memory_boundary, contract.required_contract, contract.quality)
    }

    pub fn compatibility_fact_row(&self) -> String {
        format!("{}:{}:{}", self.kind.as_str(), self.required_contract, self.compatibility_detail())
    }

    fn compatibility_detail(&self) -> &'static str {
        match self.kind {
            ReservationKind::SourceRead
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::OperatorOutput
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::BuildSide => "modeled_before_execution",
            ReservationKind::ProbeSide => "modeled_by_ordered_bounded_join_execution_path",
            ReservationKind::HashTable => "modeled_by_cudf_rs_pre_build_sizing",
            ReservationKind::FilteredProbeState => "modeled_by_cudf_rs_filtered_probe_state_observability",
            ReservationKind::MixedSizeData => "modeled_by_cudf_rs_retained_join_size_data_observability",
            ReservationKind::GatherMap => "modeled_by_cudf_rs_join_gather_map_sizing",
            ReservationKind::JoinOutput => "modeled_by_cudf_rs_dataframe_gather_or_projected_cross_output_sizing",
            ReservationKind::FinalSort
                if self.quality == ContractQuality::HardBudgetedReportWithUnresolvedPreflight =>
            {
                "modeled_by_cudf_rs_hard_budgeted_report_with_unresolved_preflight"
            },
            ReservationKind::Materialization => "modeled_by_explicit_sink_boundary",
            _ => "modeled_by_capability_analysis",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompatibilityReservationContract {
    memory_boundary: &'static str,
    required_contract: &'static str,
    quality: ContractQuality,
}

const fn compatibility_reservation_contract(kind: ReservationKind) -> CompatibilityReservationContract {
    match kind {
        ReservationKind::SourceRead => CompatibilityReservationContract {
            memory_boundary: "source_read",
            required_contract: "source_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::OperatorOutput => CompatibilityReservationContract {
            memory_boundary: "operator_output",
            required_contract: "aggregate_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::BuildSide => CompatibilityReservationContract {
            memory_boundary: "join_build_side",
            required_contract: "query_engine_build_side_source_reservation",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::ProbeSide => CompatibilityReservationContract {
            memory_boundary: "join_probe_side",
            required_contract: "query_engine_probe_side_source_reservation",
            quality: ContractQuality::QueryEngineOwnedOrdering,
        },
        ReservationKind::HashTable => CompatibilityReservationContract {
            memory_boundary: "join_hash_table",
            required_contract: "hash_join_pre_build_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::FilteredProbeState => CompatibilityReservationContract {
            memory_boundary: "join_filtered_probe_state",
            required_contract: "filtered_join_retained_probe_state",
            quality: ContractQuality::ObservedRetainedState,
        },
        ReservationKind::MixedSizeData => CompatibilityReservationContract {
            memory_boundary: "join_mixed_size_data",
            required_contract: "mixed_keyed_residual_or_conditional_full_retained_size_data",
            quality: ContractQuality::ObservedRetainedState,
        },
        ReservationKind::GatherMap => CompatibilityReservationContract {
            memory_boundary: "join_gather_map",
            required_contract: "join_gather_map_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::JoinOutput => CompatibilityReservationContract {
            memory_boundary: "join_output",
            required_contract: "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            quality: ContractQuality::ExactPreflight,
        },
        ReservationKind::FinalSort => CompatibilityReservationContract {
            memory_boundary: "join_pipeline_final_sort",
            required_contract: "final_sort_contract",
            quality: ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        },
        ReservationKind::Materialization => CompatibilityReservationContract {
            memory_boundary: "host_materialization",
            required_contract: "query_engine_explicit_sink_materialization_reservation",
            quality: ContractQuality::QueryEngineOwnedBoundary,
        },
        ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => CompatibilityReservationContract {
            memory_boundary: "non_bounded_join_compatibility_boundary",
            required_contract: "non_bounded_join_compatibility_contract",
            quality: ContractQuality::ExactPreflight,
        },
    }
}

pub const fn reservation_requirement_scope(kind: ReservationKind) -> ReservationRequirementScope {
    match kind {
        ReservationKind::OperatorOutput
        | ReservationKind::HashTable
        | ReservationKind::FilteredProbeState
        | ReservationKind::MixedSizeData
        | ReservationKind::SourceCacheRetained
        | ReservationKind::ChunkOperator
        | ReservationKind::PartialState
        | ReservationKind::FinalMerge => ReservationRequirementScope::Node,
        ReservationKind::SourceRead
        | ReservationKind::BuildSide
        | ReservationKind::ProbeSide
        | ReservationKind::GatherMap
        | ReservationKind::JoinOutput
        | ReservationKind::FinalSort
        | ReservationKind::Materialization => ReservationRequirementScope::Edge,
    }
}

/// A symbolic dependency that a conditional capability is waiting for.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintRef {
    pub scope: ConstraintScope,
    pub name: &'static str,
}

impl ConstraintRef {
    pub const fn node(node_id: NodeId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Node(node_id), name }
    }

    pub const fn edge(producer: NodeId, consumer: EdgeConsumer, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Edge { producer, consumer }, name }
    }

    pub const fn invariant(invariant_id: InvariantId, name: &'static str) -> Self {
        Self { scope: ConstraintScope::Invariant(invariant_id), name }
    }
}

/// The graph scope where a constraint belongs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintScope {
    Node(NodeId),
    Edge { producer: NodeId, consumer: EdgeConsumer },
    Invariant(InvariantId),
}

/// Capability proof result for a node, edge, or invariant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityDecision {
    Satisfied,
    Rejected { reason: &'static str, facts: BTreeMap<&'static str, String> },
    Conditional { waiting_on: Vec<ConstraintRef> },
}

impl CapabilityDecision {
    fn rejected(reason: &'static str, facts: impl IntoIterator<Item = (&'static str, String)>) -> Self {
        Self::Rejected { reason, facts: facts.into_iter().collect() }
    }
}

/// Node-local executable capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub operator: NodeOperator,
    pub executable: CapabilityDecision,
    pub produced_properties: Vec<ProducedProperty>,
    pub local_requirements: Vec<ReservationRequirement>,
}

/// Native operator shape captured without naming a whole bounded family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOperator {
    Source { format: SourceFormatCapability },
    Filter,
    Projection,
    Aggregate { grouping: AggregateGrouping, functions: Vec<AggregateFunction> },
    Sort { ordering: OrderingRequirement },
    Limit,
    Join { join_type: JoinType, key_count: usize, residual: ResidualRequirement, build_side: JoinSide },
}

/// Source format capability known by the native engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceFormatCapability {
    LocalParquet,
    GpuResident,
}

/// Aggregate grouping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateGrouping {
    Scalar,
    Grouped,
}

/// Whether a node establishes an ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderingRequirement {
    Required,
    None,
}

/// Whether a join has a residual predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualRequirement {
    None,
    Required,
}

/// Properties produced by a node for downstream edge contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProducedProperty {
    ResidentFrame,
    GroupedFrame,
    ScalarFrame,
    OrderedFrame,
}

/// Producer-consumer capability for one dependency edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeContract {
    pub producer: NodeId,
    pub consumer: EdgeConsumer,
    pub contract_kinds: Vec<EdgeContractKind>,
    pub input_role: InputRole,
    pub required_properties: Vec<RequiredProperty>,
    pub reservation_requirements: Vec<ReservationRequirement>,
    pub decision: CapabilityDecision,
}

/// Producer-consumer contract classes attached to one edge.
///
/// A single graph edge can carry more than one contract. For example, a
/// local-Parquet source feeding a join build side owns both the source-read
/// resident input contract and the build-side residency contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EdgeContractKind {
    ResidentSourceInput { format: SourceFormatCapability },
    ResidentAggregateInput,
    JoinBuildInput { side: JoinSide },
    JoinProbeInput { side: JoinSide },
    ResidentFrameFlow,
    JoinMaterialization,
    SortInput,
    ExplicitSinkMaterialization,
}

/// Consumer endpoint for an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EdgeConsumer {
    Node(NodeId),
    Sink(ResultSinkKind),
}

/// Result sink kinds as capability consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResultSinkKind {
    HostArrow,
    GpuResident,
}

/// Semantic role of an input edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputRole {
    UnaryInput,
    AggregateInput,
    SortInput,
    LimitInput,
    JoinBuildInput { side: JoinSide },
    JoinProbeInput { side: JoinSide },
    SinkInput,
}

/// Property required from an edge producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequiredProperty {
    ResidentFrame,
}

/// Stable identifier for an invariant inside one analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InvariantId(pub usize);

impl InvariantId {
    pub const fn index(self) -> usize {
        self.0
    }
}

/// A non-local statement that cannot be proven from one node or edge alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubgraphInvariant {
    pub invariant_id: InvariantId,
    pub kind: SubgraphInvariantKind,
    pub nodes: Vec<NodeId>,
    pub required_schedule: ScheduleConstraint,
    pub required_resources: Vec<ResidentResource>,
    pub decision: CapabilityDecision,
}

/// Subgraph invariant vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubgraphInvariantKind {
    BuildBeforeProbe { join_id: NodeId },
    ClosedJoinSubtree { root_join_id: NodeId },
    TopologicalBuildProbeSchedule,
    SharedResidentSubplan,
    AggregateDerivesFromSharedProducer,
    FinalBarrierOrder,
    ProjectionTransparentResidency,
    ResidualUsesBothInputs { join_id: NodeId },
    ResidentLifetimeDominatesConsumers,
}

/// Scheduling proof required by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleConstraint {
    None,
    BuildBeforeProbe,
    Topological,
    Barrier,
}

/// Resident resources whose lifetime or ordering is governed by an invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentResource {
    NodeOutput(NodeId),
    JoinBuildInput { join_id: NodeId, input: NodeId },
    JoinProbeInput { join_id: NodeId, input: NodeId },
}

/// Transitional family projection for report/executor compatibility.
///
/// This is deliberately separate from the node, edge, and invariant primitives:
/// admission should migrate toward capability proofs, while legacy reports and
/// dispatch can continue projecting a family name during the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompatibilityFamilyProjection {
    RootJoin,
    SingleJoinPostRootPipeline,
    KeylessConditionalJoinInputSubplanPipeline,
    KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline,
    MultiJoinTreePostRootPipeline,
    MultiJoinAggregateOverJoinSubplanTopKPipeline,
    MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline,
    HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline,
    MultiJoinResidentGroupedAggregateSubqueryFilterPipeline,
    KeyedHashJoinResidentSubplanPipeline,
}

/// Side-channel capability analysis for one lowered native query plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityAnalysis {
    pub nodes: Vec<NodeCapability>,
    pub edges: Vec<EdgeContract>,
    pub invariants: Vec<SubgraphInvariant>,
    pub requirements: Vec<ReservationRequirement>,
    pub closed_multi_join_subgraph_admission: CapabilityDecision,
    pub compatibility_family_projection: Option<CompatibilityFamilyProjection>,
}

impl CapabilityAnalysis {
    pub fn has_rejections(&self) -> bool {
        self.nodes.iter().any(|node| matches!(node.executable, CapabilityDecision::Rejected { .. }))
            || self.edges.iter().any(|edge| matches!(edge.decision, CapabilityDecision::Rejected { .. }))
            || self.invariants.iter().any(|invariant| matches!(invariant.decision, CapabilityDecision::Rejected { .. }))
    }

    pub fn admits_closed_multi_join_subgraph(&self) -> bool {
        self.closed_multi_join_subgraph_admission == CapabilityDecision::Satisfied
    }

    pub fn compatibility_reservation_requirements(&self) -> Vec<ReservationRequirement> {
        let mut kinds = Vec::new();
        if self.has_requirement_kind(ReservationKind::SourceRead) {
            kinds.push(ReservationKind::SourceRead);
        }
        if self.has_requirement_kind(ReservationKind::OperatorOutput) {
            kinds.push(ReservationKind::OperatorOutput);
        }
        if self.nodes.iter().any(|node| matches!(node.operator, NodeOperator::Join { .. })) {
            kinds.extend([
                ReservationKind::BuildSide,
                ReservationKind::HashTable,
                ReservationKind::ProbeSide,
                ReservationKind::FilteredProbeState,
                ReservationKind::MixedSizeData,
                ReservationKind::GatherMap,
                ReservationKind::JoinOutput,
            ]);
        }
        if self.has_requirement_kind(ReservationKind::FinalSort) {
            kinds.push(ReservationKind::FinalSort);
        }
        if self.has_requirement_kind(ReservationKind::Materialization) {
            kinds.push(ReservationKind::Materialization);
        }
        kinds.into_iter().map(ReservationRequirement::compatibility_modeled).collect()
    }

    pub fn compatibility_reservation_kinds_csv(&self) -> String {
        csv_join(self.compatibility_reservation_requirements().iter().map(|requirement| requirement.kind.as_str()))
    }

    pub fn compatibility_reservation_statuses_csv(&self) -> String {
        csv_join(
            self.compatibility_reservation_requirements()
                .iter()
                .map(|requirement| format!("{}:{}", requirement.kind.as_str(), requirement.status.as_str())),
        )
    }

    pub fn compatibility_reservation_requirements_csv(&self) -> String {
        csv_join(
            self.compatibility_reservation_requirements().iter().map(ReservationRequirement::compatibility_fact_row),
        )
    }

    pub fn compatibility_modeled_reservations_csv(&self) -> String {
        csv_join(self.compatibility_reservation_requirements().iter().filter_map(|requirement| {
            (requirement.status == ContractStatus::Modeled).then_some(requirement.kind.as_str())
        }))
    }

    fn has_requirement_kind(&self, kind: ReservationKind) -> bool {
        self.requirements.iter().any(|requirement| requirement.kind == kind)
    }
}

impl ContractStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Modeled => "modeled",
            Self::BlockedByLowerLevel => "blocked_by_lower_level",
            Self::DeferredUntilPrerequisite => "deferred_until_prerequisite",
        }
    }
}

impl QueryPlan {
    /// Runs capability analysis without changing bounded admission or execution.
    pub fn capability_analysis(&self) -> CapabilityAnalysis {
        analyze_query_capabilities(self)
    }
}

/// Runs capability analysis without changing bounded admission or execution.
pub fn analyze_query_capabilities(plan: &QueryPlan) -> CapabilityAnalysis {
    let nodes = plan.nodes().iter().map(|node| analyze_node(plan, node)).collect::<Vec<_>>();
    let mut edges =
        plan.nodes().iter().flat_map(|node| analyze_input_edges(plan, node.id, &node.kind)).collect::<Vec<_>>();
    edges.push(analyze_sink_edge(plan));

    let invariants = analyze_invariants(plan, &edges);
    let requirements = collect_requirements(&nodes, &edges);
    let closed_multi_join_subgraph_admission =
        analyze_closed_multi_join_subgraph_admission(plan, &nodes, &edges, &invariants);
    CapabilityAnalysis {
        nodes,
        edges,
        invariants,
        requirements,
        closed_multi_join_subgraph_admission,
        compatibility_family_projection: compatibility_family_projection(plan),
    }
}

fn analyze_node(plan: &QueryPlan, node: &QueryNode) -> NodeCapability {
    let node_id = node.id;
    let kind = &node.kind;
    let operator = node_operator(kind);
    let executable = node_executable_decision(plan, node);
    let produced_properties = produced_properties(node);
    let local_requirements = node_local_requirements(kind);
    NodeCapability { node_id, operator, executable, produced_properties, local_requirements }
}

fn node_operator(kind: &PlanNode) -> NodeOperator {
    match kind {
        PlanNode::Source(source) => NodeOperator::Source { format: source_format_capability(source) },
        PlanNode::Filter(_) => NodeOperator::Filter,
        PlanNode::Projection(_) => NodeOperator::Projection,
        PlanNode::Aggregate(aggregate) => NodeOperator::Aggregate {
            grouping: if aggregate.spec.group_exprs.is_empty() {
                AggregateGrouping::Scalar
            } else {
                AggregateGrouping::Grouped
            },
            functions: aggregate.spec.aggr_exprs.iter().map(|expr| expr.function).collect(),
        },
        PlanNode::Sort(sort) => NodeOperator::Sort {
            ordering: if sort.spec.expressions.is_empty() {
                OrderingRequirement::None
            } else {
                OrderingRequirement::Required
            },
        },
        PlanNode::Limit(_) => NodeOperator::Limit,
        PlanNode::Join(join) => NodeOperator::Join {
            join_type: join.spec.join_type,
            key_count: join.spec.keys.len(),
            residual: if join.spec.residual_predicate.is_some() {
                ResidualRequirement::Required
            } else {
                ResidualRequirement::None
            },
            build_side: join.spec.build_side,
        },
    }
}

fn source_format_capability(source: &SourceNode) -> SourceFormatCapability {
    match source {
        SourceNode::LocalParquet(_) => SourceFormatCapability::LocalParquet,
        SourceNode::GpuResident(_) => SourceFormatCapability::GpuResident,
    }
}

fn node_executable_decision(plan: &QueryPlan, node: &QueryNode) -> CapabilityDecision {
    match &node.kind {
        PlanNode::Source(source) => source_node_executable_decision(source),
        PlanNode::Filter(filter) => expression_node_decision(
            node.id,
            "unsupported_filter_expression_capability",
            std::iter::once(&filter.predicate),
        ),
        PlanNode::Projection(projection) => {
            let mut decision = expression_node_decision(
                node.id,
                "unsupported_projection_expression_capability",
                projection.expressions(),
            );
            if matches!(decision, CapabilityDecision::Satisfied)
                && projection.outputs.len() != node.output_schema().fields().len()
            {
                decision = CapabilityDecision::rejected(
                    "unsupported_projection_node_capability",
                    [
                        ("node_id", node.id.to_string()),
                        ("projection_output_count", projection.outputs.len().to_string()),
                        ("schema_field_count", node.output_schema().fields().len().to_string()),
                    ],
                );
            }
            decision
        },
        PlanNode::Aggregate(aggregate) => aggregate_node_executable_decision(node.id, aggregate),
        PlanNode::Sort(sort) => sort_node_executable_decision(node.id, sort),
        PlanNode::Limit(limit) => limit_node_executable_decision(plan, node.id, limit),
        PlanNode::Join(join) => join_node_executable_decision(node.id, join),
    }
}

fn source_node_executable_decision(source: &SourceNode) -> CapabilityDecision {
    match source {
        SourceNode::LocalParquet(source) => validation_result_decision(
            validate_local_parquet_source(source),
            "unsupported_local_parquet_source_capability",
            [("source_format", "local_parquet".to_string())],
        ),
        SourceNode::GpuResident(source) if source.input_name.is_empty() => CapabilityDecision::rejected(
            "unsupported_gpu_resident_source_capability",
            [("reason", "empty_input_name".to_string()), ("source_format", "gpu_resident".to_string())],
        ),
        SourceNode::GpuResident(_) => CapabilityDecision::Satisfied,
    }
}

fn expression_node_decision<'a>(
    node_id: NodeId,
    reason: &'static str,
    expressions: impl IntoIterator<Item = &'a Expr>,
) -> CapabilityDecision {
    for expression in expressions {
        if let Err(rejection) = validate_general_expr_capability(expression) {
            return expression_rejection_decision(node_id, reason, expression, rejection);
        }
    }
    CapabilityDecision::Satisfied
}

fn aggregate_node_executable_decision(node_id: NodeId, aggregate: &crate::plan::AggregateNode) -> CapabilityDecision {
    for expr in &aggregate.spec.group_exprs {
        if let Err(rejection) = validate_named_column_expr_capability(expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_aggregate_group_expression_capability",
                expr,
                rejection,
            );
        }
    }
    for aggregate_expr in &aggregate.spec.aggr_exprs {
        if let Err(rejection) =
            validate_aggregate_expr_capability(aggregate_expr, aggregate.spec.group_exprs.is_empty())
        {
            return expression_rejection_decision(
                node_id,
                "unsupported_aggregate_expression_capability",
                &aggregate_expr.expr,
                rejection,
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn sort_node_executable_decision(node_id: NodeId, sort: &crate::plan::SortNode) -> CapabilityDecision {
    if sort.spec.expressions.is_empty() {
        return CapabilityDecision::rejected(
            "unsupported_sort_node_capability",
            [("node_id", node_id.to_string()), ("reason", "empty_sort_expressions".to_string())],
        );
    }
    for sort_expr in &sort.spec.expressions {
        if let Err(rejection) = validate_column_expr_capability(&sort_expr.expr) {
            return expression_rejection_decision(
                node_id,
                "unsupported_sort_expression_capability",
                &sort_expr.expr,
                rejection,
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn limit_node_executable_decision(plan: &QueryPlan, node_id: NodeId, limit: &LimitNode) -> CapabilityDecision {
    if limit.spec.skip.checked_add(limit.spec.fetch).is_none() {
        return CapabilityDecision::rejected(
            "unsupported_limit_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "skip_fetch_overflow".to_string()),
                ("skip", limit.spec.skip.to_string()),
                ("fetch", limit.spec.fetch.to_string()),
            ],
        );
    }
    if !matches!(plan.node(limit.input).map(|node| &node.kind), Some(PlanNode::Sort(_))) {
        return CapabilityDecision::rejected(
            "unsupported_limit_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("reason", "limit_not_directly_above_sort".to_string()),
                ("input", limit.input.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn join_node_executable_decision(node_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    let Some(capability) = bounded_join_node_capability(join) else {
        return CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("key_count", join.spec.keys.len().to_string()),
                ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                ("build_side", join.spec.build_side.to_string()),
            ],
        );
    };
    if let Some(predicate) = &join.spec.residual_predicate
        && let Err(rejection) = validate_join_residual_expr_capability(predicate)
    {
        return expression_rejection_decision(
            node_id,
            "unsupported_join_residual_expression_capability",
            predicate,
            rejection,
        );
    }
    if matches!(capability, BoundedJoinNodeCapability::KeyedFilteredSemiAnti)
        && join.spec.output_projection.iter().any(|column| column.side != JoinSide::Left)
    {
        return CapabilityDecision::rejected(
            "unsupported_join_node_capability",
            [
                ("node_id", node_id.to_string()),
                ("join_type", join.spec.join_type.to_string()),
                ("reason", "semi_anti_projects_non_left_column".to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpressionCapabilityRejection {
    reason: &'static str,
    expression_family: &'static str,
}

fn expression_rejection_decision(
    node_id: NodeId,
    reason: &'static str,
    expression: &Expr,
    rejection: ExpressionCapabilityRejection,
) -> CapabilityDecision {
    CapabilityDecision::rejected(
        reason,
        [
            ("node_id", node_id.to_string()),
            ("reason", rejection.reason.to_string()),
            ("expression_family", rejection.expression_family.to_string()),
            ("expression", expression.to_string()),
        ],
    )
}

fn validation_result_decision(
    result: crate::Result<()>,
    reason: &'static str,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityDecision {
    match result {
        Ok(()) => CapabilityDecision::Satisfied,
        Err(error) => validation_error_decision(reason, error, facts),
    }
}

fn validation_error_decision(
    reason: &'static str,
    error: QueryError,
    facts: impl IntoIterator<Item = (&'static str, String)>,
) -> CapabilityDecision {
    let mut facts = facts.into_iter().collect::<BTreeMap<_, _>>();
    facts.insert("error_code", error.code().as_str().to_string());
    facts.insert("error_operation", error.operation().to_string());
    facts.insert("error_message", error.message().to_string());
    if let Some(error_reason) = error.fact_value("reason") {
        facts.insert("error_reason", error_reason.to_string());
    }
    CapabilityDecision::Rejected { reason, facts }
}

fn validate_aggregate_expr_capability(
    aggregate_expr: &AggregateExpr,
    is_scalar_aggregate: bool,
) -> Result<(), ExpressionCapabilityRejection> {
    if is_scalar_aggregate
        && aggregate_expr.function == AggregateFunction::Count
        && is_count_star_expr(&aggregate_expr.expr)
    {
        return Ok(());
    }
    validate_column_expr_capability(&aggregate_expr.expr)
}

fn is_count_star_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value) if !matches!(value, LiteralValue::Null(_)))
        || matches!(expr, Expr::Alias { expr, .. } if is_count_star_expr(expr))
}

fn validate_named_column_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_named_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

fn validate_column_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Alias { expr, .. } => validate_column_expr_capability(expr),
        _ => Err(ExpressionCapabilityRejection {
            reason: "requires_column_expression",
            expression_family: expr_family(expr),
        }),
    }
}

fn validate_general_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(_) => Ok(()),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_general_expr_capability(expr)
        },
        Expr::Cast { expr, .. } | Expr::Extract { expr, .. } => validate_general_expr_capability(expr),
        Expr::Binary { left, op, right } if expr_is_interval_literal(left) || expr_is_interval_literal(right) => {
            validate_interval_binary_expr_capability(left, *op, right)
        },
        Expr::Binary { left, right, .. } => {
            validate_general_expr_capability(left)?;
            validate_general_expr_capability(right)
        },
        Expr::Like { expr, escape, .. } => {
            validate_general_expr_capability(expr)?;
            if matches!(escape, Some(escape) if escape.len() > 1 || !escape.is_ascii()) {
                return Err(ExpressionCapabilityRejection {
                    reason: "like_escape_must_be_empty_or_one_ascii_byte",
                    expression_family: "like",
                });
            }
            Ok(())
        },
        Expr::Substring { expr, start, length } => {
            validate_general_expr_capability(expr)?;
            if *start < 1 {
                return Err(ExpressionCapabilityRejection {
                    reason: "substring_start_must_be_positive",
                    expression_family: "substring",
                });
            }
            if let Some(length) = length
                && (*length < 0 || start.checked_sub(1).and_then(|start| start.checked_add(*length)).is_none())
            {
                return Err(ExpressionCapabilityRejection {
                    reason: "substring_length_must_be_non_negative_and_in_range",
                    expression_family: "substring",
                });
            }
            Ok(())
        },
        Expr::InList { expr, list, .. } => {
            validate_general_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::Case { branches, else_expr } => {
            if branches.is_empty() {
                return Err(ExpressionCapabilityRejection {
                    reason: "case_requires_when_branch",
                    expression_family: "case",
                });
            }
            for branch in branches {
                validate_general_expr_capability(&branch.when)?;
                validate_general_expr_capability(&branch.then)?;
            }
            validate_general_expr_capability(else_expr)
        },
    }
}

fn validate_join_residual_expr_capability(expr: &Expr) -> Result<(), ExpressionCapabilityRejection> {
    match expr {
        Expr::Column(column)
            if matches!(column.qualifier(), Some(crate::expr::ColumnQualifier::Input { index: 0 | 1 })) =>
        {
            Ok(())
        },
        Expr::Column(_) => Err(ExpressionCapabilityRejection {
            reason: "join_residual_column_must_be_input_qualified",
            expression_family: "column",
        }),
        Expr::Literal(value) => validate_literal_capability(value, "literal"),
        Expr::Not(expr) | Expr::IsNull(expr) | Expr::IsNotNull(expr) | Expr::Alias { expr, .. } => {
            validate_join_residual_expr_capability(expr)
        },
        Expr::Cast { expr, .. } => validate_join_residual_expr_capability(expr),
        Expr::Binary { left, right, .. } => {
            validate_join_residual_expr_capability(left)?;
            validate_join_residual_expr_capability(right)
        },
        Expr::InList { expr, list, .. } => {
            validate_join_residual_expr_capability(expr)?;
            validate_in_list_capability(list, "in_list")
        },
        Expr::Like { .. } | Expr::Substring { .. } | Expr::Case { .. } | Expr::Extract { .. } => {
            Err(ExpressionCapabilityRejection {
                reason: "join_residual_expression_family_not_supported",
                expression_family: expr_family(expr),
            })
        },
    }
}

fn validate_in_list_capability(
    list: &[Expr],
    expression_family: &'static str,
) -> Result<(), ExpressionCapabilityRejection> {
    if list.is_empty() {
        return Err(ExpressionCapabilityRejection { reason: "in_list_must_not_be_empty", expression_family });
    }
    for item in list {
        let Expr::Literal(value) = item else {
            return Err(ExpressionCapabilityRejection { reason: "in_list_values_must_be_literals", expression_family });
        };
        validate_literal_capability(value, expression_family)?;
        if matches!(value, LiteralValue::Null(_)) {
            return Err(ExpressionCapabilityRejection {
                reason: "in_list_null_semantics_not_modeled",
                expression_family,
            });
        }
    }
    Ok(())
}

fn validate_literal_capability(
    value: &LiteralValue,
    expression_family: &'static str,
) -> Result<(), ExpressionCapabilityRejection> {
    if matches!(value, LiteralValue::IntervalMonthDayNano { .. }) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_literal_requires_datetime_interval_arithmetic",
            expression_family,
        });
    }
    Ok(())
}

fn validate_interval_binary_expr_capability(
    left: &Expr,
    op: BinaryOp,
    right: &Expr,
) -> Result<(), ExpressionCapabilityRejection> {
    let left_interval = expr_is_interval_literal(left);
    let right_interval = expr_is_interval_literal(right);
    if left_interval == right_interval {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_arithmetic_requires_exactly_one_interval_literal",
            expression_family: "binary",
        });
    }
    if left_interval && op != BinaryOp::Add {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_left_operand_requires_add",
            expression_family: "binary",
        });
    }
    if right_interval && !matches!(op, BinaryOp::Add | BinaryOp::Subtract) {
        return Err(ExpressionCapabilityRejection {
            reason: "interval_right_operand_requires_add_or_subtract",
            expression_family: "binary",
        });
    }
    validate_general_expr_capability(if left_interval { right } else { left })
}

fn expr_is_interval_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::IntervalMonthDayNano { .. }))
}

fn expr_family(expr: &Expr) -> &'static str {
    match expr {
        Expr::Column(_) => "column",
        Expr::Literal(_) => "literal",
        Expr::Not(_) => "not",
        Expr::IsNull(_) => "is_null",
        Expr::IsNotNull(_) => "is_not_null",
        Expr::Cast { .. } => "cast",
        Expr::Binary { .. } => "binary",
        Expr::Like { .. } => "like",
        Expr::Substring { .. } => "substring",
        Expr::InList { .. } => "in_list",
        Expr::Case { .. } => "case",
        Expr::Extract { .. } => "extract",
        Expr::Alias { .. } => "alias",
    }
}

fn produced_properties(node: &QueryNode) -> Vec<ProducedProperty> {
    let mut properties = Vec::new();
    if node.facts.placement == crate::plan::DataPlacement::Device {
        properties.push(ProducedProperty::ResidentFrame);
    }
    match &node.kind {
        PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => {
            properties.push(ProducedProperty::ScalarFrame);
        },
        PlanNode::Aggregate(_) => properties.push(ProducedProperty::GroupedFrame),
        PlanNode::Sort(_) => properties.push(ProducedProperty::OrderedFrame),
        _ => {},
    }
    properties
}

fn node_local_requirements(kind: &PlanNode) -> Vec<ReservationRequirement> {
    match kind {
        PlanNode::Aggregate(_) => vec![ReservationRequirement::modeled(
            ReservationKind::OperatorOutput,
            "operator_output",
            "aggregate_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        )],
        PlanNode::Join(join) if bounded_root_hash_join_compatible(join) => vec![ReservationRequirement::modeled(
            ReservationKind::HashTable,
            "join_hash_table",
            "hash_join_pre_build_reservation_size_bytes",
            ContractQuality::ExactPreflight,
        )],
        PlanNode::Join(join) if bounded_root_filtered_join_compatible(join) => vec![
            ReservationRequirement::modeled(
                ReservationKind::HashTable,
                "join_hash_table",
                "hash_join_pre_build_reservation_size_bytes",
                ContractQuality::ExactPreflight,
            ),
            ReservationRequirement::modeled(
                ReservationKind::FilteredProbeState,
                "join_filtered_probe_state",
                "filtered_join_retained_probe_state",
                ContractQuality::ObservedRetainedState,
            ),
        ],
        PlanNode::Join(join)
            if matches!(
                BoundedKeylessConditionalJoinType::try_from_join(join),
                Some(BoundedKeylessConditionalJoinType::Full)
            ) =>
        {
            vec![ReservationRequirement::modeled(
                ReservationKind::MixedSizeData,
                "join_mixed_size_data",
                "mixed_keyed_residual_or_conditional_full_retained_size_data",
                ContractQuality::ObservedRetainedState,
            )]
        },
        PlanNode::Join(join) if join.spec.residual_predicate.is_some() => vec![ReservationRequirement::modeled(
            ReservationKind::MixedSizeData,
            "join_mixed_size_data",
            "mixed_keyed_residual_or_conditional_full_retained_size_data",
            ContractQuality::ObservedRetainedState,
        )],
        _ => Vec::new(),
    }
}

fn analyze_input_edges(plan: &QueryPlan, consumer_id: NodeId, kind: &PlanNode) -> Vec<EdgeContract> {
    kind.input_ids()
        .iter()
        .enumerate()
        .map(|(input_index, producer)| {
            let consumer = EdgeConsumer::Node(consumer_id);
            let input_role = input_role(kind, input_index);
            let contract_kinds = edge_contract_kinds(plan, *producer, consumer, kind, input_role);
            let required_properties = vec![RequiredProperty::ResidentFrame];
            let reservation_requirements = edge_requirements(plan, *producer, kind, input_role);
            let decision = edge_decision(plan, *producer, consumer, &required_properties);
            EdgeContract {
                producer: *producer,
                consumer,
                contract_kinds,
                input_role,
                required_properties,
                reservation_requirements,
                decision,
            }
        })
        .collect()
}

fn edge_contract_kinds(
    plan: &QueryPlan,
    producer: NodeId,
    consumer: EdgeConsumer,
    consumer_kind: &PlanNode,
    input_role: InputRole,
) -> Vec<EdgeContractKind> {
    let mut kinds = Vec::new();
    if let Some(node) = plan.node(producer) {
        if let PlanNode::Source(source) = &node.kind {
            kinds.push(EdgeContractKind::ResidentSourceInput { format: source_format_capability(source) });
        }
        if matches!(node.kind, PlanNode::Join(_)) {
            kinds.push(EdgeContractKind::JoinMaterialization);
        }
    }
    if matches!(consumer, EdgeConsumer::Node(_)) {
        kinds.push(EdgeContractKind::ResidentFrameFlow);
    }
    if matches!(consumer_kind, PlanNode::Aggregate(_)) {
        kinds.push(EdgeContractKind::ResidentAggregateInput);
    }
    match input_role {
        InputRole::JoinBuildInput { side } => kinds.push(EdgeContractKind::JoinBuildInput { side }),
        InputRole::JoinProbeInput { side } => kinds.push(EdgeContractKind::JoinProbeInput { side }),
        _ => {},
    }
    if matches!(consumer_kind, PlanNode::Sort(_)) {
        kinds.push(EdgeContractKind::SortInput);
    }
    kinds
}

fn edge_decision(
    plan: &QueryPlan,
    producer: NodeId,
    consumer: EdgeConsumer,
    required_properties: &[RequiredProperty],
) -> CapabilityDecision {
    let Some(producer_node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "missing_edge_producer",
            [("producer", producer.to_string()), ("consumer", edge_consumer_fact(consumer))],
        );
    };
    let produced = produced_properties(producer_node);
    if let Some(required) =
        required_properties.iter().find(|required| !edge_property_is_satisfied(&produced, **required))
    {
        return CapabilityDecision::rejected(
            "missing_edge_required_property",
            [
                ("producer", producer.to_string()),
                ("consumer", edge_consumer_fact(consumer)),
                ("required_property", required_property_fact(*required).to_string()),
                ("producer_placement", producer_node.facts.placement.to_string()),
                ("producer_operator", node_operator_fact(&producer_node.kind).to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn edge_property_is_satisfied(produced: &[ProducedProperty], required: RequiredProperty) -> bool {
    match required {
        RequiredProperty::ResidentFrame => produced.contains(&ProducedProperty::ResidentFrame),
    }
}

fn edge_consumer_fact(consumer: EdgeConsumer) -> String {
    match consumer {
        EdgeConsumer::Node(node) => node.to_string(),
        EdgeConsumer::Sink(ResultSinkKind::HostArrow) => "sink:host_arrow".to_string(),
        EdgeConsumer::Sink(ResultSinkKind::GpuResident) => "sink:gpu_resident".to_string(),
    }
}

fn required_property_fact(property: RequiredProperty) -> &'static str {
    match property {
        RequiredProperty::ResidentFrame => "resident_frame",
    }
}

fn node_operator_fact(kind: &PlanNode) -> &'static str {
    match kind {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
    }
}

fn input_role(kind: &PlanNode, input_index: usize) -> InputRole {
    match kind {
        PlanNode::Aggregate(_) => InputRole::AggregateInput,
        PlanNode::Sort(_) => InputRole::SortInput,
        PlanNode::Limit(_) => InputRole::LimitInput,
        PlanNode::Join(join) => {
            let side = if input_index == 0 { JoinSide::Left } else { JoinSide::Right };
            if side == join.spec.build_side {
                InputRole::JoinBuildInput { side }
            } else {
                InputRole::JoinProbeInput { side }
            }
        },
        _ => InputRole::UnaryInput,
    }
}

fn edge_requirements(
    plan: &QueryPlan,
    producer: NodeId,
    consumer_kind: &PlanNode,
    input_role: InputRole,
) -> Vec<ReservationRequirement> {
    let mut requirements = Vec::new();
    append_source_read_edge_requirement(plan, producer, &mut requirements);
    if matches!(input_role, InputRole::JoinBuildInput { .. }) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::BuildSide,
            "join_build_side",
            "query_engine_build_side_source_reservation",
            ContractQuality::ExactPreflight,
        ));
    }
    if matches!(input_role, InputRole::JoinProbeInput { .. }) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::ProbeSide,
            "join_probe_side",
            "query_engine_probe_side_source_reservation",
            ContractQuality::QueryEngineOwnedOrdering,
        ));
    }
    append_producer_output_edge_requirements(plan, producer, &mut requirements);
    if matches!(consumer_kind, PlanNode::Sort(_)) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::FinalSort,
            "join_pipeline_final_sort",
            "final_sort_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
    requirements
}

fn append_source_read_edge_requirement(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Source(source)) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
    {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::SourceRead,
            "source_read",
            "source_contract",
            ContractQuality::HardBudgetedReportWithUnresolvedPreflight,
        ));
    }
}

fn append_producer_output_edge_requirements(
    plan: &QueryPlan,
    producer: NodeId,
    requirements: &mut Vec<ReservationRequirement>,
) {
    if matches!(plan.node(producer).map(|node| &node.kind), Some(PlanNode::Join(_))) {
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::GatherMap,
            "join_gather_map",
            "join_gather_map_reservation_size_bytes",
            ContractQuality::ExactPreflight,
        ));
        requirements.push(ReservationRequirement::modeled(
            ReservationKind::JoinOutput,
            "join_output",
            "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            ContractQuality::ExactPreflight,
        ));
    }
}

fn analyze_sink_edge(plan: &QueryPlan) -> EdgeContract {
    let producer = plan.sink().input();
    let consumer = match plan.sink() {
        crate::plan::ResultSink::HostArrow { .. } => EdgeConsumer::Sink(ResultSinkKind::HostArrow),
        crate::plan::ResultSink::GpuResident { .. } => EdgeConsumer::Sink(ResultSinkKind::GpuResident),
    };
    let mut contract_kinds = Vec::new();
    if let Some(node) = plan.node(producer) {
        if let PlanNode::Source(source) = &node.kind {
            contract_kinds.push(EdgeContractKind::ResidentSourceInput { format: source_format_capability(source) });
        }
        if matches!(node.kind, PlanNode::Join(_)) {
            contract_kinds.push(EdgeContractKind::JoinMaterialization);
        }
    }
    contract_kinds.push(EdgeContractKind::ExplicitSinkMaterialization);
    let required_properties = vec![RequiredProperty::ResidentFrame];
    let mut reservation_requirements = Vec::new();
    append_source_read_edge_requirement(plan, producer, &mut reservation_requirements);
    append_producer_output_edge_requirements(plan, producer, &mut reservation_requirements);
    reservation_requirements.push(ReservationRequirement::modeled(
        ReservationKind::Materialization,
        "host_materialization",
        "query_engine_explicit_sink_materialization_reservation",
        ContractQuality::QueryEngineOwnedBoundary,
    ));
    let decision = edge_decision(plan, producer, consumer, &required_properties);
    EdgeContract {
        producer,
        consumer,
        contract_kinds,
        input_role: InputRole::SinkInput,
        required_properties,
        reservation_requirements,
        decision,
    }
}

fn csv_join<I, T>(values: I) -> String
where
    I: IntoIterator<Item = T>,
    T: ToString,
{
    let mut output = String::new();
    for value in values {
        if !output.is_empty() {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

fn analyze_invariants(plan: &QueryPlan, edges: &[EdgeContract]) -> Vec<SubgraphInvariant> {
    let mut invariants = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
        let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
        let invariant_id = InvariantId(invariants.len());
        let decision = solve_build_before_probe_invariant(plan, node.id, build_input, probe_input);
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::BuildBeforeProbe { join_id: node.id },
            nodes: vec![build_input, probe_input, node.id],
            required_schedule: ScheduleConstraint::BuildBeforeProbe,
            required_resources: vec![
                ResidentResource::JoinBuildInput { join_id: node.id, input: build_input },
                ResidentResource::JoinProbeInput { join_id: node.id, input: probe_input },
            ],
            decision,
        });
        if join.spec.residual_predicate.is_some() {
            let invariant_id = InvariantId(invariants.len());
            let decision = solve_residual_uses_both_inputs_invariant(node.id, join);
            invariants.push(SubgraphInvariant {
                invariant_id,
                kind: SubgraphInvariantKind::ResidualUsesBothInputs { join_id: node.id },
                nodes: vec![join.inputs[0], join.inputs[1], node.id],
                required_schedule: ScheduleConstraint::None,
                required_resources: Vec::new(),
                decision,
            });
        }
    }
    append_shared_resident_subplan_invariants(plan, edges, &mut invariants);
    append_aggregate_derivation_invariants(plan, &mut invariants);
    append_projection_transparent_residency_invariants(plan, &mut invariants);
    append_final_barrier_order_invariants(plan, &mut invariants);

    if reachable_join_count_from_sink(plan).unwrap_or(0) > 1 {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::TopologicalBuildProbeSchedule,
            nodes: plan
                .nodes()
                .iter()
                .filter_map(|node| matches!(node.kind, PlanNode::Join(_)).then_some(node.id))
                .collect(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: solve_topological_build_probe_schedule_invariant(plan),
        });
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
            nodes: collect_reachable_node_ids(plan).map(|nodes| nodes.into_iter().collect()).unwrap_or_default(),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: solve_resident_lifetime_invariant(plan, edges),
        });
    }

    if matches!(closed_multi_join_subgraph_structure_decision(plan), CapabilityDecision::Satisfied)
        && let Ok(root_join_id) = post_root_join_id(plan)
    {
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ClosedJoinSubtree { root_join_id },
            nodes: collect_reachable_node_ids_from(plan, root_join_id)
                .map(|nodes| nodes.into_iter().collect())
                .unwrap_or_else(|_| vec![root_join_id]),
            required_schedule: ScheduleConstraint::Topological,
            required_resources: Vec::new(),
            decision: CapabilityDecision::Satisfied,
        });
    }
    invariants
}

fn append_shared_resident_subplan_invariants(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    invariants: &mut Vec<SubgraphInvariant>,
) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    for (producer, consumers) in consumer_nodes_for_reachable(plan, &reachable) {
        if consumers.len() <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        let mut nodes = Vec::with_capacity(consumers.len() + 1);
        nodes.push(producer);
        nodes.extend(consumers.iter().copied());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::SharedResidentSubplan,
            nodes,
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(producer)],
            decision: solve_shared_resident_subplan_invariant(plan, edges, producer, &consumers),
        });
    }
}

fn append_aggregate_derivation_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    let Ok(reachable) = collect_reachable_node_ids(plan) else {
        return;
    };
    let consumers = consumer_nodes_for_reachable(plan, &reachable);
    for node in plan.nodes() {
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            continue;
        };
        if !aggregate.spec.group_exprs.is_empty() {
            continue;
        }
        let Some((shared_producer, projection_count)) = peel_projection_chain_for_capability(plan, aggregate.input)
        else {
            continue;
        };
        if projection_count == 0 || consumers.get(&shared_producer).map_or(0, Vec::len) <= 1 {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::AggregateDerivesFromSharedProducer,
            nodes: vec![shared_producer, aggregate.input, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(shared_producer)],
            decision: solve_aggregate_derives_from_shared_producer_invariant(plan, shared_producer, node.id),
        });
    }
}

fn append_projection_transparent_residency_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    for node in plan.nodes() {
        if !matches!(node.kind, PlanNode::Projection(_)) {
            continue;
        }
        let Some((peeled_producer, projection_count)) = peel_projection_chain_for_capability(plan, node.id) else {
            continue;
        };
        if projection_count == 0 || peeled_producer == node.id {
            continue;
        }
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::ProjectionTransparentResidency,
            nodes: vec![peeled_producer, node.id],
            required_schedule: ScheduleConstraint::None,
            required_resources: vec![ResidentResource::NodeOutput(peeled_producer)],
            decision: solve_projection_transparent_residency_invariant(plan, peeled_producer, node.id),
        });
    }
}

fn append_final_barrier_order_invariants(plan: &QueryPlan, invariants: &mut Vec<SubgraphInvariant>) {
    for node in plan.nodes() {
        let PlanNode::Sort(sort) = &node.kind else {
            continue;
        };
        let invariant_id = InvariantId(invariants.len());
        invariants.push(SubgraphInvariant {
            invariant_id,
            kind: SubgraphInvariantKind::FinalBarrierOrder,
            nodes: vec![sort.input, node.id],
            required_schedule: ScheduleConstraint::Barrier,
            required_resources: vec![ResidentResource::NodeOutput(sort.input)],
            decision: solve_final_barrier_order_invariant(plan, node.id, sort.input),
        });
    }
}

fn analyze_closed_multi_join_subgraph_admission(
    plan: &QueryPlan,
    nodes: &[NodeCapability],
    edges: &[EdgeContract],
    invariants: &[SubgraphInvariant],
) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    if let Some(rejection) = first_reachable_node_rejection(nodes, &reachable) {
        return rejection;
    }
    if let Some(rejection) = first_reachable_edge_rejection(edges, &reachable) {
        return rejection;
    }
    let structure_decision = closed_multi_join_subgraph_structure_decision(plan);
    if matches!(structure_decision, CapabilityDecision::Rejected { .. }) {
        return structure_decision;
    }
    for kind in [
        InvariantSelector::TopologicalBuildProbeSchedule,
        InvariantSelector::ResidentLifetimeDominatesConsumers,
        InvariantSelector::ClosedJoinSubtree,
    ] {
        match solved_invariant_decision(invariants, kind) {
            Some(CapabilityDecision::Satisfied) => {},
            Some(CapabilityDecision::Rejected { reason, facts }) => {
                return CapabilityDecision::Rejected { reason: *reason, facts: facts.clone() };
            },
            Some(CapabilityDecision::Conditional { waiting_on }) => {
                return CapabilityDecision::Conditional { waiting_on: waiting_on.clone() };
            },
            None => {
                return CapabilityDecision::rejected(
                    "missing_closed_multi_join_subgraph_invariant",
                    [("invariant", kind.as_str().to_string())],
                );
            },
        }
    }
    CapabilityDecision::Satisfied
}

#[derive(Debug, Clone, Copy)]
enum InvariantSelector {
    TopologicalBuildProbeSchedule,
    ResidentLifetimeDominatesConsumers,
    ClosedJoinSubtree,
}

impl InvariantSelector {
    const fn as_str(self) -> &'static str {
        match self {
            Self::TopologicalBuildProbeSchedule => "topological_build_probe_schedule",
            Self::ResidentLifetimeDominatesConsumers => "resident_lifetime_dominates_consumers",
            Self::ClosedJoinSubtree => "closed_join_subtree",
        }
    }
}

fn solved_invariant_decision(
    invariants: &[SubgraphInvariant],
    selector: InvariantSelector,
) -> Option<&CapabilityDecision> {
    invariants
        .iter()
        .find(|invariant| match (&invariant.kind, selector) {
            (
                SubgraphInvariantKind::TopologicalBuildProbeSchedule,
                InvariantSelector::TopologicalBuildProbeSchedule,
            ) => true,
            (
                SubgraphInvariantKind::ResidentLifetimeDominatesConsumers,
                InvariantSelector::ResidentLifetimeDominatesConsumers,
            ) => true,
            (SubgraphInvariantKind::ClosedJoinSubtree { .. }, InvariantSelector::ClosedJoinSubtree) => true,
            _ => false,
        })
        .map(|invariant| &invariant.decision)
}

fn first_reachable_node_rejection(
    nodes: &[NodeCapability],
    reachable: &BTreeSet<NodeId>,
) -> Option<CapabilityDecision> {
    nodes.iter().find_map(|node| {
        if !reachable.contains(&node.node_id) {
            return None;
        }
        match &node.executable {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason: *reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}

fn first_reachable_edge_rejection(edges: &[EdgeContract], reachable: &BTreeSet<NodeId>) -> Option<CapabilityDecision> {
    edges.iter().find_map(|edge| {
        if !reachable.contains(&edge.producer) {
            return None;
        }
        if let EdgeConsumer::Node(consumer) = edge.consumer
            && !reachable.contains(&consumer)
        {
            return None;
        }
        match &edge.decision {
            CapabilityDecision::Rejected { reason, facts } => {
                Some(CapabilityDecision::Rejected { reason: *reason, facts: facts.clone() })
            },
            _ => None,
        }
    })
}

fn closed_multi_join_subgraph_structure_decision(plan: &QueryPlan) -> CapabilityDecision {
    let root_join_id = match post_root_join_id(plan) {
        Ok(root_join_id) => root_join_id,
        Err(decision) => return decision,
    };
    let mut join_count = 0;
    if let Err(decision) = validate_closed_multi_join_subtree(plan, root_join_id, &mut join_count) {
        return decision;
    }
    if join_count <= 1 {
        return CapabilityDecision::rejected(
            "closed_multi_join_subgraph_requires_multiple_joins",
            [("root_join_id", root_join_id.to_string()), ("join_count", join_count.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

fn validate_closed_multi_join_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
    join_count: &mut usize,
) -> Result<(), CapabilityDecision> {
    let Some(node) = plan.node(node_id) else {
        return Err(CapabilityDecision::rejected(
            "closed_multi_join_subgraph_missing_node",
            [("node_id", node_id.to_string())],
        ));
    };
    match &node.kind {
        PlanNode::Join(join) => {
            if bounded_multi_join_node_capability(join).is_none() {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_unsupported_join",
                    [
                        ("node_id", node_id.to_string()),
                        ("join_type", join.spec.join_type.to_string()),
                        ("key_count", join.spec.keys.len().to_string()),
                        ("has_residual", join.spec.residual_predicate.is_some().to_string()),
                        ("build_side", join.spec.build_side.to_string()),
                    ],
                ));
            }
            *join_count += 1;
            validate_closed_multi_join_subtree(plan, join.inputs[0], join_count)?;
            validate_closed_multi_join_subtree(plan, join.inputs[1], join_count)?;
            Ok(())
        },
        PlanNode::Source(_) => Ok(()),
        PlanNode::Filter(filter) => validate_closed_multi_join_subtree(plan, filter.input, join_count),
        PlanNode::Projection(projection) => validate_closed_multi_join_subtree(plan, projection.input, join_count),
        PlanNode::Aggregate(_) | PlanNode::Sort(_) | PlanNode::Limit(_) => Err(CapabilityDecision::rejected(
            "closed_multi_join_subgraph_unsupported_node",
            [("node_id", node_id.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        )),
    }
}

fn post_root_join_id(plan: &QueryPlan) -> Result<NodeId, CapabilityDecision> {
    let mut current = plan.sink().input();
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return Err(CapabilityDecision::rejected(
                "closed_multi_join_subgraph_cycle",
                [("node_id", current.to_string())],
            ));
        }
        let Some(node) = plan.node(current) else {
            return Err(CapabilityDecision::rejected(
                "closed_multi_join_subgraph_missing_node",
                [("node_id", current.to_string())],
            ));
        };
        match &node.kind {
            PlanNode::Join(_) => return Ok(current),
            PlanNode::Filter(filter) => current = filter.input,
            PlanNode::Projection(projection) => current = projection.input,
            PlanNode::Aggregate(aggregate) => current = aggregate.input,
            PlanNode::Sort(sort) => current = sort.input,
            PlanNode::Limit(limit) => current = limit.input,
            PlanNode::Source(_) => {
                return Err(CapabilityDecision::rejected(
                    "closed_multi_join_subgraph_missing_root_join",
                    [("sink_input", plan.sink().input().to_string())],
                ));
            },
        }
    }
}

fn solve_build_before_probe_invariant(
    plan: &QueryPlan,
    join_id: NodeId,
    build_input: NodeId,
    probe_input: NodeId,
) -> CapabilityDecision {
    let build_nodes = match collect_reachable_node_ids_from(plan, build_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    let probe_nodes = match collect_reachable_node_ids_from(plan, probe_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if build_nodes.contains(&join_id) || probe_nodes.contains(&join_id) {
        return CapabilityDecision::rejected(
            "build_before_probe_cycle",
            [
                ("join_id", join_id.to_string()),
                ("build_input", build_input.to_string()),
                ("probe_input", probe_input.to_string()),
            ],
        );
    }
    CapabilityDecision::Satisfied
}

fn solve_topological_build_probe_schedule_invariant(plan: &QueryPlan) -> CapabilityDecision {
    let mut scheduler = GraphScheduleProof::new(plan);
    scheduler.schedule_from(plan.sink().input()).map_or_else(|decision| decision, |()| CapabilityDecision::Satisfied)
}

fn solve_resident_lifetime_invariant(plan: &QueryPlan, edges: &[EdgeContract]) -> CapabilityDecision {
    let reachable = match collect_reachable_node_ids(plan) {
        Ok(reachable) => reachable,
        Err(decision) => return decision,
    };
    let mut consumers = consumer_nodes_for_reachable(plan, &reachable);
    consumers.entry(plan.sink().input()).or_default();

    for producer in reachable {
        let Some(node) = plan.node(producer) else {
            return CapabilityDecision::rejected("resident_lifetime_missing_node", [("node_id", producer.to_string())]);
        };
        if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
            return CapabilityDecision::rejected(
                "resident_lifetime_requires_resident_frame",
                [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
            );
        }
        for edge in edges.iter().filter(|edge| edge.producer == producer) {
            if let EdgeConsumer::Node(consumer) = edge.consumer
                && !consumers.get(&producer).is_some_and(|values| values.contains(&consumer))
            {
                continue;
            }
            if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
                return CapabilityDecision::rejected(
                    "resident_lifetime_edge_rejected",
                    [("producer", producer.to_string()), ("consumer", edge_consumer_fact(edge.consumer))],
                );
            }
        }
    }
    CapabilityDecision::Satisfied
}

fn solve_shared_resident_subplan_invariant(
    plan: &QueryPlan,
    edges: &[EdgeContract],
    producer: NodeId,
    consumers: &[NodeId],
) -> CapabilityDecision {
    let Some(node) = plan.node(producer) else {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_missing_node",
            [("node_id", producer.to_string())],
        );
    };
    if !produced_properties(node).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "shared_resident_subplan_requires_resident_frame",
            [("node_id", producer.to_string()), ("operator", node_operator_fact(&node.kind).to_string())],
        );
    }
    for consumer in consumers {
        let Some(edge) =
            edges.iter().find(|edge| edge.producer == producer && edge.consumer == EdgeConsumer::Node(*consumer))
        else {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_missing_edge",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        };
        if matches!(edge.decision, CapabilityDecision::Rejected { .. }) {
            return CapabilityDecision::rejected(
                "shared_resident_subplan_edge_rejected",
                [("producer", producer.to_string()), ("consumer", consumer.to_string())],
            );
        }
    }
    CapabilityDecision::Satisfied
}

fn solve_aggregate_derives_from_shared_producer_invariant(
    plan: &QueryPlan,
    shared_producer: NodeId,
    aggregate_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(shared_producer) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_node",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    let Some(aggregate) = plan.node(aggregate_id) else {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_missing_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    };
    if !matches!(aggregate.kind, PlanNode::Aggregate(_)) {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_consumer_not_aggregate",
            [("shared_producer", shared_producer.to_string()), ("aggregate_id", aggregate_id.to_string())],
        );
    }
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame) {
        return CapabilityDecision::rejected(
            "aggregate_shared_producer_requires_resident_frame",
            [("shared_producer", shared_producer.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

fn solve_projection_transparent_residency_invariant(
    plan: &QueryPlan,
    peeled_producer: NodeId,
    projection_id: NodeId,
) -> CapabilityDecision {
    let Some(producer) = plan.node(peeled_producer) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_producer",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    let Some(projection) = plan.node(projection_id) else {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_missing_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    };
    if !matches!(projection.kind, PlanNode::Projection(_)) {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_consumer_not_projection",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    if !produced_properties(producer).contains(&ProducedProperty::ResidentFrame)
        || !produced_properties(projection).contains(&ProducedProperty::ResidentFrame)
    {
        return CapabilityDecision::rejected(
            "projection_transparent_residency_requires_resident_frames",
            [("peeled_producer", peeled_producer.to_string()), ("projection_id", projection_id.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

fn solve_final_barrier_order_invariant(plan: &QueryPlan, sort_id: NodeId, sort_input: NodeId) -> CapabilityDecision {
    let input_nodes = match collect_reachable_node_ids_from(plan, sort_input) {
        Ok(nodes) => nodes,
        Err(decision) => return decision,
    };
    if input_nodes.contains(&sort_id) {
        return CapabilityDecision::rejected(
            "final_barrier_order_cycle",
            [("sort_id", sort_id.to_string()), ("sort_input", sort_input.to_string())],
        );
    }
    CapabilityDecision::Satisfied
}

fn solve_residual_uses_both_inputs_invariant(join_id: NodeId, join: &JoinNode) -> CapabilityDecision {
    let Some(predicate) = &join.spec.residual_predicate else {
        return CapabilityDecision::Satisfied;
    };
    let references_left = expr_references_input(predicate, 0);
    let references_right = expr_references_input(predicate, 1);
    if references_left && references_right {
        return CapabilityDecision::Satisfied;
    }
    CapabilityDecision::rejected(
        "residual_does_not_reference_both_inputs",
        [
            ("join_id", join_id.to_string()),
            ("references_left", references_left.to_string()),
            ("references_right", references_right.to_string()),
            ("expression", predicate.to_string()),
        ],
    )
}

fn expr_references_input(expr: &Expr, input_index: usize) -> bool {
    match expr {
        Expr::Column(column) => {
            matches!(column.qualifier(), Some(ColumnQualifier::Input { index }) if *index == input_index)
        },
        Expr::Literal(_) => false,
        Expr::Not(expr)
        | Expr::IsNull(expr)
        | Expr::IsNotNull(expr)
        | Expr::Cast { expr, .. }
        | Expr::Alias { expr, .. } => expr_references_input(expr, input_index),
        Expr::Binary { left, right, .. } => {
            expr_references_input(left, input_index) || expr_references_input(right, input_index)
        },
        Expr::Like { expr, .. } | Expr::Substring { expr, .. } | Expr::Extract { expr, .. } => {
            expr_references_input(expr, input_index)
        },
        Expr::InList { expr, list, .. } => {
            expr_references_input(expr, input_index) || list.iter().any(|item| expr_references_input(item, input_index))
        },
        Expr::Case { branches, else_expr } => {
            branches.iter().any(|branch| {
                expr_references_input(&branch.when, input_index) || expr_references_input(&branch.then, input_index)
            }) || expr_references_input(else_expr, input_index)
        },
    }
}

struct GraphScheduleProof<'a> {
    plan: &'a QueryPlan,
    state: Vec<VisitState>,
}

impl<'a> GraphScheduleProof<'a> {
    fn new(plan: &'a QueryPlan) -> Self {
        Self { plan, state: vec![VisitState::Unvisited; plan.nodes().len()] }
    }

    fn schedule_from(&mut self, node_id: NodeId) -> Result<(), CapabilityDecision> {
        let Some(node) = self.plan.node(node_id) else {
            return Err(CapabilityDecision::rejected(
                "topological_schedule_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        match self.state.get(node_id.index()).copied() {
            Some(VisitState::Done) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(CapabilityDecision::rejected(
                    "topological_schedule_cycle",
                    [("node_id", node_id.to_string())],
                ));
            },
            Some(VisitState::Unvisited) => {},
            None => {
                return Err(CapabilityDecision::rejected(
                    "topological_schedule_missing_node",
                    [("node_id", node_id.to_string())],
                ));
            },
        }
        self.state[node_id.index()] = VisitState::Visiting;
        match &node.kind {
            PlanNode::Join(join) => {
                let build_input = join.inputs[if join.spec.build_side == JoinSide::Left { 0 } else { 1 }];
                let probe_input = join.inputs[if join.spec.build_side == JoinSide::Left { 1 } else { 0 }];
                self.schedule_from(build_input)?;
                self.schedule_from(probe_input)?;
            },
            _ => {
                for input in node.input_ids() {
                    self.schedule_from(*input)?;
                }
            },
        }
        self.state[node_id.index()] = VisitState::Done;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisitState {
    Unvisited,
    Visiting,
    Done,
}

fn collect_reachable_node_ids(plan: &QueryPlan) -> Result<BTreeSet<NodeId>, CapabilityDecision> {
    collect_reachable_node_ids_from(plan, plan.sink().input())
}

fn collect_reachable_node_ids_from(plan: &QueryPlan, root: NodeId) -> Result<BTreeSet<NodeId>, CapabilityDecision> {
    let mut collector = ReachableCollector::new(plan);
    collector.collect(root)?;
    Ok(collector.reachable)
}

struct ReachableCollector<'a> {
    plan: &'a QueryPlan,
    state: Vec<VisitState>,
    reachable: BTreeSet<NodeId>,
}

impl<'a> ReachableCollector<'a> {
    fn new(plan: &'a QueryPlan) -> Self {
        Self { plan, state: vec![VisitState::Unvisited; plan.nodes().len()], reachable: BTreeSet::new() }
    }

    fn collect(&mut self, node_id: NodeId) -> Result<(), CapabilityDecision> {
        let Some(node) = self.plan.node(node_id) else {
            return Err(CapabilityDecision::rejected(
                "reachable_subgraph_missing_node",
                [("node_id", node_id.to_string())],
            ));
        };
        match self.state.get(node_id.index()).copied() {
            Some(VisitState::Done) => return Ok(()),
            Some(VisitState::Visiting) => {
                return Err(CapabilityDecision::rejected(
                    "reachable_subgraph_cycle",
                    [("node_id", node_id.to_string())],
                ));
            },
            Some(VisitState::Unvisited) => {},
            None => {
                return Err(CapabilityDecision::rejected(
                    "reachable_subgraph_missing_node",
                    [("node_id", node_id.to_string())],
                ));
            },
        }
        self.state[node_id.index()] = VisitState::Visiting;
        self.reachable.insert(node_id);
        for input in node.input_ids() {
            self.collect(*input)?;
        }
        self.state[node_id.index()] = VisitState::Done;
        Ok(())
    }
}

fn reachable_join_count_from_sink(plan: &QueryPlan) -> Result<usize, CapabilityDecision> {
    let reachable = collect_reachable_node_ids(plan)?;
    Ok(reachable
        .into_iter()
        .filter(|node_id| matches!(plan.node(*node_id).map(|node| &node.kind), Some(PlanNode::Join(_))))
        .count())
}

fn consumer_nodes_for_reachable(plan: &QueryPlan, reachable: &BTreeSet<NodeId>) -> BTreeMap<NodeId, Vec<NodeId>> {
    let mut consumers = BTreeMap::<NodeId, Vec<NodeId>>::new();
    for node_id in reachable {
        let Some(node) = plan.node(*node_id) else {
            continue;
        };
        for input in node.input_ids() {
            if reachable.contains(input) {
                consumers.entry(*input).or_default().push(*node_id);
            }
        }
    }
    consumers
}

fn peel_projection_chain_for_capability(plan: &QueryPlan, node_id: NodeId) -> Option<(NodeId, usize)> {
    let mut current = node_id;
    let mut projection_count = 0;
    let mut visited = BTreeSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        let node = plan.node(current)?;
        let PlanNode::Projection(projection) = &node.kind else {
            return Some((current, projection_count));
        };
        projection_count += 1;
        current = projection.input;
    }
}

fn collect_requirements(nodes: &[NodeCapability], edges: &[EdgeContract]) -> Vec<ReservationRequirement> {
    nodes
        .iter()
        .flat_map(|node| node.local_requirements.iter())
        .chain(edges.iter().flat_map(|edge| edge.reservation_requirements.iter()))
        .cloned()
        .collect()
}

fn compatibility_family_projection(plan: &QueryPlan) -> Option<CompatibilityFamilyProjection> {
    match analyze_bounded_join_plan_support(plan) {
        BoundedJoinPlanSupport::RootJoin => Some(CompatibilityFamilyProjection::RootJoin),
        BoundedJoinPlanSupport::SingleJoinPostRootPipeline => {
            Some(CompatibilityFamilyProjection::SingleJoinPostRootPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinInputSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinInputSubplanPipeline)
        },
        BoundedJoinPlanSupport::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline => {
            Some(CompatibilityFamilyProjection::KeylessConditionalJoinAggregateOverJoinSubplanInputsPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinTreePostRootPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinTreePostRootPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinAggregateOverJoinSubplanTopKPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinAggregateOverJoinSubplanTopKPipeline)
        },
        BoundedJoinPlanSupport::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline => {
            Some(CompatibilityFamilyProjection::MixedKeyedResidualJoinResidentGroupedAggregateInputPipeline)
        },
        BoundedJoinPlanSupport::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline => {
            Some(CompatibilityFamilyProjection::HashJoinResidentGroupedAggregateInputPostJoinAggregateTopKPipeline)
        },
        BoundedJoinPlanSupport::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline => {
            Some(CompatibilityFamilyProjection::MultiJoinResidentGroupedAggregateSubqueryFilterPipeline)
        },
        BoundedJoinPlanSupport::KeyedHashJoinResidentSubplanPipeline => {
            Some(CompatibilityFamilyProjection::KeyedHashJoinResidentSubplanPipeline)
        },
        BoundedJoinPlanSupport::Unsupported => None,
    }
}
