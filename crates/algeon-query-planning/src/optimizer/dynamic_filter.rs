//! Dynamic-filter route discovery.
//!
//! The rule matches a single-key hash join under `NullsDoNotMatch` that discards
//! probe rows matching no build row (an inner or semi join, or an anti join that
//! builds its left input), whose build key traces to one source column, or to a
//! grouping-only aggregate's key set, and whose probe key traces to a local
//! Parquet source column, to a filter above it, or to the join's own probe
//! input. In the rewrite, no node changes. The pass replaces the plan's
//! [`DynamicFilterRoute`] annotations with routes that name a producer (the hash
//! build or the aggregate key set) and a consumer; execution applies the
//! produced key payload at the consumer before rows reach the join. A route
//! changes no result because such a join keeps nothing of a probe row whose key
//! does not appear in the build. An outer join's preserved input, an anti join's
//! left input and a mark join's left input keep unmatched rows and never
//! consume. The consumer must not sit upstream of the producer, a source read by
//! other consumers must be proven constrained by the same key set
//! (`keyset_constraints`), and the route must not close a dependency cycle with
//! earlier routes (`cycle`).
//!
//! Discovery is the last step of the pass table because `QueryPlan::rewrite`
//! drops a route whose endpoint node changed; a rule running afterwards would
//! lose routes. Equalities in a join's residual predicate add further candidates
//! for the inputs whose unmatched rows the join discards (`residual_equality`).
//! Rejection reasons and the candidate diagnostics fields are report contracts.

use std::collections::BTreeSet;

use arrow::datatypes::{DataType as ArrowDataType, Field};

use super::{
    access::local_parquet_source,
    lineage::{
        AggregateGroupKeyLineage, SourceColumnLineage, aggregate_group_key_output_name, dedup_lineages,
        lineages_for_column, producer_input_is_direct_aggregate_keyset_alias, unique_lineage,
    },
    rewrite,
};
use crate::{
    backend::AnalysisContext,
    datatypes::{is_supported_list_carry_type, is_timestamp_with_timezone},
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage,
        DynamicFilterExecutionOptions, DynamicFilterId, DynamicFilterKind, DynamicFilterProducer, DynamicFilterRoute,
        dynamic_filter_consumer_source_feeds_downstream_work_fanout,
    },
    expr::{AggregateFunction, BinaryOp, ColumnQualifier, ColumnRef, Expr},
    plan::{
        AggregateNode, JoinNode, JoinNullEquality, JoinSide, JoinType, NodeId, PlanNode, QueryPlan,
        estimated_output_bytes, estimated_output_rows,
    },
    source::{LocalParquetSource, SourceColumnRef, SourceNode},
};

mod cycle;
mod keyset_constraints;
mod residual_equality;
mod route_discovery;

use cycle::{dynamic_filter_route_equivalent, dynamic_filter_route_would_create_node_cycle};
use keyset_constraints::shared_source_filter_is_keyset_constrained;
use residual_equality::{
    large_keyset_dimension_sized_local_parquet_consumer_is_not_promising, producer_keyset_exceeds_runtime_limit,
    residual_equality_route_candidates,
};
pub(crate) use route_discovery::optimize_dynamic_filter_route_discovery;

/// Stable rule name used in optimizer diagnostics.
pub const DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME: &str = "dynamic_filter_route_discovery";

/// Configuration for dynamic-filter route discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicFilterRouteDiscoveryOptions {
    /// Whether the pass may discover native dynamic-filter routes.
    pub enabled: bool,
}

impl DynamicFilterRouteDiscoveryOptions {
    /// Returns options with route discovery disabled.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }
}

impl Default for DynamicFilterRouteDiscoveryOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Diagnostics for dynamic-filter route discovery.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DynamicFilterRouteDiscoveryDiagnostics {
    /// Number of join-key orientations considered.
    pub considered_count: usize,
    /// Number of routes attached to the returned plan.
    pub route_count: usize,
    /// Per-candidate evidence in discovery order.
    pub candidates: Vec<DynamicFilterRouteCandidateDiagnostics>,
}

impl DynamicFilterRouteDiscoveryDiagnostics {
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.route_count == 0 && self.candidates.is_empty()
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.route_count += other.route_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate dynamic-filter route evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicFilterRouteCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub(crate) rule_name: &'static str,
    /// Join node considered as a dynamic-filter producer.
    pub candidate_node_id: NodeId,
    /// Route id when a route was admitted.
    pub(crate) dynamic_filter_id: Option<DynamicFilterId>,
    /// Join build side used as the producer.
    pub producer_build_side: Option<JoinSide>,
    /// Build-side input node.
    pub producer_input_node_id: Option<NodeId>,
    /// Consumer input node before lineage tracing.
    pub(crate) consumer_input_node_id: Option<NodeId>,
    /// Runtime consumer kind selected for the admitted route.
    pub consumer_kind: Option<String>,
    /// Runtime consumer node selected for the admitted route.
    pub consumer_node_id: Option<NodeId>,
    /// Source node that will consume the filter.
    pub consumer_source_node_id: Option<NodeId>,
    /// Build-side join key column.
    pub producer_key_column: Option<String>,
    /// Consumer-side join key column after source lineage tracing.
    pub consumer_key_column: Option<String>,
    /// Payload kinds the route may produce.
    pub filter_kinds: Vec<DynamicFilterKind>,
    /// Where the consumer can apply the payload.
    pub consumer_lowering: Option<DynamicFilterConsumerLowering>,
    /// Stage where this route can first reduce consumer-side work.
    pub consumer_reduction_stage: Option<DynamicFilterConsumerReductionStage>,
    /// Selected source bytes available at route discovery time.
    pub selected_source_bytes: Option<u64>,
    /// Total source bytes available at route discovery time.
    pub total_source_bytes: Option<u64>,
    /// True when the returned plan carries this route.
    pub routed: bool,
    /// Rejection reason when no route was attached.
    pub rejected_reason: Option<DynamicFilterRouteRejectedReason>,
}

/// Reasons a dynamic-filter candidate was not routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicFilterRouteRejectedReason {
    /// Pass was disabled.
    PassDisabled,
    /// The join keeps probe rows that match no build row (an outer join's
    /// preserved input, an anti or mark join's left input), matches null keys,
    /// is null-aware, has other than one key, has no output columns, or reuses
    /// a prebuilt build artifact.
    UnsupportedJoinShape,
    /// Build-side key lineage could not be resolved to one source column.
    ProducerKeyLineageUnavailable,
    /// Consumer-side key lineage could not be resolved to one source column.
    ConsumerKeyLineageUnavailable,
    /// Producer and consumer resolved to the same source node.
    SameSource,
    /// Consumer lineage did not resolve to a local Parquet source.
    ConsumerNotLocalParquet,
    /// Consumer key is not a physical file column visible to the Parquet reader.
    ConsumerColumnNotReaderVisible,
    /// Consumer key Arrow type cannot be used for the initial filter payloads.
    UnsupportedConsumerColumnType,
    /// Consumer source is already required to produce the build-side payload.
    ConsumerUpstreamOfProducer,
    /// Producer lineage crosses a shared path, so a selected residual-side
    /// build frame may be only a subset of the residual input's values.
    ProducerLineageSharedPath,
    /// Consumer source has multiple parents in the plan, so source-level
    /// filtering could affect aliases that are not constrained by the join key.
    ConsumerSourceReusedInPlan,
    /// The route would create a control-dependency cycle with an earlier
    /// dynamic-filter route.
    DynamicFilterControlCycle,
    /// Post-decode join-probe filtering would feed a scalar global decimal
    /// SUM, where the route is optional and can pessimize the aggregate
    /// backend.
    PostDecodeConsumerFeedsScalarDecimalSum,
    /// The producer is proven to exceed the runtime exact-keyset row cap, so
    /// runtime would skip every route for this producer.
    ProducerKeysetExceedsRuntimeLimit,
    /// The consumer is already dimension-sized while the producer is expected
    /// to exceed the large-keyset runtime threshold, so the route is likely to
    /// be skipped at runtime or add more dependency cost than source work saved.
    LargeKeysetDimensionSizedLocalParquetConsumer,
}

/// Whether the producer is a literal batch, possibly projected. Its key set is
/// exact and already materialized, so the size heuristics that reject a large
/// producer for a dimension-sized consumer do not apply to it.
fn producer_input_is_literal_batch_exact_keyset(plan: &QueryPlan, node_id: NodeId) -> bool {
    match plan.node(node_id).map(|node| &node.kind) {
        Some(PlanNode::Source(source)) => matches!(source.as_ref(), SourceNode::LiteralBatch(_)),
        Some(PlanNode::Projection(projection)) => producer_input_is_literal_batch_exact_keyset(plan, projection.input),
        _ => false,
    }
}

/// The join shape whose probe rows a build key payload may remove: the join
/// discards probe rows that match no build row, and its single key never
/// matches null. Under null-equal matching a null probe key matches null build
/// keys, and no payload kind represents null, so only `NullsDoNotMatch` joins
/// qualify.
fn join_shape_supported(join: &JoinNode) -> bool {
    let probe_side = match join.spec.build_side {
        JoinSide::Left => JoinSide::Right,
        JoinSide::Right => JoinSide::Left,
    };
    join_type_discards_unmatched_rows(&join.spec.join_type, probe_side)
        && join.spec.keys.len() == 1
        && join.spec.null_equality == JoinNullEquality::NullsDoNotMatch
        && !join.spec.output_projection.is_empty()
        && join.spec.build_artifact_id.is_none()
}

/// Whether a join of `join_type` drops every row of its `side` input that
/// matches no row of the other input, so that removing such rows before the
/// join leaves its output unchanged. Inner and semi joins drop unmatched rows
/// of both inputs. Left outer, anti and mark joins keep every left row, as
/// output or as its mark, and a right row that matches nothing contributes to
/// none of them. A null-aware anti join reads the right input's nulls and its
/// emptiness, so neither of its inputs qualifies.
fn join_type_discards_unmatched_rows(join_type: &JoinType, side: JoinSide) -> bool {
    match join_type {
        JoinType::Inner | JoinType::LeftSemi => true,
        JoinType::Left | JoinType::LeftAnti { null_aware: false } | JoinType::LeftMark { .. } => {
            side == JoinSide::Right
        },
        JoinType::LeftAnti { null_aware: true } | JoinType::Full | JoinType::Cross => false,
    }
}

fn subtree_contains_node(plan: &QueryPlan, root: NodeId, target: NodeId) -> bool {
    let mut stack = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(node_id) = stack.pop() {
        if !visited.insert(node_id) {
            continue;
        }
        if node_id == target {
            return true;
        }
        let Some(node) = plan.node(node_id) else {
            continue;
        };
        stack.extend(node.input_ids().iter().copied());
    }
    false
}

fn input_reference_count(plan: &QueryPlan, target: NodeId) -> usize {
    let mut count = 0usize;
    for node in plan.nodes() {
        count = count.saturating_add(node.input_ids().iter().filter(|input| **input == target).count());
    }
    count
}

/// One orientation per key: the build side always produces and the probe side
/// always consumes, because the payload exists once the build is complete.
fn route_orientations(join: &JoinNode) -> Vec<RouteOrientation<'_>> {
    join.spec
        .keys
        .iter()
        .map(|key| {
            let (producer_input, consumer_input, producer_key, consumer_key, consumer_side) = match join.spec.build_side
            {
                JoinSide::Left => (join.inputs[0], join.inputs[1], &key.left, &key.right, JoinSide::Right),
                JoinSide::Right => (join.inputs[1], join.inputs[0], &key.right, &key.left, JoinSide::Left),
            };
            RouteOrientation { producer_input, consumer_input, producer_key, consumer_key, consumer_side }
        })
        .collect()
}

struct RouteOrientation<'a> {
    producer_input: NodeId,
    consumer_input: NodeId,
    producer_key: &'a ColumnRef,
    consumer_key: &'a ColumnRef,
    consumer_side: JoinSide,
}

const fn join_side_index(side: JoinSide) -> usize {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

fn rejected_candidate(
    node_id: NodeId,
    join: &JoinNode,
    key: Option<&ColumnRef>,
    reason: DynamicFilterRouteRejectedReason,
) -> DynamicFilterRouteCandidateDiagnostics {
    let consumer_input = match join.spec.build_side {
        JoinSide::Left => join.inputs[1],
        JoinSide::Right => join.inputs[0],
    };
    let producer_input = match join.spec.build_side {
        JoinSide::Left => join.inputs[0],
        JoinSide::Right => join.inputs[1],
    };
    DynamicFilterRouteCandidateDiagnostics {
        rule_name: DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
        candidate_node_id: node_id,
        dynamic_filter_id: None,
        producer_build_side: Some(join.spec.build_side),
        producer_input_node_id: Some(producer_input),
        consumer_input_node_id: Some(consumer_input),
        consumer_kind: None,
        consumer_node_id: None,
        consumer_source_node_id: None,
        producer_key_column: key.map(|key| key.name().to_string()),
        consumer_key_column: None,
        filter_kinds: Vec::new(),
        consumer_lowering: None,
        consumer_reduction_stage: None,
        selected_source_bytes: None,
        total_source_bytes: None,
        routed: false,
        rejected_reason: Some(reason),
    }
}

fn consumer_reader_column_is_visible(source: &LocalParquetSource, column: &str) -> bool {
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column) else {
        return false;
    };
    source.read_column_file_indices.contains(&file_index)
}

fn consumer_column_type_is_supported(source: &LocalParquetSource, column: &str, ctx: AnalysisContext<'_>) -> bool {
    source.schema.field_with_name(column).ok().is_some_and(|field| dynamic_filter_key_type_is_supported(field, ctx))
}

/// The first `Filter` below `node_id`, through renaming projections and sorts,
/// that still carries `column` with a supported type. A `Limit` ends the walk
/// because filtering below it would change which rows it keeps.
fn earliest_filter_consumer_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column: &str,
    stop_at_shared_path: bool,
    ctx: AnalysisContext<'_>,
) -> Option<(DynamicFilterConsumer, String)> {
    if stop_at_shared_path && parent_counts[node_id.index()] > 1 {
        return None;
    }
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Filter(_) => node
            .output_schema()
            .field_with_name(column)
            .ok()
            .filter(|field| dynamic_filter_key_type_is_supported(field, ctx))
            .map(|_| (DynamicFilterConsumer::Filter { filter_node_id: node_id }, column.to_string())),
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .find(|output| output.field.name() == column)
            .and_then(|output| direct_projection_column_name(&output.expr))
            .and_then(|input_column| {
                earliest_filter_consumer_for_column(
                    plan,
                    parent_counts,
                    projection.input,
                    input_column,
                    stop_at_shared_path,
                    ctx,
                )
            }),
        PlanNode::Sort(sort) => {
            earliest_filter_consumer_for_column(plan, parent_counts, sort.input, column, stop_at_shared_path, ctx)
        },
        PlanNode::Limit(_) => None,
        PlanNode::Source(_)
        | PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Union(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

/// A consumer that applies the payload after decode: the earliest filter on the
/// probe side, or, when allowed, the join's own probe. `None` when the probe
/// key type cannot carry a payload.
fn post_decode_consumer_for_orientation(
    plan: &QueryPlan,
    parent_counts: &[usize],
    join_node_id: NodeId,
    orientation: &RouteOrientation<'_>,
    stop_at_shared_path: bool,
    allow_hash_join_probe_fallback: bool,
    ctx: AnalysisContext<'_>,
) -> Option<(DynamicFilterConsumer, String)> {
    let consumer_key = orientation.consumer_key.name();
    if !plan
        .node(orientation.consumer_input)
        .and_then(|node| node.output_schema().field_with_name(consumer_key).ok())
        .is_some_and(|field| dynamic_filter_key_type_is_supported(field, ctx))
    {
        return None;
    }
    let consumer = earliest_filter_consumer_for_column(
        plan,
        parent_counts,
        orientation.consumer_input,
        consumer_key,
        stop_at_shared_path,
        ctx,
    );
    consumer.or_else(|| {
        allow_hash_join_probe_fallback.then(|| {
            (
                DynamicFilterConsumer::HashJoinProbe { join_node_id, probe_side: orientation.consumer_side },
                consumer_key.to_string(),
            )
        })
    })
}

fn dynamic_filter_key_type_is_supported(field: &Field, ctx: AnalysisContext<'_>) -> bool {
    ctx.backend().supports_type(field.data_type())
        && !is_timestamp_with_timezone(field.data_type())
        && !is_supported_list_carry_type(field.data_type())
}

fn populate_selected_consumer_diagnostics(
    candidate: &mut DynamicFilterRouteCandidateDiagnostics,
    consumer: &DynamicFilterConsumer,
    consumer_column: &str,
    consumer_lowering: DynamicFilterConsumerLowering,
) {
    candidate.consumer_key_column = Some(consumer_column.to_string());
    candidate.consumer_kind = Some(consumer.kind().to_string());
    candidate.consumer_node_id = Some(consumer.node_id());
    candidate.consumer_lowering = Some(consumer_lowering);
    candidate.consumer_reduction_stage = Some(consumer_reduction_stage(consumer, consumer_lowering));
}

fn consumer_reduction_stage(
    consumer: &DynamicFilterConsumer,
    consumer_lowering: DynamicFilterConsumerLowering,
) -> DynamicFilterConsumerReductionStage {
    match (consumer, consumer_lowering) {
        (
            DynamicFilterConsumer::LocalParquetSource { .. },
            DynamicFilterConsumerLowering::ReaderPredicate | DynamicFilterConsumerLowering::MergedReaderPredicate,
        ) => DynamicFilterConsumerReductionStage::SourceReader,
        (DynamicFilterConsumer::LocalParquetSource { .. }, _) => DynamicFilterConsumerReductionStage::SourcePostDecode,
        (DynamicFilterConsumer::Filter { .. }, _) => DynamicFilterConsumerReductionStage::FilterPostDecode,
        (DynamicFilterConsumer::HashJoinProbe { .. }, _) => DynamicFilterConsumerReductionStage::JoinProbePostDecode,
    }
}

/// Whether a join-probe consumer feeds, through unary operators only, a global
/// aggregate with a `Decimal128` `SUM`. Such a route is declined; see
/// [`DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum`].
fn post_decode_consumer_feeds_scalar_decimal_sum(
    plan: &QueryPlan,
    parent_counts: &[usize],
    consumer: &DynamicFilterConsumer,
) -> bool {
    let DynamicFilterConsumer::HashJoinProbe { join_node_id, .. } = consumer else {
        return false;
    };
    let mut current = *join_node_id;
    loop {
        let Some(parent) = single_parent(plan, parent_counts, current) else {
            return false;
        };
        let Some(parent_node) = plan.node(parent) else {
            return false;
        };
        match &parent_node.kind {
            PlanNode::Projection(projection) if projection.input == current => current = parent,
            PlanNode::Filter(filter) if filter.input == current => current = parent,
            PlanNode::Sort(sort) if sort.input == current => current = parent,
            PlanNode::Aggregate(aggregate) if aggregate.input == current => {
                return aggregate.spec.group_exprs.is_empty()
                    && aggregate.spec.aggr_exprs.iter().any(|expr| {
                        expr.function == AggregateFunction::Sum
                            && parent_node
                                .output_schema()
                                .field_with_name(expr.alias.as_str())
                                .ok()
                                .is_some_and(|field| matches!(field.data_type(), ArrowDataType::Decimal128(_, _)))
                    });
            },
            _ => return false,
        }
    }
}

fn single_parent(plan: &QueryPlan, parent_counts: &[usize], node_id: NodeId) -> Option<NodeId> {
    if parent_counts[node_id.index()] != 1 {
        return None;
    }
    plan.nodes().iter().find(|candidate| candidate.input_ids().contains(&node_id)).map(|candidate| candidate.id)
}

fn direct_projection_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        Expr::Alias { expr, .. } => direct_projection_column_name(expr),
        _ => None,
    }
}

/// Where a local Parquet consumer can apply the payload. A source without a
/// reader predicate takes the payload as its predicate; one with a supported
/// predicate merges it; one whose static predicate the reader cannot lower
/// keeps the payload out of the reader.
fn consumer_lowering(source: &LocalParquetSource, ctx: AnalysisContext<'_>) -> DynamicFilterConsumerLowering {
    match &source.reader_predicate {
        Some(predicate) if source.reader_predicate_is_supported(predicate, ctx) => {
            DynamicFilterConsumerLowering::MergedReaderPredicate
        },
        Some(_) => DynamicFilterConsumerLowering::SkippedUnsupportedStaticPredicateMix,
        None => DynamicFilterConsumerLowering::ReaderPredicate,
    }
}
