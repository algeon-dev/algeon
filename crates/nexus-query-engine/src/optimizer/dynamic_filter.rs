use arrow::datatypes::DataType as ArrowDataType;

use super::rewrite;
use crate::{
    datatypes::is_supported_arrow_type,
    dynamic_filter::{
        DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterConsumerReductionStage, DynamicFilterId,
        DynamicFilterKind, DynamicFilterProducer, DynamicFilterRoute,
    },
    expr::{AggregateFunction, ColumnRef, Expr},
    plan::{JoinNode, JoinSide, JoinType, NodeId, PlanNode, QueryPlan},
    source::{LocalParquetSource, SourceColumnRef, SourceNode},
};

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
    /// Returns true when the pass produced no route evidence.
    pub fn is_empty(&self) -> bool {
        self.considered_count == 0 && self.route_count == 0 && self.candidates.is_empty()
    }

    /// Merges another diagnostics value into this one.
    pub fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.route_count += other.route_count;
        self.candidates.extend(other.candidates);
    }
}

/// Per-candidate dynamic-filter route evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicFilterRouteCandidateDiagnostics {
    /// Stable rule name for this diagnostic row.
    pub rule_name: &'static str,
    /// Join node considered as a dynamic-filter producer.
    pub candidate_node_id: NodeId,
    /// Route id when a route was admitted.
    pub dynamic_filter_id: Option<DynamicFilterId>,
    /// Join build side used as the producer.
    pub producer_build_side: Option<JoinSide>,
    /// Build-side input node.
    pub producer_input_node_id: Option<NodeId>,
    /// Consumer input node before lineage tracing.
    pub consumer_input_node_id: Option<NodeId>,
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
    /// Join shape cannot produce a route yet.
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
    /// Consumer source has multiple parents in the plan, so source-level
    /// filtering could affect aliases that are not constrained by the join key.
    ConsumerSourceReusedInPlan,
    /// Post-decode join-probe filtering would feed a scalar global decimal
    /// SUM, where the route is optional and can pessimize the aggregate
    /// backend.
    PostDecodeConsumerFeedsScalarDecimalSum,
}

pub(crate) struct DynamicFilterRouteDiscoveryResult {
    pub plan: QueryPlan,
    pub change: super::NativeOptimizerChange,
    pub diagnostics: DynamicFilterRouteDiscoveryDiagnostics,
}

pub(crate) fn optimize_dynamic_filter_route_discovery(
    plan: QueryPlan,
    options: &DynamicFilterRouteDiscoveryOptions,
) -> DynamicFilterRouteDiscoveryResult {
    if !options.enabled {
        return DynamicFilterRouteDiscoveryResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: DynamicFilterRouteDiscoveryDiagnostics::default(),
        };
    }

    let mut diagnostics = DynamicFilterRouteDiscoveryDiagnostics::default();
    let mut routes = Vec::new();
    let parent_counts = rewrite::parent_counts(&plan);

    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };

        if !join_shape_supported(join) {
            diagnostics.considered_count += 1;
            diagnostics.candidates.push(rejected_candidate(
                node.id,
                join,
                None,
                DynamicFilterRouteRejectedReason::UnsupportedJoinShape,
            ));
            continue;
        }

        for orientation in route_orientations(join) {
            diagnostics.considered_count += 1;
            let mut candidate = DynamicFilterRouteCandidateDiagnostics {
                rule_name: DYNAMIC_FILTER_ROUTE_DISCOVERY_RULE_NAME,
                candidate_node_id: node.id,
                dynamic_filter_id: None,
                producer_build_side: Some(join.spec.build_side),
                producer_input_node_id: Some(orientation.producer_input),
                consumer_input_node_id: Some(orientation.consumer_input),
                consumer_kind: None,
                consumer_node_id: None,
                consumer_source_node_id: None,
                producer_key_column: Some(orientation.producer_key.name().to_string()),
                consumer_key_column: None,
                filter_kinds: Vec::new(),
                consumer_lowering: None,
                consumer_reduction_stage: None,
                selected_source_bytes: None,
                total_source_bytes: None,
                routed: false,
                rejected_reason: None,
            };

            let Some(producer_lineage) = unique_lineage(
                &plan,
                &parent_counts,
                orientation.producer_input,
                orientation.producer_key.name(),
                AggregateGroupKeyLineage::Producer,
            ) else {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ProducerKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };
            let aggregate_group_key_producer =
                producer_lineage.aggregate_group_key_node_id.filter(|aggregate_node_id| {
                    producer_input_is_direct_aggregate_keyset_alias(
                        &plan,
                        orientation.producer_input,
                        orientation.producer_key.name(),
                        *aggregate_node_id,
                    )
                });
            let producer_dependency_root = aggregate_group_key_producer.unwrap_or(orientation.producer_input);
            let consumer_lineages = dedup_lineages(lineages_for_column(
                &plan,
                &parent_counts,
                orientation.consumer_input,
                orientation.consumer_key.name(),
                AggregateGroupKeyLineage::Consumer,
            ));
            if consumer_lineages.is_empty() {
                candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerKeyLineageUnavailable);
                diagnostics.candidates.push(candidate);
                continue;
            };

            let multiple_consumer_lineages = consumer_lineages.len() > 1;
            for consumer_lineage in consumer_lineages {
                let mut candidate = candidate.clone();
                candidate.consumer_source_node_id = Some(consumer_lineage.source_node_id);
                candidate.consumer_key_column = Some(consumer_lineage.source_column.clone());

                if producer_lineage.source_node_id == consumer_lineage.source_node_id {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::SameSource);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                if subtree_contains_node(&plan, producer_dependency_root, consumer_lineage.source_node_id) {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerUpstreamOfProducer);
                    diagnostics.candidates.push(candidate);
                    continue;
                }
                let consumer_source = local_parquet_source(&plan, consumer_lineage.source_node_id);
                let (consumer, consumer_column, consumer_lowering) = if let Some(consumer_source) = consumer_source {
                    let stats = consumer_source.diagnostics.selected_row_group_stats.as_ref();
                    candidate.selected_source_bytes = stats.and_then(|stats| stats.selected_source_bytes());
                    candidate.total_source_bytes = stats.and_then(|stats| stats.total_source_bytes());

                    if input_reference_count(&plan, consumer_lineage.source_node_id) > 1
                        || consumer_lineage.has_shared_path
                    {
                        if multiple_consumer_lineages {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }
                        let Some((consumer, consumer_column)) = post_decode_consumer_for_orientation(
                            &plan,
                            &parent_counts,
                            node.id,
                            &orientation,
                            true,
                            false,
                        ) else {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::ConsumerSourceReusedInPlan);
                            diagnostics.candidates.push(candidate);
                            continue;
                        };
                        if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                            populate_selected_consumer_diagnostics(
                                &mut candidate,
                                &consumer,
                                &consumer_column,
                                DynamicFilterConsumerLowering::PostDecodeOnly,
                            );
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }
                        candidate.consumer_key_column = Some(consumer_column.clone());
                        (consumer, consumer_column, DynamicFilterConsumerLowering::PostDecodeOnly)
                    } else {
                        if !consumer_reader_column_is_visible(consumer_source, &consumer_lineage.source_column) {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::ConsumerColumnNotReaderVisible);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }
                        if !consumer_column_type_is_supported(consumer_source, &consumer_lineage.source_column) {
                            candidate.rejected_reason =
                                Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                            diagnostics.candidates.push(candidate);
                            continue;
                        }

                        (
                            DynamicFilterConsumer::LocalParquetSource {
                                source_node_id: consumer_lineage.source_node_id,
                            },
                            consumer_lineage.source_column.clone(),
                            consumer_lowering(consumer_source),
                        )
                    }
                } else if multiple_consumer_lineages {
                    candidate.rejected_reason = Some(DynamicFilterRouteRejectedReason::ConsumerNotLocalParquet);
                    diagnostics.candidates.push(candidate);
                    continue;
                } else {
                    let Some((consumer, consumer_column)) =
                        post_decode_consumer_for_orientation(&plan, &parent_counts, node.id, &orientation, false, true)
                    else {
                        candidate.rejected_reason =
                            Some(DynamicFilterRouteRejectedReason::UnsupportedConsumerColumnType);
                        diagnostics.candidates.push(candidate);
                        continue;
                    };
                    if post_decode_consumer_feeds_scalar_decimal_sum(&plan, &parent_counts, &consumer) {
                        populate_selected_consumer_diagnostics(
                            &mut candidate,
                            &consumer,
                            &consumer_column,
                            DynamicFilterConsumerLowering::PostDecodeOnly,
                        );
                        candidate.rejected_reason =
                            Some(DynamicFilterRouteRejectedReason::PostDecodeConsumerFeedsScalarDecimalSum);
                        diagnostics.candidates.push(candidate);
                        continue;
                    }
                    candidate.consumer_key_column = Some(consumer_column.clone());
                    (consumer, consumer_column, DynamicFilterConsumerLowering::PostDecodeOnly)
                };
                let filter_kinds =
                    vec![DynamicFilterKind::Range, DynamicFilterKind::SmallInList, DynamicFilterKind::HashSetContains];
                let id = DynamicFilterId::new(routes.len());
                let producer_column = if aggregate_group_key_producer.is_some() {
                    producer_lineage.source_column.clone()
                } else {
                    orientation.producer_key.name().to_string()
                };
                let producer = aggregate_group_key_producer
                    .map(|aggregate_node_id| DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id })
                    .unwrap_or(DynamicFilterProducer::HashJoinBuild {
                        join_node_id: node.id,
                        build_side: join.spec.build_side,
                        build_input: orientation.producer_input,
                    });
                let route = DynamicFilterRoute::new(
                    id,
                    producer,
                    consumer,
                    vec![producer_column],
                    vec![consumer_column],
                    filter_kinds.clone(),
                    consumer_lowering,
                );
                candidate.consumer_kind = Some(route.consumer.kind().to_string());
                candidate.consumer_node_id = Some(route.consumer.node_id());
                candidate.consumer_reduction_stage = Some(route.consumer_reduction_stage());
                routes.push(route);

                candidate.dynamic_filter_id = Some(id);
                candidate.filter_kinds = filter_kinds;
                candidate.consumer_lowering = Some(consumer_lowering);
                candidate.routed = true;
                diagnostics.route_count += 1;
                diagnostics.candidates.push(candidate);
            }
        }
    }

    let change =
        if routes.is_empty() { super::NativeOptimizerChange::Unchanged } else { super::NativeOptimizerChange::Changed };
    let plan = plan.with_dynamic_filter_routes(routes);
    DynamicFilterRouteDiscoveryResult { plan, change, diagnostics }
}

fn join_shape_supported(join: &JoinNode) -> bool {
    join.spec.join_type == JoinType::Inner
        && join.spec.keys.len() == 1
        && join.spec.null_equality == crate::plan::JoinNullEquality::NullsDoNotMatch
        && !join.spec.output_projection.is_empty()
        && join.spec.build_artifact_id.is_none()
}

fn subtree_contains_node(plan: &QueryPlan, root: NodeId, target: NodeId) -> bool {
    let mut stack = vec![root];
    let mut visited = std::collections::BTreeSet::new();
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceColumnLineage {
    source_node_id: NodeId,
    source_column: String,
    aggregate_group_key_node_id: Option<NodeId>,
    has_shared_path: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AggregateGroupKeyLineage {
    Producer,
    Consumer,
    Reject,
}

fn unique_lineage(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Option<SourceColumnLineage> {
    let mut lineages =
        dedup_lineages(lineages_for_column(plan, parent_counts, node_id, column_name, aggregate_group_key_lineage));
    (lineages.len() == 1).then(|| lineages.remove(0))
}

fn dedup_lineages(mut lineages: Vec<SourceColumnLineage>) -> Vec<SourceColumnLineage> {
    lineages.sort_by(|left, right| {
        left.source_node_id
            .cmp(&right.source_node_id)
            .then_with(|| left.source_column.cmp(&right.source_column))
            .then_with(|| left.aggregate_group_key_node_id.cmp(&right.aggregate_group_key_node_id))
            .then_with(|| left.has_shared_path.cmp(&right.has_shared_path))
    });
    lineages.dedup();
    lineages
}

fn lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            SourceNode::GpuResident(source) if source.schema.field_with_name(column_name).is_ok() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            SourceNode::LiteralBatch(source) if source.schema.field_with_name(column_name).is_ok() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => {
            lineages_for_column(plan, parent_counts, filter.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| {
                projection_expr_lineages(
                    plan,
                    parent_counts,
                    projection.input,
                    &output.expr,
                    aggregate_group_key_lineage,
                )
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Sort(sort) => {
            lineages_for_column(plan, parent_counts, sort.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Limit(limit) => {
            lineages_for_column(plan, parent_counts, limit.input, column_name, aggregate_group_key_lineage)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                lineages_for_column(plan, parent_counts, input, output.column.name(), aggregate_group_key_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(aggregate) if aggregate_group_key_lineage != AggregateGroupKeyLineage::Reject => {
            aggregate_group_key_lineages(
                plan,
                parent_counts,
                aggregate,
                node_id,
                column_name,
                aggregate_group_key_lineage,
            )
            .into_iter()
            .map(mark_shared)
            .collect()
        },
        PlanNode::Aggregate(_) => Vec::new(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| {
                lineages_for_column(plan, parent_counts, *input, column_name, aggregate_group_key_lineage)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Window(_) | PlanNode::EdgeNormalize(_) | PlanNode::GraphAlgorithm(_) => Vec::new(),
    }
}

fn aggregate_group_key_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    aggregate: &crate::plan::AggregateNode,
    aggregate_node_id: NodeId,
    column_name: &str,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    if aggregate.spec.grouping_sets.is_some() {
        return Vec::new();
    }
    aggregate
        .spec
        .group_exprs
        .iter()
        .filter_map(|expr| aggregate_group_key_output_name(expr).map(|output_name| (expr, output_name)))
        .filter(|(_, output_name)| *output_name == column_name)
        .flat_map(|(expr, output_name)| match aggregate_group_key_lineage {
            AggregateGroupKeyLineage::Producer => {
                projection_expr_lineages(plan, parent_counts, aggregate.input, expr, aggregate_group_key_lineage)
                    .into_iter()
                    .map(move |mut lineage| {
                        lineage.source_column = output_name.to_string();
                        lineage
                    })
                    .collect::<Vec<_>>()
            },
            AggregateGroupKeyLineage::Consumer => {
                consumer_pre_aggregate_expr_lineages(plan, parent_counts, aggregate.input, expr)
            },
            AggregateGroupKeyLineage::Reject => Vec::new(),
        })
        .map(|mut lineage| {
            lineage.aggregate_group_key_node_id = Some(aggregate_node_id);
            lineage
        })
        .collect()
}

fn aggregate_group_key_output_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.display_name()),
        Expr::Alias { alias, .. } => Some(alias.as_str()),
        _ => None,
    }
}

fn producer_input_is_direct_aggregate_keyset_alias(
    plan: &QueryPlan,
    node_id: NodeId,
    column_name: &str,
    aggregate_node_id: NodeId,
) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if node_id == aggregate_node_id {
        return matches!(node.kind, PlanNode::Aggregate(_))
            && node.output_schema().field_with_name(column_name).is_ok();
    }
    match &node.kind {
        PlanNode::Projection(projection) => {
            projection.outputs.iter().filter(|output| output.field.name() == column_name).any(|output| {
                producer_expr_is_direct_aggregate_keyset_alias(plan, projection.input, &output.expr, aggregate_node_id)
            })
        },
        PlanNode::Sort(sort) => {
            producer_input_is_direct_aggregate_keyset_alias(plan, sort.input, column_name, aggregate_node_id)
        },
        PlanNode::Source(_)
        | PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Limit(_)
        | PlanNode::Join(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => false,
    }
}

fn producer_expr_is_direct_aggregate_keyset_alias(
    plan: &QueryPlan,
    input: NodeId,
    expr: &Expr,
    aggregate_node_id: NodeId,
) -> bool {
    match expr {
        Expr::Column(column) => {
            producer_input_is_direct_aggregate_keyset_alias(plan, input, column.name(), aggregate_node_id)
        },
        Expr::Alias { expr, .. } => {
            producer_expr_is_direct_aggregate_keyset_alias(plan, input, expr, aggregate_node_id)
        },
        _ => false,
    }
}

fn projection_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
    aggregate_group_key_lineage: AggregateGroupKeyLineage,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => {
            lineages_for_column(plan, parent_counts, input, column.name(), aggregate_group_key_lineage)
        },
        Expr::Alias { expr, .. } => {
            projection_expr_lineages(plan, parent_counts, input, expr, aggregate_group_key_lineage)
        },
        _ => Vec::new(),
    }
}

fn consumer_pre_aggregate_expr_lineages(
    plan: &QueryPlan,
    parent_counts: &[usize],
    input: NodeId,
    expr: &Expr,
) -> Vec<SourceColumnLineage> {
    match expr {
        Expr::Column(column) => consumer_pre_aggregate_lineages_for_column(plan, parent_counts, input, column.name()),
        Expr::Alias { expr, .. } => consumer_pre_aggregate_expr_lineages(plan, parent_counts, input, expr),
        _ => Vec::new(),
    }
}

fn consumer_pre_aggregate_lineages_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column_name: &str,
) -> Vec<SourceColumnLineage> {
    let Some(node) = plan.node(node_id) else {
        return Vec::new();
    };
    let has_shared_path = parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1;
    let mark_shared = |mut lineage: SourceColumnLineage| {
        lineage.has_shared_path |= has_shared_path;
        lineage
    };
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.source_column_ref(column_name).is_some() => {
                vec![SourceColumnLineage {
                    source_node_id: node_id,
                    source_column: column_name.to_string(),
                    aggregate_group_key_node_id: None,
                    has_shared_path,
                }]
            },
            _ => Vec::new(),
        },
        PlanNode::Filter(filter) => {
            consumer_pre_aggregate_lineages_for_column(plan, parent_counts, filter.input, column_name)
                .into_iter()
                .map(mark_shared)
                .collect()
        },
        PlanNode::Projection(projection) => projection
            .outputs
            .iter()
            .filter(|output| output.field.name() == column_name)
            .flat_map(|output| {
                consumer_pre_aggregate_expr_lineages(plan, parent_counts, projection.input, &output.expr)
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Join(join) => join
            .spec
            .output_projection
            .iter()
            .filter(|output| output.output_name == column_name)
            .flat_map(|output| {
                let input = match output.side {
                    JoinSide::Left => join.inputs[0],
                    JoinSide::Right => join.inputs[1],
                };
                consumer_pre_aggregate_lineages_for_column(plan, parent_counts, input, output.column.name())
            })
            .map(mark_shared)
            .collect(),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .flat_map(|input| consumer_pre_aggregate_lineages_for_column(plan, parent_counts, *input, column_name))
            .map(mark_shared)
            .collect(),
        PlanNode::Aggregate(_)
        | PlanNode::Sort(_)
        | PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => Vec::new(),
    }
}

fn local_parquet_source(plan: &QueryPlan, node_id: NodeId) -> Option<&LocalParquetSource> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else {
        return None;
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        return None;
    };
    Some(source)
}

fn consumer_reader_column_is_visible(source: &LocalParquetSource, column: &str) -> bool {
    let Some(SourceColumnRef::File(file_index)) = source.source_column_ref(column) else {
        return false;
    };
    source.read_column_file_indices.contains(&file_index)
}

fn consumer_column_type_is_supported(source: &LocalParquetSource, column: &str) -> bool {
    source.schema.field_with_name(column).ok().is_some_and(|field| is_supported_arrow_type(field.data_type()))
}

fn earliest_filter_consumer_for_column(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    column: &str,
    stop_at_shared_path: bool,
) -> Option<(DynamicFilterConsumer, String)> {
    if stop_at_shared_path && parent_counts.get(node_id.index()).copied().unwrap_or_default() > 1 {
        return None;
    }
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Filter(_) => node
            .output_schema()
            .field_with_name(column)
            .ok()
            .filter(|field| is_supported_arrow_type(field.data_type()))
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
                )
            }),
        PlanNode::Sort(sort) => {
            earliest_filter_consumer_for_column(plan, parent_counts, sort.input, column, stop_at_shared_path)
        },
        PlanNode::Limit(_) => None,
        PlanNode::Source(_)
        | PlanNode::Join(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Union(_)
        | PlanNode::Window(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn post_decode_consumer_for_orientation(
    plan: &QueryPlan,
    parent_counts: &[usize],
    join_node_id: NodeId,
    orientation: &RouteOrientation<'_>,
    stop_at_shared_path: bool,
    allow_hash_join_probe_fallback: bool,
) -> Option<(DynamicFilterConsumer, String)> {
    let consumer_key = orientation.consumer_key.name();
    if !plan
        .node(orientation.consumer_input)
        .and_then(|node| node.output_schema().field_with_name(consumer_key).ok())
        .is_some_and(|field| is_supported_arrow_type(field.data_type()))
    {
        return None;
    }
    let consumer = earliest_filter_consumer_for_column(
        plan,
        parent_counts,
        orientation.consumer_input,
        consumer_key,
        stop_at_shared_path,
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
    if parent_counts.get(node_id.index()).copied().unwrap_or_default() != 1 {
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

fn consumer_lowering(source: &LocalParquetSource) -> DynamicFilterConsumerLowering {
    match &source.reader_predicate {
        Some(predicate) if source.reader_predicate_is_supported(predicate) => {
            DynamicFilterConsumerLowering::MergedReaderPredicate
        },
        Some(_) => DynamicFilterConsumerLowering::SkippedUnsupportedStaticPredicateMix,
        None => DynamicFilterConsumerLowering::ReaderPredicate,
    }
}
