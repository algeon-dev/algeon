use super::{
    AggregateGroupKeyLineage, BinaryOp, ColumnQualifier, DynamicFilterExecutionOptions, DynamicFilterProducer, Expr,
    JoinNode, JoinSide, JoinType, LocalParquetSource, NodeId, PlanNode, QueryPlan, SourceColumnLineage, SourceNode,
    dedup_lineages, estimated_output_bytes, estimated_output_rows, join_shape_supported, join_side_index,
    lineages_for_column, local_parquet_source, subtree_contains_node, unique_lineage,
};

#[derive(Debug, Clone)]
pub(super) struct ResidualEqualityRouteCandidate {
    pub(super) producer: DynamicFilterProducer,
    pub(super) producer_build_side: JoinSide,
    pub(super) producer_input: NodeId,
    pub(super) producer_column: String,
    pub(super) producer_lineage_has_shared_path: bool,
    pub(super) consumer_input: NodeId,
    pub(super) consumer_lineage: SourceColumnLineage,
}

pub(super) fn residual_equality_route_candidates(
    plan: &QueryPlan,
    parent_counts: &[usize],
    join_node_id: NodeId,
    join: &JoinNode,
) -> Vec<ResidualEqualityRouteCandidate> {
    if join.spec.join_type != JoinType::Inner {
        return Vec::new();
    }
    let Some(residual) = &join.spec.residual_predicate else {
        return Vec::new();
    };
    let mut equalities = Vec::new();
    collect_residual_input_equalities(residual, &mut equalities);
    let mut candidates = Vec::new();
    for equality in equalities {
        candidates.extend(residual_equality_route_candidates_for_orientation(
            plan,
            parent_counts,
            join_node_id,
            join,
            equality.left_column.as_str(),
            equality.right_column.as_str(),
            JoinSide::Left,
            JoinSide::Right,
        ));
        candidates.extend(residual_equality_route_candidates_for_orientation(
            plan,
            parent_counts,
            join_node_id,
            join,
            equality.right_column.as_str(),
            equality.left_column.as_str(),
            JoinSide::Right,
            JoinSide::Left,
        ));
    }
    candidates
}

#[expect(clippy::too_many_arguments, reason = "planning inputs are distinct proof and constraint values")]
pub(super) fn residual_equality_route_candidates_for_orientation(
    plan: &QueryPlan,
    parent_counts: &[usize],
    residual_join_node_id: NodeId,
    residual_join: &JoinNode,
    producer_column_at_residual_input: &str,
    consumer_column_at_residual_input: &str,
    producer_side: JoinSide,
    consumer_side: JoinSide,
) -> Vec<ResidualEqualityRouteCandidate> {
    let producer_input = residual_join.inputs[join_side_index(producer_side)];
    let consumer_input = residual_join.inputs[join_side_index(consumer_side)];
    let Some(producer_lineage) = unique_lineage(
        plan,
        parent_counts,
        producer_input,
        producer_column_at_residual_input,
        AggregateGroupKeyLineage::Reject,
    ) else {
        return Vec::new();
    };
    let consumer_lineages = dedup_lineages(lineages_for_column(
        plan,
        parent_counts,
        consumer_input,
        consumer_column_at_residual_input,
        AggregateGroupKeyLineage::Consumer,
    ));
    if consumer_lineages.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for consumer_lineage in consumer_lineages {
        let Some(consumer_source) = local_parquet_source(plan, consumer_lineage.source_node_id) else {
            continue;
        };
        if !residual_equality_source_filter_is_promising(
            local_parquet_source(plan, producer_lineage.source_node_id),
            consumer_source,
        ) {
            continue;
        }
        for producer in residual_equality_hash_build_producers_for_lineage(
            plan,
            parent_counts,
            residual_join_node_id,
            producer_input,
            &producer_lineage,
        ) {
            candidates.push(ResidualEqualityRouteCandidate {
                producer: DynamicFilterProducer::HashJoinBuild {
                    join_node_id: producer.join_node_id,
                    build_side: producer.build_side,
                    build_input: producer.build_input,
                },
                producer_build_side: producer.build_side,
                producer_input: producer.build_input,
                producer_column: producer.producer_column,
                producer_lineage_has_shared_path: producer_lineage.has_shared_path,
                consumer_input,
                consumer_lineage: consumer_lineage.clone(),
            });
        }
    }
    candidates
}

#[derive(Debug, Clone)]
struct ResidualEqualityHashBuildProducer {
    join_node_id: NodeId,
    build_side: JoinSide,
    build_input: NodeId,
    pub(super) producer_column: String,
}

fn residual_equality_hash_build_producers_for_lineage(
    plan: &QueryPlan,
    parent_counts: &[usize],
    residual_join_node_id: NodeId,
    producer_side_root: NodeId,
    producer_lineage: &SourceColumnLineage,
) -> Vec<ResidualEqualityHashBuildProducer> {
    let mut producers = Vec::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        if !subtree_contains_node(plan, producer_side_root, node.id) {
            continue;
        }
        if !join_shape_supported(join) {
            continue;
        }
        let build_input = match join.spec.build_side {
            JoinSide::Left => join.inputs[0],
            JoinSide::Right => join.inputs[1],
        };
        if build_input != producer_lineage.source_node_id
            && !simple_source_path_contains_node(plan, build_input, producer_lineage.source_node_id)
        {
            continue;
        }
        let Some(producer_column) = source_lineage_column_at_node(
            plan,
            parent_counts,
            build_input,
            producer_lineage.source_node_id,
            producer_lineage.source_column.as_str(),
        ) else {
            continue;
        };
        if node.id == residual_join_node_id && subtree_contains_node(plan, build_input, producer_lineage.source_node_id)
        {
            continue;
        }
        producers.push(ResidualEqualityHashBuildProducer {
            join_node_id: node.id,
            build_side: join.spec.build_side,
            build_input,
            producer_column,
        });
    }
    producers.sort_by_key(|producer| producer.join_node_id);
    producers
}

fn source_lineage_column_at_node(
    plan: &QueryPlan,
    parent_counts: &[usize],
    node_id: NodeId,
    source_node_id: NodeId,
    source_column: &str,
) -> Option<String> {
    let node = plan.node(node_id)?;
    node.output_schema()
        .fields()
        .iter()
        .filter_map(|field| {
            let lineages = dedup_lineages(lineages_for_column(
                plan,
                parent_counts,
                node_id,
                field.name(),
                AggregateGroupKeyLineage::Producer,
            ));
            lineages
                .iter()
                .any(|lineage| lineage.source_node_id == source_node_id && lineage.source_column == source_column)
                .then(|| field.name().to_string())
        })
        .next()
}

fn simple_source_path_contains_node(plan: &QueryPlan, root: NodeId, target: NodeId) -> bool {
    let Some(node) = plan.node(root) else {
        return false;
    };
    if root == target {
        return true;
    }
    match &node.kind {
        PlanNode::Filter(filter) => simple_source_path_contains_node(plan, filter.input, target),
        PlanNode::Projection(projection) => simple_source_path_contains_node(plan, projection.input, target),
        PlanNode::Sort(sort) => simple_source_path_contains_node(plan, sort.input, target),
        PlanNode::Source(_)
        | PlanNode::Limit(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Join(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => false,
    }
}

fn residual_equality_source_filter_is_promising(
    producer_source: Option<&LocalParquetSource>,
    consumer_source: &LocalParquetSource,
) -> bool {
    let Some(producer_source) = producer_source else {
        return false;
    };
    if producer_source.reader_predicate.is_some() {
        return true;
    }
    let producer_stats = producer_source.diagnostics.selected_row_group_stats.as_ref();
    let consumer_stats = consumer_source.diagnostics.selected_row_group_stats.as_ref();
    if let (Some(producer_rows), Some(consumer_rows)) =
        (producer_stats.and_then(|stats| stats.selected_rows), consumer_stats.and_then(|stats| stats.selected_rows))
    {
        return producer_rows < consumer_rows;
    }
    if let (Some(producer_bytes), Some(consumer_bytes)) = (
        producer_stats.and_then(|stats| stats.selected_source_bytes()),
        consumer_stats.and_then(|stats| stats.selected_source_bytes()),
    ) {
        return producer_bytes < consumer_bytes;
    }
    false
}

pub(super) fn large_keyset_dimension_sized_local_parquet_consumer_is_not_promising(
    plan: &QueryPlan,
    producer_input: NodeId,
    consumer_source: &LocalParquetSource,
    options: DynamicFilterExecutionOptions,
) -> bool {
    let Some(consumer_stats) = consumer_source.diagnostics.selected_row_group_stats.as_ref() else {
        return false;
    };
    let Some(consumer_bytes) = consumer_stats.selected_source_bytes() else {
        return false;
    };
    if consumer_bytes > options.dimension_sized_consumer_bytes {
        return false;
    }

    let Some(producer_rows) = estimated_output_rows(plan, producer_input) else {
        return false;
    };
    if producer_rows <= options.large_local_parquet_range_producer_rows as u64 {
        return false;
    }

    consumer_stats.selected_row_groups <= options.tiny_local_parquet_consumer_row_groups
        || estimated_output_bytes(plan, producer_input).is_some_and(|producer_bytes| producer_bytes > consumer_bytes)
}

pub(super) fn producer_keyset_exceeds_runtime_limit(
    plan: &QueryPlan,
    producer_input: NodeId,
    options: DynamicFilterExecutionOptions,
) -> bool {
    exact_unfiltered_output_rows(plan, producer_input).is_some_and(|rows| rows > options.max_build_rows as u64)
}

fn exact_unfiltered_output_rows(plan: &QueryPlan, node_id: NodeId) -> Option<u64> {
    match &plan.node(node_id)?.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.reader_predicate.is_none() => {
                let rows = source
                    .files
                    .iter()
                    .map(|file| {
                        file.row_group_rows.as_ref()?.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows))
                    })
                    .try_fold(0_u64, |sum, rows| sum.checked_add(rows?))?;
                Some(source.scan_limit.map_or(rows, |limit| rows.min(limit as u64)))
            },
            SourceNode::LiteralBatch(source) => Some(source.row_count() as u64),
            SourceNode::LocalParquet(_) | SourceNode::GpuResident(_) => None,
        },
        PlanNode::Projection(projection) => exact_unfiltered_output_rows(plan, projection.input),
        PlanNode::Sort(sort) => exact_unfiltered_output_rows(plan, sort.input),
        PlanNode::Limit(limit) => exact_unfiltered_output_rows(plan, limit.input).map(|rows| {
            limit.spec.fetch.map_or(rows.saturating_sub(limit.spec.skip as u64), |fetch| {
                rows.min(limit.spec.skip.saturating_add(fetch) as u64).saturating_sub(limit.spec.skip as u64)
            })
        }),
        PlanNode::Union(union) => union
            .inputs
            .iter()
            .try_fold(0_u64, |total, input| total.checked_add(exact_unfiltered_output_rows(plan, *input)?)),
        PlanNode::Filter(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Join(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_)
        | PlanNode::VectorAlgorithm(_)
        | PlanNode::MlAlgorithm(_) => None,
    }
}

#[derive(Debug, Clone)]
struct ResidualInputEquality {
    left_column: String,
    right_column: String,
}

fn collect_residual_input_equalities(expr: &Expr, out: &mut Vec<ResidualInputEquality>) {
    match expr {
        Expr::Binary { left, op: BinaryOp::And, right, .. } => {
            collect_residual_input_equalities(left, out);
            collect_residual_input_equalities(right, out);
        },
        Expr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            if let Some(equality) = residual_input_equality(left, right) {
                out.push(equality);
            }
        },
        Expr::Alias { expr, .. } => collect_residual_input_equalities(expr, out),
        _ => {},
    }
}

fn residual_input_equality(left: &Expr, right: &Expr) -> Option<ResidualInputEquality> {
    let (left_side, left_column) = residual_input_column(left)?;
    let (right_side, right_column) = residual_input_column(right)?;
    match (left_side, right_side) {
        (JoinSide::Left, JoinSide::Right) => Some(ResidualInputEquality { left_column, right_column }),
        (JoinSide::Right, JoinSide::Left) => {
            Some(ResidualInputEquality { left_column: right_column, right_column: left_column })
        },
        _ => None,
    }
}

fn residual_input_column(expr: &Expr) -> Option<(JoinSide, String)> {
    let Expr::Column(column) = expr else {
        return None;
    };
    match column.qualifier()? {
        ColumnQualifier::Input { index: 0 } => Some((JoinSide::Left, column.name().to_string())),
        ColumnQualifier::Input { index: 1 } => Some((JoinSide::Right, column.name().to_string())),
        _ => None,
    }
}
