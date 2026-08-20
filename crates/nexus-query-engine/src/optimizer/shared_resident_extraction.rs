use std::collections::BTreeSet;

use super::rewrite;
use crate::{
    graph_memory::query_plan_hard_row_upper_bound,
    plan::{NodeId, PlanFacts, PlanNode, QueryPlan, ResultSink, SchemaBehavior},
    source::{GpuResidentSource, SourceNode},
};

/// One closed shared DAG fragment extracted behind a GPU-resident boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedResidentProducerPlan {
    /// Runtime occurrence identity used by the rewritten consumer source.
    pub input_name: String,
    /// Exact producer plan whose output remains device-resident.
    pub plan: QueryPlan,
}

/// A consumer plan and the shared producer fragments it consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedResidentPlan {
    /// Rewritten plan whose repeated closed subplans are resident sources.
    pub consumer: QueryPlan,
    /// Producer plans in deterministic consumer dependency order.
    pub producers: Vec<SharedResidentProducerPlan>,
}

/// Extracts closed repeated subplans so downstream consumers can use retained runtime evidence.
pub fn extract_shared_resident_subplans(plan: QueryPlan) -> SharedResidentPlan {
    // Dynamic-filter routes carry node identities that extraction cannot remap yet.
    if !plan.dynamic_filter_routes().is_empty() {
        return SharedResidentPlan { consumer: plan, producers: Vec::new() };
    }

    let parents = parents_by_node(&plan);
    let mut selected = Vec::new();
    for node in plan.nodes().iter().rev() {
        if parents[node.id.index()].len() < 2
            || subtree_contains_gpu_resident_source(&plan, node.id)
            || !subtree_is_closed(&plan, node.id, &parents)
            || selected.iter().any(|selected_root| subtree_contains(&plan, *selected_root, node.id))
        {
            continue;
        }
        selected.push(node.id);
    }
    selected.sort();

    let mut nodes = plan.nodes().to_vec();
    let mut producers = Vec::with_capacity(selected.len());
    let mut input_names = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut input_index = 0;
    for root in selected {
        let producer = rewrite::finish_compacted_rewritten_plan(
            plan.nodes().to_vec(),
            root,
            &ResultSink::GpuResident { input: root },
        )
        .with_rewrite_state_from(&plan);
        let input_name = loop {
            let candidate = format!("__nexus_shared_resident_{input_index}");
            input_index += 1;
            if input_names.insert(candidate.clone()) {
                break candidate;
            }
        };
        let schema = producer.output_schema().clone();
        let mut source = GpuResidentSource::new(&input_name, schema.clone());
        if let Some(row_count_upper_bound) = query_plan_hard_row_upper_bound(&producer) {
            source = source.with_row_count_upper_bound(row_count_upper_bound);
        }
        nodes[root.index()].kind = PlanNode::Source(Box::new(SourceNode::GpuResident(source)));
        nodes[root.index()].facts = PlanFacts::device(schema, SchemaBehavior::SourceOutput);
        producers.push(SharedResidentProducerPlan { input_name, plan: producer });
    }

    let consumer =
        rewrite::finish_compacted_rewritten_plan(nodes, plan.root_id(), plan.sink()).with_rewrite_state_from(&plan);
    SharedResidentPlan { consumer, producers }
}

fn parents_by_node(plan: &QueryPlan) -> Vec<Vec<NodeId>> {
    let mut parents = vec![Vec::new(); plan.node_count()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            parents[input.index()].push(node.id);
        }
    }
    parents
}

fn subtree_is_closed(plan: &QueryPlan, root: NodeId, parents: &[Vec<NodeId>]) -> bool {
    let subtree = subtree_nodes(plan, root);
    subtree.iter().all(|node| *node == root || parents[node.index()].iter().all(|parent| subtree.contains(parent)))
}

fn subtree_contains_gpu_resident_source(plan: &QueryPlan, root: NodeId) -> bool {
    subtree_nodes(plan, root).into_iter().any(|node| {
        matches!(
            &plan.node(node).expect("reachable shared node must exist").kind,
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))
        )
    })
}

fn subtree_contains(plan: &QueryPlan, root: NodeId, candidate: NodeId) -> bool {
    subtree_nodes(plan, root).contains(&candidate)
}

fn subtree_nodes(plan: &QueryPlan, root: NodeId) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        if nodes.insert(node_id) {
            stack.extend(plan.node(node_id).expect("reachable shared node must exist").input_ids().iter().copied());
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlanBuilder},
        source::{LocalParquetFile, LocalParquetSource},
    };

    #[test]
    fn repeated_filtered_source_becomes_one_resident_producer() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let file = LocalParquetFile::new("/tmp/shared-resident-not-read.parquet")
            .with_file_size_bytes(64)
            .with_row_group_bytes(vec![64])
            .with_projected_row_group_bytes(vec![64], vec![64])
            .with_row_group_rows(vec![8]);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![file],
            Arc::clone(&schema),
            vec!["id".to_string(), "name".to_string()],
        )));
        let filtered = builder.add_filter(
            source,
            Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::Literal(LiteralValue::Utf8("a".to_string()))),
            Arc::clone(&schema),
        );
        let output = Arc::new(Schema::new(vec![
            Field::new("left_name", DataType::Utf8, false),
            Field::new("right_name", DataType::Utf8, false),
        ]));
        let join = builder.add_join(
            filtered,
            filtered,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("name"), "left_name"),
                JoinOutputColumn::new(JoinSide::Right, ColumnRef::unqualified("name"), "right_name"),
            ]),
            Arc::clone(&output),
        );
        let split = extract_shared_resident_subplans(builder.finish_with_host_sink_schema(join, output));

        assert_eq!(split.producers.len(), 1);
        assert_eq!(split.producers[0].plan.root_node().output_schema(), &schema);
        let resident_sources = split
            .consumer
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Source(source) => match source.as_ref() {
                    SourceNode::GpuResident(source) => Some(source),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(resident_sources.len(), 1);
        assert_eq!(resident_sources[0].row_count_upper_bound, Some(8));
        assert_eq!(
            split.consumer.nodes().iter().find(|node| matches!(node.kind, PlanNode::Join(_))).unwrap().input_ids(),
            &[resident_sources_node_id(&split.consumer), resident_sources_node_id(&split.consumer)]
        );
    }

    fn resident_sources_node_id(plan: &QueryPlan) -> NodeId {
        plan.nodes()
            .iter()
            .find(|node| {
                matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_)))
            })
            .expect("consumer must contain the extracted resident source")
            .id
    }
}
