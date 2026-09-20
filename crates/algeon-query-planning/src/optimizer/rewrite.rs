use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::plan::{NodeId, NodeRemap, QueryNode, QueryPlan, ResultSink};
// Canonical home is `crate::plan`; re-exported while optimizer rules still
// spell node-input remapping through this module.
pub(super) use crate::plan::{clone_plan_node_with_inputs, remap_plan_node_inputs};

pub(super) fn sink_with_input(sink: &ResultSink, input: NodeId) -> ResultSink {
    match sink {
        ResultSink::GpuResident { .. } => ResultSink::GpuResident { input },
        ResultSink::HostArrow { schema, .. } => ResultSink::HostArrow { input, schema: Arc::clone(schema) },
    }
}

pub(super) fn rewrite_sink(sink: &ResultSink, input: NodeId) -> ResultSink {
    sink_with_input(sink, input)
}

pub(super) fn finish_rewritten_plan(nodes: Vec<QueryNode>, root: NodeId, sink: &ResultSink) -> QueryPlan {
    let sink = rewrite_sink(sink, root);
    QueryPlan::from_rewritten_parts(nodes, root, sink)
}

pub(super) fn finish_compacted_rewritten_plan(
    nodes: Vec<QueryNode>,
    root: NodeId,
    sink: &ResultSink,
) -> (QueryPlan, NodeRemap) {
    let (nodes, root, remap) = compact_reachable_rewritten_nodes(nodes, root);
    (finish_rewritten_plan(nodes, root, sink), remap)
}

pub(super) fn sink_reachable_nodes(plan: &QueryPlan) -> BTreeSet<NodeId> {
    let mut reachable = BTreeSet::new();
    let mut stack = vec![plan.sink().input()];
    while let Some(node_id) = stack.pop() {
        if !reachable.insert(node_id) {
            continue;
        }
        if let Some(node) = plan.node(node_id) {
            stack.extend(node.input_ids().iter().copied());
        }
    }
    reachable
}

pub(crate) fn parent_counts(plan: &QueryPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.node_count()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            counts[input.index()] += 1;
        }
    }
    counts
}

pub(super) fn compact_reachable_rewritten_nodes(
    nodes: Vec<QueryNode>,
    root: NodeId,
) -> (Vec<QueryNode>, NodeId, NodeRemap) {
    let mut reachable = vec![false; nodes.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let node = &nodes[id.index()];
        if reachable[id.index()] {
            continue;
        }
        reachable[id.index()] = true;
        stack.extend(node.input_ids().iter().copied());
    }

    let mut forward = BTreeMap::new();
    for node in nodes.iter().filter(|node| reachable[node.id.index()]) {
        forward.insert(node.id, NodeId::from_index_for_internal_use(forward.len()));
    }
    let new_root = *forward.get(&root).expect("rewritten root must be reachable");
    let compacted = nodes
        .into_iter()
        .filter(|node| reachable[node.id.index()])
        .map(|node| {
            let id = *forward.get(&node.id).expect("reachable node must have a remapped id");
            QueryNode { id, kind: remap_plan_node_inputs(node.kind, &forward), facts: node.facts }
        })
        .collect::<Vec<_>>();
    (compacted, new_root, NodeRemap::new(forward))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
        plan::{
            AdvisoryOutputBinding, AdvisoryOutputStatistics, AdvisoryStatistic, AggregateNode, EdgeNormalizeNode,
            EdgeNormalizeSpec, FilterNode, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams,
            GraphAlgorithmSpec, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs,
            GraphConstructionOptions, GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphPredicateInput, JoinKey,
            JoinNode, JoinSpec, JoinType, LimitNode, PlanFacts, PlanNode, ProjectionNode, ProjectionOutput,
            QueryPlanBuilder, SchemaBehavior, SortNode, UnionNode, VectorAlgorithm, VectorAlgorithmNode,
            VectorAlgorithmParams, VectorAlgorithmSpec, VectorBruteForceKnnParams, VectorDistanceMetric,
            VectorKMeansParams, VectorPcaParams, VectorRelationBinding, VectorRowIdBinding, WindowNode, WindowSpec,
            vector_kmeans_output_schema, vector_knn_output_schema, vector_pca_output_schema,
        },
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn clone_plan_node_with_inputs_remaps_every_current_plan_node_variant() {
        for (label, node, expected_inputs) in rewrite_helper_coverage_cases() {
            let cloned = clone_plan_node_with_inputs(&node, remapped);

            assert_eq!(cloned.input_ids(), expected_inputs.as_slice(), "{label}");
            assert_graph_bfs_side_inputs_match_input_list(label, &cloned);
            assert_vector_bindings_match_input_list(label, &cloned);
        }
    }

    #[test]
    fn remap_plan_node_inputs_remaps_every_current_plan_node_variant() {
        let remap = (0..16).map(|index| (node_id(index), remapped(node_id(index)))).collect::<BTreeMap<_, _>>();
        for (label, node, expected_inputs) in rewrite_helper_coverage_cases() {
            let remapped = remap_plan_node_inputs(node, &remap);

            assert_eq!(remapped.input_ids(), expected_inputs.as_slice(), "{label}");
            assert_graph_bfs_side_inputs_match_input_list(label, &remapped);
            assert_vector_bindings_match_input_list(label, &remapped);
        }
    }

    #[test]
    fn finish_compacted_rewritten_plan_remaps_root_sink_and_drops_unreachable_nodes() {
        let schema = test_schema(&["x"]);
        let source = QueryNode {
            id: node_id(0),
            kind: PlanNode::Source(Box::new(SourceNode::GpuResident(GpuResidentSource::new(
                "reachable",
                Arc::clone(&schema),
            )))),
            facts: PlanFacts::device(Arc::clone(&schema), SchemaBehavior::SourceOutput),
        };
        let unreachable = QueryNode {
            id: node_id(1),
            kind: PlanNode::Source(Box::new(SourceNode::GpuResident(GpuResidentSource::new(
                "unreachable",
                Arc::clone(&schema),
            )))),
            facts: PlanFacts::device(Arc::clone(&schema), SchemaBehavior::SourceOutput),
        };
        let projection = QueryNode {
            id: node_id(2),
            kind: PlanNode::Projection(ProjectionNode::new(
                node_id(0),
                vec![ProjectionOutput::new(Expr::column("x"), Field::new("x", DataType::Int32, false))],
            )),
            facts: PlanFacts::device(Arc::clone(&schema), SchemaBehavior::Projection),
        };
        let sink = ResultSink::HostArrow { input: node_id(2), schema };

        let (plan, _) = finish_compacted_rewritten_plan(vec![source, unreachable, projection], node_id(2), &sink);

        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.root_id(), node_id(1));
        assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == node_id(1)));
        let PlanNode::Projection(projection) = &plan.root_node().kind else {
            panic!("compacted root should remain the projection");
        };
        assert_eq!(projection.input, node_id(0));
        assert!(plan.unreachable_node_ids().is_empty());
    }

    #[test]
    fn compaction_keeps_advisory_on_moved_node() {
        let schema = test_schema(&["x"]);
        let mut builder = QueryPlanBuilder::new();
        let a = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("a", Arc::clone(&schema))));
        let _unreachable =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("x", Arc::clone(&schema))));
        let b = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("b", Arc::clone(&schema))));
        let join = builder.add_join(
            a,
            b,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "x", "x"), ColumnRef::input(1, "x", "x"))],
            ),
            Arc::clone(&schema),
        );
        builder.set_advisory_output(AdvisoryOutputBinding {
            node: b,
            schema: Arc::clone(&schema),
            frontend: "test frontend".to_owned(),
            frontend_version: "1".to_owned(),
            statistics: AdvisoryOutputStatistics {
                rows: AdvisoryStatistic::Inexact(100),
                bytes: AdvisoryStatistic::Inexact(800),
            },
            unknown_reason: None,
        });
        let source = builder.finish_with_host_sink(join);
        assert_eq!(b, node_id(2));

        let (compacted, remap) =
            finish_compacted_rewritten_plan(source.nodes().to_vec(), source.root_id(), source.sink());
        let rewritten = compacted.with_advisory_outputs_from_remapped(&source, &remap);

        assert_eq!(rewritten.node_count(), 3);
        let bindings = rewritten.advisory_outputs();
        assert_eq!(bindings.len(), 1, "advisory must survive on its moved node");
        assert_eq!(bindings[0].node, node_id(1));
        assert_eq!(bindings[0].statistics.rows, AdvisoryStatistic::Inexact(100));
        assert_eq!(bindings[0].statistics.bytes, AdvisoryStatistic::Inexact(800));
        assert_eq!(bindings[0].unknown_reason, None);
    }

    fn rewrite_helper_coverage_cases() -> Vec<(&'static str, PlanNode, Vec<NodeId>)> {
        vec![
            (
                "source",
                PlanNode::Source(Box::new(SourceNode::GpuResident(GpuResidentSource::new(
                    "input",
                    test_schema(&["x"]),
                )))),
                vec![],
            ),
            (
                "filter",
                PlanNode::Filter(FilterNode { input: node_id(0), predicate: Expr::column("x") }),
                vec![remapped(node_id(0))],
            ),
            (
                "projection",
                PlanNode::Projection(ProjectionNode::new(
                    node_id(1),
                    vec![ProjectionOutput::new(Expr::column("x"), Field::new("x", DataType::Int32, false))],
                )),
                vec![remapped(node_id(1))],
            ),
            (
                "aggregate",
                PlanNode::Aggregate(AggregateNode {
                    input: node_id(2),
                    spec: AggregateSpec::new(vec![Expr::column("x")], Vec::new()),
                }),
                vec![remapped(node_id(2))],
            ),
            (
                "window",
                PlanNode::Window(WindowNode {
                    input: node_id(3),
                    spec: WindowSpec::new(Vec::new(), Vec::new(), Vec::new()),
                }),
                vec![remapped(node_id(3))],
            ),
            (
                "sort",
                PlanNode::Sort(SortNode {
                    input: node_id(4),
                    spec: SortSpec::new(vec![SortExpr::new(Expr::column("x"), false, false)]),
                }),
                vec![remapped(node_id(4))],
            ),
            (
                "limit",
                PlanNode::Limit(LimitNode { input: node_id(5), spec: LimitSpec::fetch(10) }),
                vec![remapped(node_id(5))],
            ),
            (
                "join",
                PlanNode::Join(JoinNode {
                    inputs: [node_id(6), node_id(7)],
                    spec: JoinSpec::new(JoinType::Inner, Vec::new()),
                }),
                vec![remapped(node_id(6)), remapped(node_id(7))],
            ),
            (
                "union",
                PlanNode::Union(UnionNode { inputs: vec![node_id(8), node_id(9), node_id(10)] }),
                vec![remapped(node_id(8)), remapped(node_id(9)), remapped(node_id(10))],
            ),
            (
                "edge_normalize",
                PlanNode::EdgeNormalize(EdgeNormalizeNode {
                    input: node_id(11),
                    spec: EdgeNormalizeSpec::python_cugraph(
                        GraphEdgeColumnBindings::new("src", "dst", Option::<String>::None),
                        &GraphConstructionOptions::default(),
                    ),
                }),
                vec![remapped(node_id(11))],
            ),
            (
                "graph_algorithm",
                graph_algorithm_with_all_bfs_side_inputs(),
                (0..=6).map(|index| remapped(node_id(index))).collect(),
            ),
            ("vector_brute_force_knn", vector_brute_force_knn(), vec![remapped(node_id(12)), remapped(node_id(13))]),
            (
                "vector_kmeans_with_initial_centroids",
                vector_kmeans_with_initial_centroids(),
                vec![remapped(node_id(13)), remapped(node_id(14))],
            ),
            ("vector_pca", vector_pca(), vec![remapped(node_id(15))]),
        ]
    }

    fn graph_algorithm_with_all_bfs_side_inputs() -> PlanNode {
        let params = GraphBfsParams {
            source_vertex: 0,
            source_vertices: Some(GraphPredicateInput::new(node_id(1), "source_vertex")),
            depth_limit: None,
            predicates: GraphBfsPredicateInputs {
                include_vertices: Some(GraphPredicateInput::new(node_id(2), "vertex")),
                exclude_vertices: Some(GraphPredicateInput::new(node_id(3), "vertex")),
                target_vertices: Some(GraphPredicateInput::new(node_id(4), "vertex")),
                include_edge_ids: Some(GraphPredicateInput::new(node_id(5), "edge_id")),
                include_edges: Some(GraphEdgePredicateInput::new(node_id(6), "src", "dst", Some("edge_id"))),
            },
            endpoint_policy: GraphBfsEndpointPolicy::Strict,
            path_output: None,
            return_target_info: true,
        };
        let spec = GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(params)),
        );
        PlanNode::GraphAlgorithm(GraphAlgorithmNode { input: node_id(0), inputs: (0..=6).map(node_id).collect(), spec })
    }

    fn assert_graph_bfs_side_inputs_match_input_list(label: &str, node: &PlanNode) {
        let PlanNode::GraphAlgorithm(graph) = node else {
            return;
        };
        let GraphAlgorithmParams::Bfs(params) = &graph.spec.params else {
            panic!("{label}: expected BFS graph params");
        };
        assert_eq!(graph.input, remapped(node_id(0)), "{label}: primary graph input");
        assert_eq!(graph.inputs, (0..=6).map(|index| remapped(node_id(index))).collect::<Vec<_>>(), "{label}");
        assert_eq!(params.source_vertices.as_ref().map(|input| input.input), Some(remapped(node_id(1))));
        assert_eq!(params.predicates.include_vertices.as_ref().map(|input| input.input), Some(remapped(node_id(2))));
        assert_eq!(params.predicates.exclude_vertices.as_ref().map(|input| input.input), Some(remapped(node_id(3))));
        assert_eq!(params.predicates.target_vertices.as_ref().map(|input| input.input), Some(remapped(node_id(4))));
        assert_eq!(params.predicates.include_edge_ids.as_ref().map(|input| input.input), Some(remapped(node_id(5))));
        assert_eq!(params.predicates.include_edges.as_ref().map(|input| input.input), Some(remapped(node_id(6))));
    }

    fn vector_brute_force_knn() -> PlanNode {
        let params = VectorBruteForceKnnParams::new(
            vector_relation_binding(node_id(12)),
            vector_relation_binding(node_id(13)),
            3,
            VectorDistanceMetric::L2Expanded,
        );
        vector_algorithm_node(
            VectorAlgorithm::BruteForceKnn,
            VectorAlgorithmParams::BruteForceKnn(params),
            vector_knn_output_schema(DataType::Int32, DataType::Int32),
        )
    }

    fn vector_kmeans_with_initial_centroids() -> PlanNode {
        let params = VectorKMeansParams::new(vector_relation_binding(node_id(13)), 4)
            .with_initial_centroids(vector_relation_binding(node_id(14)));
        vector_algorithm_node(
            VectorAlgorithm::KMeans,
            VectorAlgorithmParams::KMeans(params),
            vector_kmeans_output_schema(DataType::Int32),
        )
    }

    fn vector_pca() -> PlanNode {
        let params = VectorPcaParams::new(vector_relation_binding(node_id(15)), 1);
        vector_algorithm_node(
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(params),
            vector_pca_output_schema(DataType::Int32, 1),
        )
    }

    fn vector_algorithm_node(
        algorithm: VectorAlgorithm,
        params: VectorAlgorithmParams,
        output_schema: Arc<Schema>,
    ) -> PlanNode {
        let spec = VectorAlgorithmSpec::new(algorithm, params, output_schema);
        let inputs = spec.params.relation_bindings().map(|binding| binding.input).collect();
        PlanNode::VectorAlgorithm(VectorAlgorithmNode { inputs, spec })
    }

    fn vector_relation_binding(input: NodeId) -> VectorRelationBinding {
        VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"])
    }

    fn assert_vector_bindings_match_input_list(label: &str, node: &PlanNode) {
        let PlanNode::VectorAlgorithm(vector) = node else {
            return;
        };
        match &vector.spec.params {
            VectorAlgorithmParams::BruteForceKnn(params) => {
                assert_eq!(vector.inputs, [params.dataset.input, params.queries.input], "{label}");
            },
            VectorAlgorithmParams::IvfFlat(params) => {
                assert_eq!(vector.inputs, [params.dataset.input, params.queries.input], "{label}");
            },
            VectorAlgorithmParams::IvfPq(params) => {
                assert_eq!(vector.inputs, [params.dataset.input, params.queries.input], "{label}");
            },
            VectorAlgorithmParams::Cagra(params) => {
                assert_eq!(vector.inputs, [params.dataset.input, params.queries.input], "{label}");
            },
            VectorAlgorithmParams::PairwiseDistance(params) => {
                assert_eq!(vector.inputs, [params.x.input, params.y.input], "{label}");
            },
            VectorAlgorithmParams::TopK(params) => {
                assert_eq!(vector.inputs, [params.input.input], "{label}");
            },
            VectorAlgorithmParams::KMeans(params) => {
                assert_eq!(vector.inputs.first().copied(), Some(params.input.input), "{label}: input");
                assert_eq!(
                    vector.inputs.get(1).copied(),
                    params.initial_centroids.as_ref().map(|binding| binding.input),
                    "{label}: initial centroids"
                );
            },
            VectorAlgorithmParams::Pca(params) => {
                assert_eq!(vector.inputs, [params.input.input], "{label}");
            },
        }
    }

    fn test_schema(fields: &[&str]) -> Arc<Schema> {
        Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
    }

    fn remapped(id: NodeId) -> NodeId {
        node_id(id.index() + 100)
    }

    fn node_id(index: usize) -> NodeId {
        NodeId::from_index_for_internal_use(index)
    }
}
