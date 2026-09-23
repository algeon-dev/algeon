use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::plan::{GraphAlgorithmParams, GraphAlgorithmSpec, NodeId, PlanNode, QueryNode, QueryPlan, ResultSink};

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

pub(super) fn finish_compacted_rewritten_plan(nodes: Vec<QueryNode>, root: NodeId, sink: &ResultSink) -> QueryPlan {
    let (nodes, root) = compact_reachable_rewritten_nodes(nodes, root);
    finish_rewritten_plan(nodes, root, sink)
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

#[doc(hidden)]
pub fn parent_counts(plan: &QueryPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.node_count()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            counts[input.index()] += 1;
        }
    }
    counts
}

pub(super) fn clone_plan_node_with_inputs(
    kind: &PlanNode,
    mut rewrite_input: impl FnMut(NodeId) -> NodeId,
) -> PlanNode {
    match kind {
        PlanNode::Source(source) => PlanNode::Source(source.clone()),
        PlanNode::Filter(filter) => {
            let mut filter = filter.clone();
            filter.input = rewrite_input(filter.input);
            PlanNode::Filter(filter)
        },
        PlanNode::Projection(projection) => {
            let mut projection = projection.clone();
            projection.input = rewrite_input(projection.input);
            PlanNode::Projection(projection)
        },
        PlanNode::Aggregate(aggregate) => {
            let mut aggregate = aggregate.clone();
            aggregate.input = rewrite_input(aggregate.input);
            PlanNode::Aggregate(aggregate)
        },
        PlanNode::Window(window) => {
            let mut window = window.clone();
            window.input = rewrite_input(window.input);
            PlanNode::Window(window)
        },
        PlanNode::Sort(sort) => {
            let mut sort = sort.clone();
            sort.input = rewrite_input(sort.input);
            PlanNode::Sort(sort)
        },
        PlanNode::Limit(limit) => {
            let mut limit = limit.clone();
            limit.input = rewrite_input(limit.input);
            PlanNode::Limit(limit)
        },
        PlanNode::Join(join) => {
            let mut join = join.clone();
            join.inputs = [rewrite_input(join.inputs[0]), rewrite_input(join.inputs[1])];
            PlanNode::Join(join)
        },
        PlanNode::Union(union) => {
            let mut union = union.clone();
            union.inputs = union.inputs.iter().map(|input| rewrite_input(*input)).collect();
            PlanNode::Union(union)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let mut edge_normalize = edge_normalize.clone();
            edge_normalize.input = rewrite_input(edge_normalize.input);
            PlanNode::EdgeNormalize(edge_normalize)
        },
        PlanNode::GraphAlgorithm(graph) => {
            let mut graph = graph.clone();
            let old_inputs = graph.inputs.clone();
            if old_inputs.is_empty() {
                graph.input = rewrite_input(graph.input);
                graph.inputs = vec![graph.input];
                return PlanNode::GraphAlgorithm(graph);
            }

            let new_inputs = old_inputs.iter().map(|input| rewrite_input(*input)).collect::<Vec<_>>();
            let input_remap = old_inputs.iter().copied().zip(new_inputs.iter().copied()).collect::<BTreeMap<_, _>>();
            graph.input = new_inputs[0];
            graph.inputs = new_inputs;
            remap_graph_algorithm_spec_side_inputs(&mut graph.spec, &input_remap);
            PlanNode::GraphAlgorithm(graph)
        },
        PlanNode::VectorAlgorithm(vector) => {
            let mut vector = vector.clone();
            vector.spec.params.remap_relation_inputs(&mut rewrite_input);
            vector.inputs = vector.spec.params.relation_bindings().map(|binding| binding.input).collect();
            PlanNode::VectorAlgorithm(vector)
        },
    }
}

pub(super) fn remap_plan_node_inputs(kind: PlanNode, remap: &BTreeMap<NodeId, NodeId>) -> PlanNode {
    clone_plan_node_with_inputs(&kind, |input| remap_node_id(input, remap))
}

pub(super) fn compact_reachable_rewritten_nodes(nodes: Vec<QueryNode>, root: NodeId) -> (Vec<QueryNode>, NodeId) {
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

    let mut remap = BTreeMap::new();
    for node in nodes.iter().filter(|node| reachable[node.id.index()]) {
        remap.insert(node.id, NodeId::from_index_for_internal_use(remap.len()));
    }
    let new_root = *remap.get(&root).expect("rewritten root must be reachable");
    let compacted = nodes
        .into_iter()
        .filter(|node| reachable[node.id.index()])
        .map(|node| {
            let id = *remap.get(&node.id).expect("reachable node must have a remapped id");
            QueryNode { id, kind: remap_plan_node_inputs(node.kind, &remap), facts: node.facts }
        })
        .collect::<Vec<_>>();
    (compacted, new_root)
}

fn remap_node_id(id: NodeId, remap: &BTreeMap<NodeId, NodeId>) -> NodeId {
    *remap.get(&id).expect("reachable input node must have a remapped id")
}

fn remap_graph_algorithm_spec_side_inputs(spec: &mut GraphAlgorithmSpec, input_remap: &BTreeMap<NodeId, NodeId>) {
    let GraphAlgorithmParams::Bfs(params) = &mut spec.params else {
        return;
    };
    if let Some(input) = &mut params.source_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.exclude_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.target_vertices {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_edge_ids {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
    if let Some(input) = &mut params.predicates.include_edges {
        input.input = remap_graph_side_input(input.input, input_remap);
    }
}

fn remap_graph_side_input(input: NodeId, input_remap: &BTreeMap<NodeId, NodeId>) -> NodeId {
    *input_remap.get(&input).expect("graph side-input binding must be present in graph input list")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, Expr, LimitSpec, SortExpr, SortSpec},
        plan::{
            AggregateNode, EdgeNormalizeNode, EdgeNormalizeSpec, FilterNode, GraphAlgorithm, GraphAlgorithmNode,
            GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs,
            GraphConstructionOptions, GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphPredicateInput, JoinNode,
            JoinSpec, JoinType, LimitNode, PlanFacts, ProjectionNode, ProjectionOutput, SchemaBehavior, SortNode,
            UnionNode, VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams, VectorAlgorithmSpec,
            VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams,
            VectorRelationBinding, VectorRowIdBinding, WindowNode, WindowSpec, vector_kmeans_output_schema,
            vector_knn_output_schema, vector_pca_output_schema,
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

        let plan = finish_compacted_rewritten_plan(vec![source, unreachable, projection], node_id(2), &sink);

        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.root_id(), node_id(1));
        assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == node_id(1)));
        let PlanNode::Projection(projection) = &plan.root_node().kind else {
            panic!("compacted root should remain the projection");
        };
        assert_eq!(projection.input, node_id(0));
        assert!(plan.unreachable_node_ids().is_empty());
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
