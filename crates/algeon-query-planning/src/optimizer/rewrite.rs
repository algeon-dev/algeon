// Canonical home is `crate::plan`; re-exported while optimizer rules still
// spell node-input remapping through this module.
use crate::plan::QueryPlan;
pub(super) use crate::plan::clone_plan_node_with_inputs;

pub(crate) fn parent_counts(plan: &QueryPlan) -> Vec<usize> {
    let mut counts = vec![0; plan.node_count()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            counts[input.index()] += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        dynamic_filter::{
            DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
            DynamicFilterProducer, DynamicFilterRoute,
        },
        expr::{AggregateSpec, ColumnRef, Expr, LimitSpec, SortExpr, SortSpec},
        plan::{
            AdvisoryOutputBinding, AdvisoryOutputStatistics, AdvisoryStatistic, AggregateNode, EdgeNormalizeNode,
            EdgeNormalizeSpec, FilterNode, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams,
            GraphAlgorithmSpec, GraphBfsEndpointPolicy, GraphBfsParams, GraphBfsPredicateInputs,
            GraphConstructionOptions, GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphPredicateInput, JoinKey,
            JoinNode, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitNode, NodeId, PlanNode, ProjectionNode,
            ProjectionOutput, QueryNode, QueryPlanBuilder, ResultSink, RewrittenNodes, SortNode, UnionNode,
            VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams, VectorAlgorithmSpec,
            VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansParams, VectorPcaParams,
            VectorRelationBinding, VectorRowIdBinding, WindowNode, WindowSpec, remap_plan_node_inputs,
            vector_kmeans_output_schema, vector_knn_output_schema, vector_pca_output_schema,
        },
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn clone_plan_node_with_inputs_remaps_every_current_plan_node_variant() {
        for (label, node, expected_inputs) in rewrite_helper_coverage_cases() {
            let cloned = clone_plan_node_with_inputs(&node, remapped);

            assert_eq!(cloned.input_ids(), expected_inputs.as_slice(), "{label}");
        }
    }

    #[test]
    fn remap_plan_node_inputs_remaps_every_current_plan_node_variant() {
        let remap = (0..16).map(|index| (node_id(index), remapped(node_id(index)))).collect::<BTreeMap<_, _>>();
        for (label, node, expected_inputs) in rewrite_helper_coverage_cases() {
            let remapped = remap_plan_node_inputs(node, &remap);

            assert_eq!(remapped.input_ids(), expected_inputs.as_slice(), "{label}");
        }
    }

    #[test]
    fn rebuilt_rewrite_compacts_unreachable_nodes_and_remaps_root_and_sink() {
        let schema = test_schema(&["x"]);
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("reachable", Arc::clone(&schema))));
        let projection = builder
            .add_projection(
                source,
                vec![ProjectionOutput::new(Expr::column("x"), Field::new("x", DataType::Int32, false))],
            )
            .unwrap();
        let source_plan = builder.finish_with_host_sink(projection).unwrap();
        let unreachable = SourceNode::GpuResident(GpuResidentSource::new("unreachable", Arc::clone(&schema)));
        let mut nodes = source_plan.nodes().to_vec();
        nodes.insert(
            1,
            QueryNode { id: node_id(1), kind: PlanNode::Source(Box::new(unreachable)), facts: nodes[0].facts.clone() },
        );
        nodes[2].id = node_id(2);
        let origin = BTreeMap::from([(source, node_id(0)), (projection, node_id(2))]);

        let plan = source_plan.rewrite(RewrittenNodes::rebuilt(nodes, node_id(2), origin)).unwrap();

        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.root_id(), node_id(1));
        assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == node_id(1)));
        let PlanNode::Projection(projection) = &plan.root_node().kind else {
            panic!("compacted root should remain the projection");
        };
        assert_eq!(projection.input, node_id(0));
    }

    #[test]
    fn compaction_keeps_advisory_on_moved_node() {
        let schema = test_schema(&["x"]);
        let mut builder = QueryPlanBuilder::new();
        let x = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("x", Arc::clone(&schema))));
        let a = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("a", Arc::clone(&schema))));
        let b = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("b", Arc::clone(&schema))));
        let join_spec = || {
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "x", "x"), ColumnRef::input(1, "x", "x"))],
            )
        };
        let join = builder.add_join(a, b, join_spec(), Arc::clone(&schema)).unwrap();
        let root = builder.add_union(vec![join, x], Arc::clone(&schema)).unwrap();
        builder
            .set_advisory_output(AdvisoryOutputBinding {
                node: b,
                schema: Arc::clone(&schema),
                frontend: "test frontend".to_owned(),
                frontend_version: "1".to_owned(),
                statistics: AdvisoryOutputStatistics {
                    rows: AdvisoryStatistic::Inexact(100),
                    bytes: AdvisoryStatistic::Inexact(800),
                },
                unknown_reason: None,
            })
            .unwrap();
        let source = builder.finish_with_host_sink(root).unwrap();
        assert_eq!(b, node_id(2));
        // The rewrite drops the union branch `x`; every kept node moves down
        // one index.
        let mut nodes = source.nodes()[1..4].to_vec();
        for (index, node) in nodes.iter_mut().enumerate() {
            node.id = node_id(index);
        }
        nodes[2].kind = PlanNode::Join(JoinNode { inputs: [node_id(0), node_id(1)], spec: join_spec() });
        let origin = BTreeMap::from([(a, node_id(0)), (b, node_id(1)), (join, node_id(2))]);

        let rewritten = source.rewrite(RewrittenNodes::rebuilt(nodes, node_id(2), origin)).unwrap();

        assert_eq!(rewritten.node_count(), 3);
        let bindings = rewritten.advisory_outputs();
        assert_eq!(bindings.len(), 1, "advisory must survive on its moved node");
        assert_eq!(bindings[0].node, node_id(1));
        assert_eq!(bindings[0].statistics.rows, AdvisoryStatistic::Inexact(100));
        assert_eq!(bindings[0].statistics.bytes, AdvisoryStatistic::Inexact(800));
        assert_eq!(bindings[0].unknown_reason, None);
    }

    #[test]
    fn rebuilt_rewrite_moves_dynamic_filter_routes_with_renumbered_nodes() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let source = |builder: &mut QueryPlanBuilder, name: &str| {
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(name, Arc::clone(&schema))))
        };
        let join_spec = || {
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))],
            )
            .with_output_projection(vec![JoinOutputColumn::new(
                JoinSide::Left,
                ColumnRef::unqualified("id"),
                "id",
            )])
        };
        let mut builder = QueryPlanBuilder::new();
        let x = source(&mut builder, "x");
        let a = source(&mut builder, "a");
        let b = source(&mut builder, "b");
        let first_join = builder.add_join(a, b, join_spec(), Arc::clone(&schema)).unwrap();
        let c = source(&mut builder, "c");
        let second_join = builder.add_join(first_join, c, join_spec(), Arc::clone(&schema)).unwrap();
        let root = builder.add_union(vec![second_join, x], Arc::clone(&schema)).unwrap();
        let route = DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::HashJoinBuild {
                join_node_id: first_join,
                build_side: JoinSide::Right,
                build_input: b,
            },
            DynamicFilterConsumer::HashJoinProbe { join_node_id: second_join, probe_side: JoinSide::Left },
            vec!["id".to_string()],
            vec!["id".to_string()],
            vec![DynamicFilterKind::Range],
            DynamicFilterConsumerLowering::PostDecodeOnly,
        );
        let plan = builder.finish_gpu_resident(root).unwrap().with_dynamic_filter_routes(vec![route]);
        // The rewrite drops the union and its branch `x`, which precedes every
        // route endpoint, so each kept node moves down one index.
        let shifted = |id: NodeId| node_id(id.index() - 1);
        let mut nodes = plan.nodes()[1..6].to_vec();
        for node in &mut nodes {
            node.id = shifted(node.id);
            node.kind =
                remap_plan_node_inputs(node.kind.clone(), &[a, b, first_join, c].map(|id| (id, shifted(id))).into());
        }
        let origin = [a, b, first_join, c, second_join].into_iter().map(|id| (id, shifted(id))).collect();

        let rewritten = plan.rewrite(RewrittenNodes::rebuilt(nodes, shifted(second_join), origin)).unwrap();

        let routes = rewritten.dynamic_filter_routes();
        assert_eq!(routes.len(), 1, "pure renumbering must keep the route");
        let DynamicFilterProducer::HashJoinBuild { join_node_id, build_input, .. } = &routes[0].producer else {
            panic!("producer kind must survive the remap");
        };
        assert_eq!(*join_node_id, shifted(first_join));
        assert_eq!(*build_input, shifted(b));
        let DynamicFilterConsumer::HashJoinProbe { join_node_id, .. } = &routes[0].consumer else {
            panic!("consumer kind must survive the remap");
        };
        assert_eq!(*join_node_id, shifted(second_join));
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
            source_vertices: Some(GraphPredicateInput::new(1, "source_vertex")),
            depth_limit: None,
            predicates: GraphBfsPredicateInputs {
                include_vertices: Some(GraphPredicateInput::new(2, "vertex")),
                exclude_vertices: Some(GraphPredicateInput::new(3, "vertex")),
                target_vertices: Some(GraphPredicateInput::new(4, "vertex")),
                include_edge_ids: Some(GraphPredicateInput::new(5, "edge_id")),
                include_edges: Some(GraphEdgePredicateInput::new(6, "src", "dst", Some("edge_id"))),
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
        PlanNode::GraphAlgorithm(GraphAlgorithmNode { inputs: (0..=6).map(node_id).collect(), spec })
    }

    fn vector_brute_force_knn() -> PlanNode {
        let params = VectorBruteForceKnnParams::new(
            vector_relation_binding(0),
            vector_relation_binding(1),
            3,
            VectorDistanceMetric::L2Expanded,
        );
        vector_algorithm_node(
            vec![node_id(12), node_id(13)],
            VectorAlgorithm::BruteForceKnn,
            VectorAlgorithmParams::BruteForceKnn(params),
            vector_knn_output_schema(DataType::Int32, DataType::Int32),
        )
    }

    fn vector_kmeans_with_initial_centroids() -> PlanNode {
        let params =
            VectorKMeansParams::new(vector_relation_binding(0), 4).with_initial_centroids(vector_relation_binding(1));
        vector_algorithm_node(
            vec![node_id(13), node_id(14)],
            VectorAlgorithm::KMeans,
            VectorAlgorithmParams::KMeans(params),
            vector_kmeans_output_schema(DataType::Int32),
        )
    }

    fn vector_pca() -> PlanNode {
        let params = VectorPcaParams::new(vector_relation_binding(0), 1);
        vector_algorithm_node(
            vec![node_id(15)],
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(params),
            vector_pca_output_schema(DataType::Int32, 1),
        )
    }

    fn vector_algorithm_node(
        inputs: Vec<NodeId>,
        algorithm: VectorAlgorithm,
        params: VectorAlgorithmParams,
        output_schema: Arc<Schema>,
    ) -> PlanNode {
        let spec = VectorAlgorithmSpec::new(algorithm, params, output_schema);
        PlanNode::VectorAlgorithm(VectorAlgorithmNode { inputs, spec })
    }

    fn vector_relation_binding(input: usize) -> VectorRelationBinding {
        VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"])
    }

    fn test_schema(fields: &[&str]) -> Arc<Schema> {
        Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
    }

    fn remapped(id: NodeId) -> NodeId {
        node_id(id.index() + 100)
    }

    fn node_id(index: usize) -> NodeId {
        NodeId::from_index_for_test(index)
    }
}
