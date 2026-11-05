//! `validate_query_plan` contract tests that exercise real RAPIDS support
//! semantics (filter-predicate lowering, cuGraph vertex id types), bound to
//! the engine's [`RapidsBackend`](query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].

use arrow::datatypes::DataType as ArrowDataType;
use query_engine::rapids_backend::rapids_analysis_context;
use query_planning::{
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions, GraphEdgeColumnBindings,
        QueryPlan, graph_result_schema_for_vertex_type,
    },
    policy::PlanningPolicy,
    validate_query_plan,
};

use super::*;

#[test]
fn validate_filter_uses_filter_lowering_for_large_left_deep_boolean_predicates() {
    let schema = Arc::new(Schema::new(vec![Field::new("ca_zip", ArrowDataType::Utf8, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let filter = builder.add_filter(source, left_deep_zip_range_disjunction(384)).unwrap();
    let plan = builder.finish_with_host_sink(filter).unwrap();

    validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect("large q08-class filter predicate should validate without generic lowering");
}

#[test]
fn validate_accepts_whole_input_aggregate_window_without_partition_keys() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int32, false),
        Field::new("v", ArrowDataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(input_schema)));
    let window = builder
        .add_window(
            source,
            WindowSpec::new(
                Vec::new(),
                Vec::new(),
                vec![WindowFunction::aggregate(
                    AggregateFunction::Sum,
                    Expr::column("v"),
                    Field::new("window_v", ArrowDataType::Int64, true),
                )],
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(window).unwrap();

    validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect("an empty partition list makes the whole input one partition");
}

#[test]
fn validate_graph_algorithm_rejects_incompatible_directedness() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", ArrowDataType::Int64, false),
        Field::new("dst", ArrowDataType::Int64, false),
        Field::new("weight", ArrowDataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::StronglyConnectedComponents,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, ..Default::default() },
        GraphAlgorithmParams::StronglyConnectedComponents(Default::default()),
    );
    let graph = builder.add_graph_algorithm(source, Vec::new(), spec).unwrap();
    let plan = builder.finish_with_host_sink(graph).unwrap();

    let error = validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect_err("SCC must reject undirected graph construction");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.graph_algorithm.validate");
    assert_eq!(error.fact_value("reason"), Some("unsupported_graph_directedness"));
    assert_eq!(error.fact_value("algorithm"), Some("strongly_connected_components"));
    assert_eq!(error.fact_value("directed"), Some("false"));
    assert_eq!(error.fact_value("required_directed"), Some("true"));
}

#[test]
fn validate_graph_algorithm_rejects_int64_for_c_api_int32_only_algorithms() {
    for (algorithm, params) in [
        (GraphAlgorithm::MinimumSpanningTree, GraphAlgorithmParams::MinimumSpanningTree(Default::default())),
        (
            GraphAlgorithm::SpectralModularityMaximization,
            GraphAlgorithmParams::SpectralModularityMaximization(Default::default()),
        ),
    ] {
        let int64_plan = graph_algorithm_validation_plan(
            algorithm,
            params.clone(),
            Arc::new(Schema::new(vec![
                Field::new("src", ArrowDataType::Int64, false),
                Field::new("dst", ArrowDataType::Int64, false),
                Field::new("weight", ArrowDataType::Float64, false),
            ])),
        );
        let error = validate_query_plan(&int64_plan, rapids_analysis_context(PlanningPolicy::default()))
            .expect_err("cuGraph C API algorithms must reject Int64 vertices");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(error.operation(), "query_engine.graph_algorithm.validate");
        assert_eq!(error.fact_value("reason"), Some("unsupported_vertex_dtype"));
        assert_eq!(error.fact_value("algorithm"), Some(algorithm.as_str()));
        assert_eq!(error.fact_value("edge_column_role"), Some("source"));
        assert_eq!(error.fact_value("vertex_id_type_support"), Some("int32_only_legacy_edge_type"));
        assert_eq!(error.fact_value("supported_vertex_dtypes"), Some("Int32"));

        let int32_plan = graph_algorithm_validation_plan(
            algorithm,
            params,
            Arc::new(Schema::new(vec![
                Field::new("src", ArrowDataType::Int32, false),
                Field::new("dst", ArrowDataType::Int32, false),
                Field::new("weight", ArrowDataType::Float64, false),
            ])),
        );
        validate_query_plan(&int32_plan, rapids_analysis_context(PlanningPolicy::default()))
            .expect("cuGraph C API algorithms must keep accepting Int32 vertices");
    }
}

#[test]
fn validate_local_parquet_source_uses_output_schema_override_for_available_columns() {
    let physical_schema = Arc::new(Schema::new(vec![
        Field::new("old_status", ArrowDataType::Int64, true),
        Field::new("status", ArrowDataType::Int64, true),
    ]));
    let logical_schema = Arc::new(Schema::new(vec![Field::new("status", ArrowDataType::Int64, true)]));
    let file = LocalParquetFile::new("/tmp/renamed.parquet");
    let source = LocalParquetSource::from_files(vec![file], physical_schema, vec!["old_status".to_string()])
        .with_column_roles(vec!["status".to_string()], Vec::new())
        .with_column_source_refs(vec![0], vec![SourceColumnRef::File(0)], Vec::new())
        .with_output_schema_override(Arc::clone(&logical_schema));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("status"), Field::new("status", ArrowDataType::Int64, true))],
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect("logical output alias should be available downstream");
}

#[test]
fn builder_computes_nested_regexp_replace_as_column_outputs_of_earlier_projections() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int64, false),
        Field::new("s", ArrowDataType::Utf8, true),
    ]));
    let strip_a = || Expr::regexp_replace(Expr::column("s"), "^a(.*)$", "\\1", ArrowDataType::Utf8);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", ArrowDataType::Int64, false),
        Field::new("lowered", ArrowDataType::Utf8, true),
        Field::new("picked", ArrowDataType::Utf8, true),
        Field::new("chained", ArrowDataType::Utf8, true),
        Field::new("direct", ArrowDataType::Utf8, true),
    ]));
    let outputs = vec![
        Expr::column("id"),
        Expr::lower(strip_a()),
        Expr::case(
            vec![CaseWhen::new(Expr::like(Expr::column("s"), "%a%"), strip_a())],
            Expr::literal(LiteralValue::Null(ArrowDataType::Utf8)),
        ),
        Expr::regexp_replace(
            Expr::regexp_replace(Expr::upper(Expr::column("s")), "A", "b", ArrowDataType::Utf8),
            "b",
            "c",
            ArrowDataType::Utf8,
        ),
        Expr::regexp_replace(Expr::column("s"), "x", "y", ArrowDataType::Utf8).alias("direct"),
    ];
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(input_schema)));
    let projection = builder.add_projection(source, projection_outputs(outputs, Arc::clone(&output_schema))).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect("staged REGEXP_REPLACE projections should validate");
    assert_eq!(plan.output_schema().as_ref(), output_schema.as_ref());
    let replacements = plan_projection_outputs(&plan).filter(|output| output.expr.contains_regexp_replace()).count();
    assert_eq!(replacements, 4, "LOWER and CASE share one replacement; the chain needs two; direct needs one");
}

#[test]
fn builder_computes_equal_regexp_replace_inputs_and_replacements_once() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("s", ArrowDataType::Utf8, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("first", ArrowDataType::Utf8, true),
        Field::new("second", ArrowDataType::Utf8, true),
        Field::new("direct", ArrowDataType::Utf8, true),
        Field::new("lowered", ArrowDataType::Utf8, true),
    ]));
    let strip_p = || Expr::regexp_replace(Expr::column("s"), "p", "r", ArrowDataType::Utf8);
    let outputs = vec![
        Expr::regexp_replace(Expr::upper(Expr::column("s")), "A", "b", ArrowDataType::Utf8),
        Expr::regexp_replace(Expr::upper(Expr::column("s")), "C", "d", ArrowDataType::Utf8),
        strip_p().alias("direct"),
        Expr::lower(strip_p()),
    ];
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(input_schema)));
    let projection = builder.add_projection(source, projection_outputs(outputs, Arc::clone(&output_schema))).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect("staged REGEXP_REPLACE projections should validate");
    assert_eq!(plan.output_schema().as_ref(), output_schema.as_ref());
    let count = |predicate: fn(&Expr) -> bool| {
        plan_projection_outputs(&plan).filter(|output| output.expr.any(predicate)).count()
    };
    assert_eq!(count(|expr| matches!(expr, Expr::Upper(_))), 1, "both replacements read one UPPER(s) column");
    assert_eq!(
        count(|expr| matches!(expr, Expr::RegexpReplace { .. })),
        3,
        "one replacement per distinct input and pattern; the direct output and LOWER share theirs"
    );
}

#[test]
fn validate_rejects_projection_regexp_replace_outside_a_whole_output_over_a_column() {
    let schema = Arc::new(Schema::new(vec![Field::new("s", ArrowDataType::Utf8, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let replacement = Expr::regexp_replace(Expr::column("s"), "^a(.*)$", "\\1", ArrowDataType::Utf8);
    let projection =
        builder.add_projection(source, projection_outputs(vec![replacement.clone()], Arc::clone(&schema))).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let mut nodes = plan.nodes().to_vec();
    let PlanNode::Projection(projection) = &mut nodes[projection.index()].kind else {
        panic!("the builder should return the projection it added");
    };
    projection.outputs[0].expr = Expr::lower(replacement);
    let plan = plan.with_nodes_for_test(nodes).unwrap();

    let error = validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect_err("the projection executor does not size REGEXP_REPLACE inside LOWER");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.fact_value("reason"), Some("projection_regexp_replace_requires_column_output"));
    assert_eq!(error.fact_value("expression_family"), Some("regexp_replace"));
}

#[test]
fn validate_rejects_a_decimal_division_whose_output_type_only_a_cast_declares() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", ArrowDataType::Decimal128(10, 2), false),
        Field::new("b", ArrowDataType::Decimal128(10, 2), false),
    ]));
    let decimal = ArrowDataType::Decimal128(5, 1);
    let quotient = Expr::cast(Expr::binary(Expr::column("a"), BinaryOp::Divide, Expr::column("b")), decimal.clone());
    let output_schema = Arc::new(Schema::new(vec![Field::new("r", decimal, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
    let projection = builder.add_projection(source, projection_outputs(vec![quotient], output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = validate_query_plan(&plan, rapids_analysis_context(PlanningPolicy::default()))
        .expect_err("the cast target is not the result type of the decimal division it casts");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.message(), "native decimal division requires a declared decimal output type");
}

fn plan_projection_outputs(plan: &QueryPlan) -> impl Iterator<Item = &ProjectionOutput> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Projection(projection) => Some(projection.outputs.iter()),
            _ => None,
        })
        .flatten()
}

fn left_deep_zip_range_disjunction(count: usize) -> Expr {
    let mut ranges = (0..count).map(zip_range_predicate);
    let first = ranges.next().expect("test predicate needs at least one range");
    ranges.fold(first, |left, right| Expr::binary(left, BinaryOp::Or, right))
}

fn zip_range_predicate(index: usize) -> Expr {
    let lower = format!("{:05}", 10000 + index);
    let upper = format!("{:05}", 10001 + index);
    let lower_bound = Expr::binary(Expr::column("ca_zip"), BinaryOp::GtEq, Expr::literal(LiteralValue::Utf8(lower)));
    let upper_bound = Expr::binary(Expr::column("ca_zip"), BinaryOp::Lt, Expr::literal(LiteralValue::Utf8(upper)));
    Expr::binary(lower_bound, BinaryOp::And, upper_bound)
}

fn graph_algorithm_validation_plan(
    algorithm: GraphAlgorithm,
    params: GraphAlgorithmParams,
    schema: Arc<Schema>,
) -> QueryPlan {
    let vertex_type = schema.field_with_name("src").unwrap().data_type().clone();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
    let output_schema = graph_result_schema_for_vertex_type(algorithm, &params, &vertex_type);
    let spec = GraphAlgorithmSpec::new(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        algorithm.definition().default_graph_options(&GraphConstructionOptions::default()),
        params,
        Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(source, Vec::new(), spec).unwrap();
    builder.finish_with_host_sink_schema(graph, output_schema).unwrap()
}
