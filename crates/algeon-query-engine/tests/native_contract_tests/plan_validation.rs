//! `validate_query_plan` contract tests that exercise real RAPIDS support
//! semantics (filter-predicate lowering, cuGraph vertex id types), bound to
//! the engine's [`RapidsBackend`](query_engine::rapids_backend::RapidsBackend)
//! through [`rapids_analysis_context`].

use arrow::datatypes::DataType as ArrowDataType;
use query_engine::{
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions, GraphEdgeColumnBindings,
        QueryPlan, graph_result_schema_for_vertex_type,
    },
    rapids_backend::rapids_analysis_context,
    validate_query_plan,
};

use super::*;

#[test]
fn validate_filter_uses_filter_lowering_for_large_left_deep_boolean_predicates() {
    let schema = Arc::new(Schema::new(vec![Field::new("ca_zip", ArrowDataType::Utf8, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let filter = builder.add_filter(source, left_deep_zip_range_disjunction(384), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(filter);

    validate_query_plan(&plan, rapids_analysis_context())
        .expect("large q08-class filter predicate should validate without generic lowering");
}

#[test]
fn validate_rejects_global_whole_partition_aggregate_window() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int32, false),
        Field::new("v", ArrowDataType::Int64, true),
    ]));
    let window_schema = Arc::new(Schema::new(vec![
        Field::new("k", ArrowDataType::Int32, false),
        Field::new("v", ArrowDataType::Int64, true),
        Field::new("window_v", ArrowDataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(input_schema)));
    let window = builder.add_window(
        source,
        WindowSpec::new(
            Vec::new(),
            Vec::new(),
            vec![WindowFunction::aggregate(AggregateFunction::Sum, Expr::column("v"), "window_v")],
        ),
        window_schema,
    );
    let plan = builder.finish_with_host_sink(window);

    let error = validate_query_plan(&plan, rapids_analysis_context())
        .expect_err("global whole-partition aggregate windows are unsupported");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedOperator);
    assert_eq!(error.operation(), "query_engine.window.validate");
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
    let graph = builder.add_graph_algorithm(source, spec);
    let plan = builder.finish_with_host_sink(graph);

    let error = validate_query_plan(&plan, rapids_analysis_context())
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
        let error = validate_query_plan(&int64_plan, rapids_analysis_context())
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
        validate_query_plan(&int32_plan, rapids_analysis_context())
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
    let file = LocalParquetFile::new("/tmp/renamed.parquet").with_field_id_read_bindings(vec![
        ParquetFieldIdReadBinding::new(2, "status", "old_status", 0, true, ArrowDataType::Int64),
    ]);
    let source = LocalParquetSource::from_files(vec![file], physical_schema, vec!["old_status".to_string()])
        .with_column_roles(vec!["status".to_string()], Vec::new())
        .with_column_source_refs(vec![0], vec![SourceColumnRef::File(0)], Vec::new())
        .with_output_schema_override(Arc::clone(&logical_schema));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        vec![ProjectionOutput::new(Expr::column("status"), Field::new("status", ArrowDataType::Int64, true))],
    );
    let plan = builder.finish_with_host_sink(projection);

    validate_query_plan(&plan, rapids_analysis_context()).expect("logical output alias should be available downstream");
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
        params.clone(),
        Arc::clone(&output_schema),
    );
    let graph = builder.add_graph_algorithm(source, spec);
    builder.finish_with_host_sink_schema(graph, output_schema)
}
