use std::sync::Arc;

use arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use super::*;
use crate::{
    admission::envelope::AttemptReservationProducerPhase,
    capability::{ContractQuality, ReservationRequirement},
    chunk::SourceChunkPlanningOptions,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, Expr, LimitSpec, LiteralValue, SortExpr,
        SortSpec,
    },
    graph_memory::{GraphMemoryCalibrationOverride, graph_memory_bounded_admission_evidence_for_plan},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphConstructionPolicy, GraphEdgeColumnBindings, GraphForceAtlas2Params, GraphPageRankParams, JoinKey,
        JoinOutputColumn, JoinSide, JoinSpec, JoinType, ProjectionOutput, QueryPlan, QueryPlanBuilder,
        graph_result_schema_for_params,
    },
    runtime::{ReservationKind, SourceChunkPlanningMode, SourceSchedulerPolicy},
    source::{GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

fn capability() -> ReservationRequirement {
    ReservationRequirement::modeled(
        ReservationKind::OperatorOutput,
        "test",
        "test_contract",
        ContractQuality::CapabilityProof,
    )
}

fn graph_bfs_plan() -> QueryPlan {
    graph_bfs_plan_with_source_vertex(0)
}

fn graph_bfs_plan_with_source_vertex(source_vertex: i64) -> QueryPlan {
    graph_bfs_plan_with_source_vertex_and_rows(source_vertex, 20)
}

fn graph_bfs_plan_with_source_vertex_and_rows(source_vertex: i64, row_count_upper_bound: u64) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("edges", schema).with_row_count_upper_bound(row_count_upper_bound),
    ));
    let graph = builder.add_graph_algorithm(
        edges,
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Bfs,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(source_vertex, None))),
        ),
    );
    builder.finish_with_host_sink(graph)
}

fn materializing_pagerank_plan_and_producer() -> (QueryPlan, QueryPlan) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let mut producer_builder = QueryPlanBuilder::new();
    let batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef, Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef],
    )
    .unwrap();
    let source = producer_builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&edge_schema), vec![batch])));
    let producer = producer_builder.finish_gpu_resident(source);

    let mut graph_builder = QueryPlanBuilder::new();
    let resident = graph_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("edges", Arc::clone(&edge_schema))
            .with_row_count_upper_bound(2)
            .with_graph_vertex_upper_bound(2),
    ));
    let projected = graph_builder.add_projection(
        resident,
        vec![
            ProjectionOutput::new(Expr::column("src"), edge_schema.field(0).as_ref().clone()),
            ProjectionOutput::new(Expr::column("dst"), edge_schema.field(1).as_ref().clone()),
        ],
    );
    let graph = graph_builder.add_graph_algorithm(
        projected,
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<String>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        ),
    );
    (graph_builder.finish_with_host_sink(graph), producer)
}

fn filtered_pagerank_plan_and_producer(with_limit: bool) -> (QueryPlan, QueryPlan) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let mut producer_builder = QueryPlanBuilder::new();
    let source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/deferred-filtered-edges.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10_000])
                .with_row_group_bytes(vec![160_000])
                .with_projected_row_group_bytes(vec![160_000], vec![160_000]),
        ],
        Arc::clone(&edge_schema),
        vec!["src".to_owned(), "dst".to_owned()],
    )));
    let filtered = producer_builder.add_filter(
        source,
        Expr::binary(Expr::column("src"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
        Arc::clone(&edge_schema),
    );
    let producer_root = if with_limit {
        producer_builder.add_limit(filtered, LimitSpec::new(0, Some(100)), Arc::clone(&edge_schema))
    } else {
        filtered
    };
    let producer = producer_builder.finish_gpu_resident(producer_root);

    let mut graph_builder = QueryPlanBuilder::new();
    let resident = graph_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("edges", Arc::clone(&edge_schema)).with_row_count_upper_bound(10_000),
    ));
    let graph = graph_builder.add_graph_algorithm(
        resident,
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<String>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        ),
    );
    (graph_builder.finish_with_host_sink(graph), producer)
}

fn scheduled_materializing_pagerank_proof() -> QueryPlanCapabilityProof {
    let (graph_plan, producer_plan) = materializing_pagerank_plan_and_producer();
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("materializing PageRank must produce graph evidence");
    let graph = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan_with_resident_inputs(
        &graph_plan,
        [("edges", &producer_plan)],
    )
    .expect("resident producer evidence must bind")
    .expect("PageRank must issue a graph proof")
    .with_graph_memory_admission_evidence(&evidence)
    .expect("graph evidence must attach");
    QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
        &graph_plan,
        graph,
        "edges",
        &producer_plan,
        std::iter::empty(),
    )
    .expect("materializing input schedule must resolve")
    .expect("projection between source and graph requires a materializing schedule")
}

#[test]
fn filtered_edge_producer_defers_only_the_graph_phase_under_one_attempt_proof() {
    let (graph_plan, producer_plan) = filtered_pagerank_plan_and_producer(false);
    let graph = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("PageRank must issue its attested execution proof");
    let combined = QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
        &graph_plan,
        graph,
        "edges",
        &producer_plan,
        std::iter::empty(),
    )
    .expect("the deferred contract must be derivable")
    .expect("Source-Filter-GPU sink is the supported deferred topology");
    let descriptor = combined
        .as_requirement()
        .deferred_graph_phase_for_plan(&graph_plan)
        .expect("descriptor lookup must remain unambiguous")
        .expect("the exact graph scope retains its deferred descriptor");
    assert_eq!(descriptor.input_name(), "edges");
    assert_eq!(descriptor.input_schema(), producer_plan.output_schema());
    assert_eq!(descriptor.static_edge_row_upper_bound(), 10_000);
    assert_eq!(
        descriptor.producer_execution_scope_fingerprint(),
        canonical_execution_scope_fingerprint(&producer_plan)
    );
    assert_eq!(combined.soft_working_set_hint_bytes(), None);
    let envelope = combined.resolve_reservation_envelope(reservation_policy()).expect("producer phase must resolve");
    assert!(envelope.complete_phase_peak_bytes().is_some());

    let mut consumer_builder = QueryPlanBuilder::new();
    let graph_result = consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "graph_result",
        Arc::clone(graph_plan.output_schema()),
    )));
    let consumer_plan = consumer_builder.finish_with_host_sink(graph_result);
    let consumer = QueryPlanCapabilityProof::for_attested_graph_result_passthrough_plan(&consumer_plan, "graph_result")
        .expect("the exact deferred graph result handoff must issue a passthrough proof");
    let composed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [ComposedGraphProducerCandidate::new(
            Some("graph_result".to_owned()),
            Some("graph_result".to_owned()),
            Some(combined),
        )],
    )
    .expect("the deferred graph phase must remain composable with its exact host passthrough");
    let envelope = composed.resolve_reservation_envelope(reservation_policy()).expect("composed proof must resolve");
    let grant_bytes = envelope.complete_phase_peak_bytes().expect("the producer phase remains bounded");
    envelope
        .validate_composed_execution(&consumer_plan, &["graph_result".to_owned()], grant_bytes)
        .expect("the exact runtime occurrence must match the deferred passthrough");
    let mismatch = envelope
        .validate_composed_execution(&consumer_plan, &["different_result".to_owned()], grant_bytes)
        .expect_err("a different runtime occurrence must remain fail closed");
    assert_eq!(mismatch.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
}

#[test]
fn exact_limit_edge_producer_stays_on_the_static_graph_admission_path() {
    let (graph_plan, producer_plan) = filtered_pagerank_plan_and_producer(true);
    let graph = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("PageRank must issue its attested execution proof");
    assert!(
        QueryPlanCapabilityProof::try_combine_cugraph_materializing_input_attempt(
            &graph_plan,
            graph,
            "edges",
            &producer_plan,
            std::iter::empty(),
        )
        .expect("an unsupported deferred topology is a normal shape miss")
        .is_none()
    );
}

fn ordinary_native_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_with_host_sink(source)
}

#[test]
fn grouped_aggregate_owner_sums_each_selected_row_group_bound() {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value_count", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/aggregate-owner-row-groups.parquet")
                .with_row_groups(vec![0, 2])
                .with_row_group_rows(vec![1, 100, 3]),
        ],
        source_schema,
        vec!["key".to_owned(), "value".to_owned()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("key")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "value_count")],
        ),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let owner_terms = generic_aggregate_owner_terms(&plan).expect("aggregate owners must be derivable");
    let partial_bytes = owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::AggregatePartialState)
        .expect("aggregate partial owner must be present")
        .bytes();
    let mut partial_columns = output_schema
        .fields()
        .iter()
        .map(|field| fixed_width_consumer_output_column(field.data_type()).unwrap())
        .collect::<Vec<_>>();
    partial_columns.push(fixed_width_consumer_output_column(&DataType::UInt64).unwrap());
    let expected = [1, 3]
        .into_iter()
        .map(|input_rows| {
            crate::exec::partial_aggregate_upper_bound_bytes(
                crate::exec::PartialAggregateShape::Grouped { input_rows },
                &partial_columns,
            )
            .unwrap()
        })
        .sum::<u64>();
    let whole_input_per_group = crate::exec::partial_aggregate_upper_bound_bytes(
        crate::exec::PartialAggregateShape::Grouped { input_rows: 4 },
        &partial_columns,
    )
    .unwrap()
        * 2;

    assert_eq!(partial_bytes, expected);
    assert!(partial_bytes < whole_input_per_group);
}

#[test]
fn non_sort_requirement_does_not_validate_candidate_device() {
    let plan = ordinary_native_plan();
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("ordinary plan must issue a proof");
    let resolved = proof
        .requirement_for_device(usize::MAX)
        .expect("non-sort requirements do not need a CUDA device for resolution");

    assert_eq!(resolved.as_requirement(), proof.as_requirement());
    assert!(resolved.as_requirement().reservation_shapes.iter().all(|shape| shape.sort_metadata().is_none()));
}

fn graph_bfs_proof_with_typed_output(source_vertex: i64) -> QueryPlanCapabilityProof {
    let plan = graph_bfs_plan_with_source_vertex(source_vertex);
    graph_bfs_proof_with_typed_output_for_plan(plan)
}

fn graph_bfs_proof_with_typed_output_and_rows(
    source_vertex: i64,
    row_count_upper_bound: u64,
) -> QueryPlanCapabilityProof {
    let plan = graph_bfs_plan_with_source_vertex_and_rows(source_vertex, row_count_upper_bound);
    graph_bfs_proof_with_typed_output_for_plan(plan)
}

fn graph_bfs_proof_with_typed_output_for_plan(plan: QueryPlan) -> QueryPlanCapabilityProof {
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("test graph plan must produce typed admission evidence");
    QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("test graph plan must issue a capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence")
}

fn production_graph_proof(algorithm: GraphAlgorithm) -> QueryPlanCapabilityProof {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let params = match algorithm {
        GraphAlgorithm::PageRank => GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        GraphAlgorithm::ForceAtlas2 => GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
        _ => panic!("the production proof fixture only covers PageRank and ForceAtlas2"),
    };
    let mut builder = QueryPlanBuilder::new();
    let edge = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("graph_edges", Arc::clone(&edge_schema))
            .with_row_count_upper_bound(8)
            .with_graph_vertex_upper_bound(2),
    ));
    let graph = builder.add_graph_algorithm(
        edge,
        GraphAlgorithmSpec::with_stable_schema(
            algorithm,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            params,
        ),
    );
    let plan = builder.finish_gpu_resident(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("the production graph IR must produce finite memory evidence");
    QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("the production graph IR must issue a capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("the production graph proof must retain finite graph evidence")
}

fn pagerank_forceatlas2_join_consumer() -> QueryPlan {
    let pagerank_params = GraphAlgorithmParams::PageRank(GraphPageRankParams::default());
    let forceatlas2_params = GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default());
    let pagerank_schema = graph_result_schema_for_params(GraphAlgorithm::PageRank, &pagerank_params);
    let forceatlas2_schema = graph_result_schema_for_params(GraphAlgorithm::ForceAtlas2, &forceatlas2_params);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("pagerank_value", DataType::Float64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let pagerank =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("pagerank", Arc::clone(&pagerank_schema))));
    let pagerank_filter = builder.add_filter(
        pagerank,
        Expr::binary(Expr::column("value"), BinaryOp::Gt, Expr::literal(crate::expr::LiteralValue::Float64(0.0))),
        Arc::clone(&pagerank_schema),
    );
    let pagerank_projected = builder.add_projection(
        pagerank_filter,
        vec![
            ProjectionOutput::new(Expr::column("vertex"), Field::new("vertex", DataType::Int64, false)),
            ProjectionOutput::new(Expr::column("value"), Field::new("pagerank_value", DataType::Float64, false)),
        ],
    );
    let forceatlas2 = builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("forceatlas2", Arc::clone(&forceatlas2_schema))));
    let join = builder.add_join(
        pagerank_projected,
        forceatlas2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                crate::expr::ColumnRef::input(0, "vertex", "vertex"),
                crate::expr::ColumnRef::input(1, "vertex", "vertex"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "vertex"),
            JoinOutputColumn::left(
                crate::expr::ColumnRef::input(0, "pagerank_value", "pagerank_value"),
                "pagerank_value",
            ),
            JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "x", "x"), "x"),
            JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "y", "y"), "y"),
        ]),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}

fn relational_gpu_producer_proof() -> QueryPlanCapabilityProof {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/requirement-relational-producer.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![8])
                .with_row_group_bytes(vec![128])
                .with_projected_row_group_bytes(vec![64], vec![128]),
        ],
        schema,
        vec!["value".to_owned()],
    )));
    let plan = builder.finish_gpu_resident(source);
    QueryPlanCapabilityProof::for_query_plan(&plan).expect("bounded relational GPU producer must issue a proof")
}

fn graph_output_union_consumer(input_names: &[&str]) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let inputs = input_names
        .iter()
        .map(|name| builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(*name, Arc::clone(&schema)))))
        .collect();
    let root = builder.add_union(inputs, schema);
    builder.finish_with_host_sink(root)
}

fn mixed_local_graph_union_consumer(source: LocalParquetSource) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(source));
    let graph = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("graph", Arc::clone(&schema))));
    let root = builder.add_union(vec![local, graph], schema);
    builder.finish_with_host_sink(root)
}

fn mixed_local_graph_union_aggregate_consumer(source: LocalParquetSource) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("value_count", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(source));
    let graph = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("graph", Arc::clone(&input_schema))));
    let union = builder.add_union(vec![local, graph], Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(
        union,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "value_count")],
        ),
        Arc::clone(&output_schema),
    );
    builder.finish_with_host_sink(aggregate)
}

fn mixed_local_graph_union_source(row_count: Option<u64>, bytes: Option<u64>) -> LocalParquetSource {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut file = LocalParquetFile::new("/tmp/requirement-mixed-composition.parquet");
    if let Some(row_count) = row_count {
        file = file.with_row_group_rows(vec![row_count]);
    }
    if let Some(bytes) = bytes {
        file = file.with_projected_row_group_bytes(vec![bytes / 2], vec![bytes]);
    }
    LocalParquetSource::from_files(vec![file], schema, vec!["value".to_owned()])
}

fn graph_output_projection_consumer(expr: Expr, output_type: DataType) -> QueryPlan {
    graph_output_projection_consumer_with_input(DataType::Int64, expr, output_type)
}

fn graph_output_projection_consumer_with_input(input_type: DataType, expr: Expr, output_type: DataType) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", input_type, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("only", input_schema)));
    let projection =
        builder.add_projection(source, vec![ProjectionOutput::new(expr, Field::new("projected", output_type, false))]);
    builder.finish_with_host_sink(projection)
}

fn graph_output_case_projection_consumer() -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("score", DataType::Float64, false),
        Field::new("fallback", DataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("only", input_schema)));
    let case = Expr::case(
        vec![CaseWhen::new(
            Expr::binary(Expr::column("score"), BinaryOp::Eq, Expr::column("fallback")),
            Expr::column("score"),
        )],
        Expr::literal(LiteralValue::Null(DataType::Float64)),
    );
    let projection = builder
        .add_projection(source, vec![ProjectionOutput::new(case, Field::new("derived_score", DataType::Float64, true))]);
    builder.finish_with_host_sink(projection)
}

fn graph_output_global_aggregate_consumer() -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("only", input_schema)));
    let aggregate_input_schema = Arc::new(Schema::new(vec![
        Field::new("__nexus_native_aggregate_arg_0", DataType::Int64, false),
        Field::new("__nexus_native_aggregate_arg_1", DataType::Float64, false),
    ]));
    let aggregate_input = builder.add_projection(
        source,
        vec![
            ProjectionOutput::new(
                Expr::literal(crate::expr::LiteralValue::Int64(1)).alias("__nexus_native_aggregate_arg_0"),
                aggregate_input_schema.field(0).as_ref().clone(),
            ),
            ProjectionOutput::new(
                Expr::column("value").alias("__nexus_native_aggregate_arg_1"),
                aggregate_input_schema.field(1).as_ref().clone(),
            ),
        ],
    );
    let aggregate = builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::column("__nexus_native_aggregate_arg_0"),
                    "vertices",
                ),
                AggregateExpr::new(AggregateFunction::Sum, Expr::column("__nexus_native_aggregate_arg_1"), "total"),
            ],
        ),
        Arc::new(Schema::new(vec![
            Field::new("vertices", DataType::Int64, false),
            Field::new("total", DataType::Float64, true),
        ])),
    );
    let projection = builder.add_projection(
        aggregate,
        vec![
            ProjectionOutput::new(Expr::column("vertices"), Field::new("vertices", DataType::Int64, false)),
            ProjectionOutput::new(Expr::column("total"), Field::new("total", DataType::Float64, true)),
        ],
    );
    builder.finish_with_host_sink(projection)
}

fn graph_output_global_nunique_consumer(input_type: DataType, nullable: bool, grouped: bool) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![Field::new("label", input_type.clone(), nullable)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("only", Arc::clone(&input_schema)).with_row_count_upper_bound(128),
    ));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            grouped.then(|| Expr::column("label")).into_iter().collect(),
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "rows"),
                AggregateExpr::new(AggregateFunction::NUnique, Expr::column("label"), "components"),
            ],
        ),
        if grouped {
            Arc::new(Schema::new(vec![
                Field::new("label", input_type, nullable),
                Field::new("rows", DataType::Int64, false),
                Field::new("components", DataType::Int64, false),
            ]))
        } else {
            Arc::new(Schema::new(vec![
                Field::new("rows", DataType::Int64, false),
                Field::new("components", DataType::Int64, false),
            ]))
        },
    );
    builder.finish_with_host_sink(aggregate)
}

fn graph_output_filter_consumer(input_name: &str) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, Arc::clone(&schema))));
    let filter = builder.add_filter(source, Expr::column("value"), schema);
    builder.finish_with_host_sink(filter)
}

fn graph_output_inner_join_consumer(key_type: DataType, key_nullable: bool, include_unary_branch: bool) -> QueryPlan {
    graph_output_inner_join_consumer_with_contract(key_type, key_nullable, include_unary_branch, false)
}

fn graph_output_inner_join_consumer_with_contract(
    key_type: DataType,
    key_nullable: bool,
    include_unary_branch: bool,
    unique_lookup: bool,
) -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", key_type.clone(), key_nullable),
        Field::new("value", DataType::Float64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_vertex", key_type.clone(), key_nullable),
        Field::new("left_value", DataType::Float64, false),
        Field::new("right_value", DataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
    let projected_left = if include_unary_branch {
        let filtered_left = builder.add_filter(left, Expr::column("vertex"), Arc::clone(&input_schema));
        builder.add_projection(
            filtered_left,
            vec![
                ProjectionOutput::new(Expr::column("vertex"), input_schema.field(0).as_ref().clone()),
                ProjectionOutput::new(Expr::column("value"), input_schema.field(1).as_ref().clone()),
            ],
        )
    } else {
        left
    };
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
    let spec = JoinSpec::new(
        JoinType::Inner,
        vec![JoinKey::new(
            crate::expr::ColumnRef::input(0, "vertex", "vertex"),
            crate::expr::ColumnRef::input(1, "vertex", "vertex"),
        )],
    )
    .with_build_side(JoinSide::Right)
    .with_output_projection(vec![
        JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "left_vertex"),
        JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "value", "value"), "left_value"),
        JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "value", "value"), "right_value"),
    ]);
    let spec = if unique_lookup { spec.with_at_most_one_match_per_bounded_row(JoinSide::Left) } else { spec };
    let join = builder.add_join(projected_left, right, spec, output_schema);
    builder.finish_with_host_sink(join)
}

fn graph_output_inner_join_sort_limit_consumer() -> QueryPlan {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("left_vertex", DataType::Int64, false),
        Field::new("left_value", DataType::Float64, false),
        Field::new("right_value", DataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&input_schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&input_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                crate::expr::ColumnRef::input(0, "vertex", "vertex"),
                crate::expr::ColumnRef::input(1, "vertex", "vertex"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "left_vertex"),
            JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "value", "value"), "left_value"),
            JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "value", "value"), "right_value"),
        ]),
        Arc::clone(&join_schema),
    );
    let sorted = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("left_value"), false, false)]),
        Arc::clone(&join_schema),
    );
    let limited = builder.add_limit(sorted, LimitSpec::new(0, Some(2)), join_schema);
    builder.finish_with_host_sink(limited)
}

fn reservation_policy() -> AttemptReservationPolicy {
    AttemptReservationPolicy::new(
        64 * 1024,
        1,
        3,
        1,
        SourceSchedulerPolicy::new(2),
        SourceChunkPlanningMode::default(),
        SourceChunkPlanningOptions::default(),
    )
}

#[test]
fn known_local_parquet_sources_keep_a_separate_modeled_source_stage() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/requirement-known.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![100])
                .with_projected_row_group_bytes(vec![40], vec![96]),
        ],
        schema,
        vec!["value".to_owned()],
    )));
    let plan = builder.finish_with_host_sink(source);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    assert_eq!(requirement.source_stage_bytes, Some(96));
    assert_eq!(requirement.soft_working_set_hint_bytes(), None);
    assert_eq!(requirement.estimate_quality(), EstimateQuality::Modeled);
}

#[test]
fn unknown_local_parquet_sources_reject_before_envelope_resolution() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/requirement-unknown.parquet",
        schema,
        vec!["value".to_owned()],
    )));
    let plan = builder.finish_with_host_sink(source);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    assert_eq!(requirement.source_stage_bytes, None);
    assert_eq!(requirement.soft_working_set_hint_bytes(), None);
    assert_eq!(requirement.estimate_quality(), EstimateQuality::Unknown);
    let proof = QueryPlanCapabilityProof { requirement };
    let error = proof
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("unknown local source must fail before placement");
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_local_source_bytes_unavailable"));
}

#[test]
fn mixed_local_and_gpu_sources_retain_local_owner_terms() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/requirement-mixed.parquet").with_file_size_bytes(96)],
        Arc::clone(&schema),
        vec!["value".to_owned()],
    )));
    let gpu = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("gpu", Arc::clone(&schema))));
    let root = builder.add_union(vec![local, gpu], schema);
    let plan = builder.finish_with_host_sink(root);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    assert_eq!(requirement.owner_terms.len(), 1);
    assert!(requirement.owner_terms[0].is_source_stage());
    assert_eq!(requirement.owner_terms[0].bytes(), 96);
    QueryPlanCapabilityProof { requirement }
        .resolve_reservation_envelope(reservation_policy())
        .expect("external GPU inputs do not need an engine-owned byte term");
}

#[test]
fn empty_literal_batch_does_not_invalidate_local_source_owner_terms() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/requirement-empty-literal.parquet").with_file_size_bytes(96)],
        Arc::clone(&schema),
        vec!["value".to_owned()],
    )));
    let literal = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&schema))));
    let root = builder.add_union(vec![local, literal], schema);
    let plan = builder.finish_with_host_sink(root);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    assert_eq!(requirement.owner_terms.len(), 1);
    assert!(requirement.owner_terms[0].is_source_stage());
    assert_eq!(requirement.owner_terms[0].bytes(), 96);
    QueryPlanCapabilityProof { requirement }
        .resolve_reservation_envelope(reservation_policy())
        .expect("an empty literal batch does not add an unowned source allocation");
}

#[test]
fn computed_gpu_sink_keeps_source_and_retained_output_owners_distinct() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/requirement-computed-gpu-sink.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![3])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![64]),
        ],
        Arc::clone(&schema),
        vec!["value".to_owned()],
    )));
    let gpu = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("gpu", Arc::clone(&schema)).with_row_count_upper_bound(2),
    ));
    let root = builder.add_union(vec![local, gpu], Arc::clone(&schema));
    let plan = builder.finish_gpu_resident(root);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    let source_terms = requirement.owner_terms.iter().filter(|term| term.is_source_stage()).collect::<Vec<_>>();
    let retained_terms = requirement.owner_terms.iter().filter(|term| term.is_retained_output()).collect::<Vec<_>>();
    assert_eq!(source_terms.len(), 2);
    assert_ne!(source_terms[0].id(), source_terms[1].id());
    assert_eq!(retained_terms.len(), 1);
    assert_eq!(retained_terms[0].retained_row_count_upper_bound(), Some(5));
    let mut source_bytes = source_terms.iter().map(|term| term.bytes()).collect::<Vec<_>>();
    source_bytes.sort_unstable();
    let mut expected_source_bytes = vec![64, bounded_consumer_frame_bytes(2, &schema).unwrap()];
    expected_source_bytes.sort_unstable();
    assert_eq!(source_bytes, expected_source_bytes);
    assert_eq!(retained_terms[0].bytes(), bounded_consumer_frame_bytes(5, &schema).unwrap());
    QueryPlanCapabilityProof { requirement }
        .resolve_reservation_envelope(reservation_policy())
        .expect("computed GPU sink must have a complete owner inventory");
}

#[test]
fn composed_union_reuses_exact_local_source_stage_alongside_gpu_producer() {
    let consumer_plan = mixed_local_graph_union_consumer(mixed_local_graph_union_source(Some(3), Some(64)));
    let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
        .expect("composition policy issues the mixed local/GPU consumer proof");
    let direct_error = consumer
        .clone()
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("a mixed consumer cannot resolve before a retained producer is bound");
    assert_eq!(direct_error.fact_value("schedule_reason"), Some("consumer_producer_binding_required"));
    let aggregated = AttemptCapabilityEvidence::aggregate([consumer.clone().into_requirement()]).unwrap();
    let aggregate_error = QueryPlanCapabilityProof { requirement: aggregated }
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("generic aggregation must preserve the incomplete composition marker");
    assert_eq!(aggregate_error.fact_value("schedule_reason"), Some("consumer_producer_binding_required"));
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect("the exact local branch and one GPU producer must compose");

    let source_terms = combined
        .as_requirement()
        .owner_terms
        .iter()
        .filter(|term| term.id.role == OwnerTermRole::SourceStage)
        .collect::<Vec<_>>();
    assert_eq!(source_terms.len(), 1, "composition must reuse one canonical SourceStage owner");
    assert_eq!(source_terms[0].bytes(), 64);
    let consumer_shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("consumer scope is retained");
    assert_eq!(consumer_shape.producer_occurrences().len(), 1);
    assert_eq!(consumer_shape.producer_occurrences()[0].occurrence_id(), "graph");
    let groups = consumer_shape
        .ordered_schedule()
        .expect("the mixed union must retain an ordered consumer schedule")
        .consumer_peak_term_groups();
    assert_eq!(groups.len(), 2, "union branches and final output are sequential peak alternatives");
    assert!(groups[0].iter().any(|term_id| term_id.role == OwnerTermRole::SourceStage));
    assert!(groups[1].iter().any(|term_id| term_id.role == OwnerTermRole::ConsumerOutput));
    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("mixed local/GPU composition has one finite owner inventory");
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_union_then_aggregate_preserves_each_sequential_peak_group() {
    let consumer_plan = mixed_local_graph_union_aggregate_consumer(mixed_local_graph_union_source(Some(3), Some(64)));
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("the mixed union aggregate consumer must issue a composed proof"),
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect("the sequential union and aggregate must compose");
    let shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("the consumer scope is retained")
        .ordered_schedule()
        .expect("the nested consumer must retain an ordered schedule");
    let groups = shape.consumer_peak_term_groups();
    assert_eq!(groups.len(), 3, "union branch, union output, and aggregate are sequential alternatives");
    assert!(groups[0].iter().any(|term_id| term_id.role == OwnerTermRole::SourceStage));
    assert!(groups[1].iter().any(|term_id| term_id.role == OwnerTermRole::ConsumerOutput));
    assert!(groups[2].iter().any(|term_id| term_id.role == OwnerTermRole::AggregatePartialState));
    combined.resolve_reservation_envelope(reservation_policy()).expect("the exact nested schedule must resolve");
}

#[test]
fn composed_join_sort_limit_preserves_operator_stage_alternatives() {
    let consumer_plan = graph_output_inner_join_sort_limit_consumer();
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("the join sort-limit consumer must issue a composed proof"),
        [
            ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .expect("the nested join sort-limit must compose");
    let shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("the consumer scope is retained")
        .ordered_schedule()
        .expect("the nested consumer must retain an ordered schedule");
    let groups = shape.consumer_peak_term_groups();
    assert_eq!(groups.len(), 3, "join, sort, and limit stages must remain sequential alternatives");
    assert!(groups[0].iter().any(|term_id| term_id.role == OwnerTermRole::ConsumerHashTable));
    assert!(groups[1].iter().any(|term_id| term_id.role == OwnerTermRole::ConsumerOutput));
    assert!(groups[2].iter().any(|term_id| term_id.role == OwnerTermRole::ConsumerOutput));
    combined.resolve_reservation_envelope(reservation_policy()).expect("the exact nested schedule must resolve");
}

#[test]
fn composed_join_accepts_duplicate_producers_with_materializing_input_schedules() {
    let consumer_plan = graph_output_inner_join_sort_limit_consumer();
    let producer = scheduled_materializing_pagerank_proof();
    let combined = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("the outer join consumer must issue a composed proof"),
        [
            ComposedGraphProducerCandidate::new(
                Some("left".to_owned()),
                Some("left".to_owned()),
                Some(producer.clone()),
            ),
            ComposedGraphProducerCandidate::new(Some("right".to_owned()), Some("right".to_owned()), Some(producer)),
        ],
    )
    .expect("outer composition must preserve and namespace both nested producer schedules");

    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("nested producer schedules must retain one complete owner inventory");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
}

#[test]
fn composed_filter_sort_preserves_alternative_filter_peak_group() {
    let consumer_plan = graph_result_filter_sort_plan(None);
    let producer = graph_bfs_proof_with_typed_output(0);
    let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &consumer_plan,
        "graph_result",
        &producer,
    )
    .unwrap();
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [("graph_result".to_owned(), producer)],
    )
    .unwrap();
    let resolved = combined
        .requirement_for_device_with_preflight(0, |_, _| Ok(SortRadixPreflight::new_for_test(64, 128, 256, 128)))
        .unwrap();
    let schedule = resolved
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .and_then(AttemptReservationShape::ordered_schedule)
        .expect("ordered schedule is attached");
    assert!(schedule.consumer_peak_term_groups().len() >= 2);
    assert!(
        !schedule
            .consumer_dominated_term_ids()
            .iter()
            .any(|term_id| term_id.role_label() == OwnerTermRole::ConsumerOutput.as_str())
    );
    assert!(
        !schedule
            .consumer_dominated_term_ids()
            .iter()
            .any(|term_id| term_id.role_label() == OwnerTermRole::ConsumerPredicateMask.as_str())
    );
    resolved
        .resolve_reservation_envelope(reservation_policy())
        .expect("dominated consumer owners remain part of a complete ordered inventory");
}

#[test]
fn composed_gpu_sink_defers_generic_output_owner_until_producer_binding() {
    let consumer_plan = mixed_local_graph_union_consumer(mixed_local_graph_union_source(Some(3), Some(64)))
        .clone_with_gpu_resident_sink();
    let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
        .expect("the exact GPU-sink consumer must issue a composition proof");
    let [shape] = consumer.as_requirement().reservation_shapes.as_slice() else {
        panic!("the composed consumer must retain one exact reservation scope");
    };
    assert_eq!(
        shape.owner_term_rejection_reason(),
        None,
        "the resolver owns the GPU-sink output alias after producer binding"
    );
    assert!(
        consumer.as_requirement().owner_terms.iter().any(|term| term.id.role == OwnerTermRole::SourceStage),
        "the mixed consumer must retain its local source-stage owner"
    );

    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect("exact GPU-sink composed consumer must resolve after producer binding");
    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("the rebound GPU-sink composition must resolve a finite envelope");
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_union_rejects_unknown_local_rows_or_bytes_before_placement() {
    let unknown_rows = mixed_local_graph_union_consumer(mixed_local_graph_union_source(None, Some(64)));
    let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&unknown_rows).unwrap();
    let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &unknown_rows,
        consumer,
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect_err("missing local footer rows cannot be turned into a consumer bound");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_local_source_rows_unavailable"));

    let unknown_bytes = mixed_local_graph_union_consumer(mixed_local_graph_union_source(Some(3), None));
    let consumer = QueryPlanCapabilityProof::for_composed_consumer_plan(&unknown_bytes).unwrap();
    let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &unknown_bytes,
        consumer,
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect_err("missing local source bytes cannot be turned into a consumer bound");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_local_source_owner_missing"));
}

#[test]
fn composed_consumer_rejects_literal_source_mixed_with_gpu_producer() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let local = builder.add_source(SourceNode::LocalParquet(mixed_local_graph_union_source(Some(3), Some(64))));
    let literal_batch = arrow::record_batch::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow::array::Int64Array::from(vec![1]))],
    )
    .unwrap();
    let literal = builder.add_source(SourceNode::LiteralBatch(crate::source::LiteralBatchSource::new(
        Arc::clone(&schema),
        vec![literal_batch],
    )));
    let root = builder.add_union(vec![local, literal], schema);
    let plan = builder.finish_with_host_sink(root);
    let proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&plan).unwrap();
    let error = proof.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_literal_source_bytes_unavailable"));
}

#[test]
fn saturated_local_source_rejects_before_owner_term_insertion() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/requirement-saturated.parquet").with_file_size_bytes(u64::MAX)],
        schema,
        vec!["value".to_owned()],
    )));
    let plan = builder.finish_with_host_sink(source);

    let requirement = AttemptCapabilityEvidence::for_query_plan(&plan);
    assert!(requirement.owner_terms.is_empty());
    let error = QueryPlanCapabilityProof { requirement }
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("saturated local source must fail before placement");
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_saturated"));
}

#[test]
fn plan_issued_source_terms_reject_aggregate_source_stage_overflow() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let plan_for_bytes = |path: &str, bytes| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new(path).with_file_size_bytes(bytes)],
            Arc::clone(&schema),
            vec!["value".to_owned()],
        )));
        builder.finish_with_host_sink(source)
    };
    let first = QueryPlanCapabilityProof::for_query_plan(&plan_for_bytes(
        "/tmp/requirement-overflow-first.parquet",
        u64::MAX - 1,
    ))
    .expect("finite source must issue a proof");
    let second =
        QueryPlanCapabilityProof::for_query_plan(&plan_for_bytes("/tmp/requirement-overflow-second.parquet", 2))
            .expect("finite source must issue a proof");
    let error =
        QueryPlanCapabilityProof::combine(first, [second]).expect_err("aggregate source-stage overflow must reject");
    assert_eq!(error.fact_value("reason"), Some("attempt_requirement_source_stage_overflow"));
}

#[test]
fn zero_column_count_source_resolves_a_parquet_read_floor_without_a_partial_target() {
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/requirement-count.parquet")
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![4_096])
            .with_projected_row_group_bytes(vec![0], vec![0]),
    ]);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let minimum_read_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("sink plan must issue a proof");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    let requirement = proof.into_requirement().with_reservation_envelope(&envelope);

    assert_eq!(requirement.source_stage_bytes, Some(0));
    assert_eq!(envelope.required_phase_floor_bytes(), minimum_read_budget);
    assert_eq!(envelope.complete_phase_peak_bytes(), None);
    assert_eq!(requirement.soft_working_set_hint_bytes(), None);
    assert_eq!(requirement.required_floor_bytes_for(1).unwrap(), minimum_read_budget);
}

#[test]
fn graph_evidence_does_not_double_count_its_prepared_local_source_frame() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { policy: GraphConstructionPolicy::RawLibcugraph, ..Default::default() },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/retained-graph-source.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&schema),
        vec!["src".to_owned(), "dst".to_owned()],
    )));
    let graph = builder.add_graph_algorithm(source, spec.clone());
    let plan = builder.finish_with_host_sink(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("graph plan must issue an engine-owned capability proof");

    assert_eq!(proof.soft_working_set_hint_bytes(), None);
    assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
    let mut ambiguous = proof.clone();
    let source_term = ambiguous
        .requirement
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::SourceStage)
        .expect("shared graph proof has one source-stage candidate")
        .rebind_composed_for_test("duplicate-source")
        .expect("test occurrence rebind keeps the source owner identity");
    ambiguous.requirement.owner_terms.push(source_term);
    let error = ambiguous
        .with_graph_memory_admission_evidence(&evidence)
        .expect_err("shared graph attachment cannot choose between two source owners");
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_graph_input_source_stage_ambiguous"));
    let proof = proof.with_graph_memory_admission_evidence(&evidence).unwrap();
    let requirement = proof.as_requirement();
    assert_eq!(
        requirement.owner_terms.iter().filter(|term| term.id().role == OwnerTermRole::SourceStage).count(),
        0,
        "shared prepared edge frame has one graph owner, not a second source-stage owner",
    );
    let expected_peak =
        evidence.bounded_live_graph_peak_with_source_stage_bytes(96).expect("shared graph live reservation fits");
    assert_eq!(requirement.soft_working_set_hint_bytes(), Some(expected_peak));
    let owner_sum = requirement.conservative_owner_sum().expect("shared graph owner inventory is finite");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).expect("shared graph envelope resolves");
    assert_eq!(owner_sum, envelope.complete_phase_peak_bytes().expect("shared graph peak is complete"));
    let expected_owner_sum = requirement
        .owner_terms
        .iter()
        .try_fold(0_u64, |total, term| total.checked_add(term.bytes()))
        .expect("shared graph owner terms fit");
    assert_eq!(owner_sum, expected_owner_sum, "shared source frame is charged by the graph owner exactly once");
}

#[test]
fn graph_evidence_adds_a_source_stage_that_normalizes_into_a_separate_graph_input() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    )
    .with_edge_normalize_inserted(true);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/normalized-graph-source.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&schema),
        vec!["src".to_owned(), "dst".to_owned()],
    )));
    let graph = builder.add_graph_algorithm(source, spec.clone());
    let plan = builder.finish_with_host_sink(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("graph plan must issue an engine-owned capability proof");

    assert_eq!(proof.soft_working_set_hint_bytes(), None);
    assert_eq!(proof.as_requirement().source_stage_bytes, Some(96));
    let proof = proof.with_graph_memory_admission_evidence(&evidence).unwrap();
    let requirement = proof.as_requirement();
    let source_term = requirement
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::SourceStage)
        .expect("normalized graph keeps its source-stage owner");
    let source_term_id = source_term.id().clone();
    let groups = requirement.isolated_phase_term_groups();
    let source_groups = groups.iter().filter(|group| group.contains(&source_term_id)).collect::<Vec<_>>();
    assert_eq!(source_groups.len(), 2, "normalized source survives in both graph phase alternatives");
    assert!(
        source_groups
            .iter()
            .all(|group| { group.iter().any(|term_id| term_id.role == OwnerTermRole::GraphPersistent) })
    );
    let expected_peak = evidence
        .bounded_live_graph_peak_with_source_stage_bytes(96)
        .expect("normalized source and graph live reservation fit");
    assert_eq!(requirement.soft_working_set_hint_bytes(), Some(expected_peak));
    let owner_sum = requirement.conservative_owner_sum().expect("normalized graph owner inventory is finite");
    let envelope =
        proof.resolve_reservation_envelope(reservation_policy()).expect("normalized graph envelope resolves");
    assert_eq!(owner_sum, envelope.complete_phase_peak_bytes().expect("normalized graph peak is complete"));
    let expected_owner_sum = requirement
        .owner_terms
        .iter()
        .try_fold(0_u64, |total, term| total.checked_add(term.bytes()))
        .expect("normalized graph owner terms fit");
    assert_eq!(owner_sum, expected_owner_sum, "conservative owner sum charges the normalized source once");
}

#[test]
fn normalized_source_schedule_allows_the_same_stage_in_both_alternatives() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    )
    .with_edge_normalize_inserted(true);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/normalized-schedule-source.parquet")
                .with_row_group_rows(vec![10])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&schema),
        vec!["src".to_owned(), "dst".to_owned()],
    )));
    let graph = builder.add_graph_algorithm(source, spec);
    let plan = builder.finish_with_host_sink(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();
    let producer = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .unwrap()
        .with_graph_memory_admission_evidence(&evidence)
        .unwrap();
    let consumer_plan = graph_output_union_consumer(&["graph"]);
    let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan).unwrap(),
        [("graph".to_owned(), producer)],
    )
    .expect("normalized source phase alternatives are complete");
    let envelope = composed
        .resolve_reservation_envelope(reservation_policy())
        .expect("normalized source schedule resolves with one shared owner");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
}

#[test]
fn graph_evidence_without_a_source_stage_remains_the_native_peak() {
    let plan = graph_bfs_plan();
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("graph plan must issue an engine-owned capability proof");

    assert_eq!(proof.soft_working_set_hint_bytes(), None);
    let proof = proof.with_graph_memory_admission_evidence(&evidence).unwrap();
    assert_eq!(
        proof.soft_working_set_hint_bytes(),
        Some(evidence.bounded_live_graph_peak_bytes().expect("graph live reservation fits"))
    );
    let requirement = proof.as_requirement();
    let owner_ids = requirement.owner_terms.iter().map(|term| term.id.clone()).collect::<HashSet<_>>();
    assert_eq!(owner_ids.len(), requirement.owner_terms.len(), "floor top-up has one owner identity");
    let owner_sum = requirement.conservative_owner_sum().expect("tiny graph owner inventory is finite");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).expect("tiny graph envelope resolves");
    assert!(owner_sum >= crate::graph_memory::GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_sum));
}

#[test]
fn graph_memory_evidence_attachment_rejects_missing_ambiguous_and_mismatched_scopes_atomically() {
    let plan = graph_bfs_plan();
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();

    let missing = QueryPlanCapabilityProof::for_query_plan(&plan).expect("native plan must issue a proof");
    let missing_before = missing.clone();
    let error = missing
        .clone()
        .with_graph_memory_admission_evidence(&evidence)
        .expect_err("generic proofs do not attest a graph execution scope");
    assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_missing"));
    assert_eq!(missing, missing_before);

    let second_plan = graph_bfs_plan_with_source_vertex(1);
    let ambiguous = QueryPlanCapabilityProof::combine(
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap(),
        [QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan).unwrap()],
    )
    .unwrap();
    let ambiguous_before = ambiguous.clone();
    let error = ambiguous
        .clone()
        .with_graph_memory_admission_evidence(&evidence)
        .expect_err("multiple attested graph scopes are ambiguous");
    assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_ambiguous"));
    assert_eq!(ambiguous, ambiguous_before);

    let mismatched_evidence =
        graph_memory_bounded_admission_evidence_for_plan(&second_plan, GraphMemoryCalibrationOverride::none(), None)
            .unwrap();
    let mismatched = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
    let mismatched_before = mismatched.clone();
    let error = mismatched
        .clone()
        .with_graph_memory_admission_evidence(&mismatched_evidence)
        .expect_err("evidence must match the attested graph plan");
    assert_eq!(error.fact_value("reason"), Some("graph_memory_evidence_scope_mismatch"));
    assert_eq!(mismatched, mismatched_before);
}

#[test]
fn required_floor_uses_checked_arithmetic() {
    let requirement = AttemptCapabilityEvidence::from_parts(u64::MAX, 1, None, EstimateQuality::Unknown, Vec::new());
    assert_eq!(
        requirement.required_floor_bytes().unwrap_err().fact_value("reason"),
        Some("attempt_requirement_floor_overflow")
    );
}

#[test]
fn test_placement_overlay_uses_the_maximum_floor_instead_of_sum() {
    let plan = ordinary_native_plan();
    let mut proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("ordinary plan issues a proof");
    proof.requirement.configured_floor_bytes = 7;
    let overlay = AttemptCapabilityEvidence::from_parts(0, 0, Some(11), EstimateQuality::Measured, Vec::new());

    let resolved =
        proof.into_requirement().with_test_placement_overlay(overlay).expect("placement-only evidence is valid");

    assert_eq!(resolved.configured_floor_bytes, 11);
    assert_eq!(resolved.required_floor_bytes_for(0).unwrap(), 11);
    assert_ne!(resolved.configured_floor_bytes, 18);
}

#[test]
fn test_placement_target_survives_proof_envelope_resolution_and_floor_sizing() {
    let plan = ordinary_native_plan();
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("ordinary plan issues a proof");
    let resolved = proof.requirement_for_device(0).expect("ordinary proof resolves on a device");
    let envelope = resolved
        .resolve_reservation_envelope(reservation_policy())
        .expect("ordinary proof resolves to a finite envelope");
    let sized = resolved
        .as_requirement()
        .clone()
        .with_reservation_envelope(&envelope)
        .with_test_placement_overlay(AttemptCapabilityEvidence::from_parts(
            0,
            0,
            Some(1_000_000),
            EstimateQuality::Measured,
            Vec::new(),
        ))
        .expect("placement-only evidence is valid");

    assert_eq!(sized.configured_floor_bytes, 1_000_000);
    assert_eq!(sized.required_floor_bytes_for(0).unwrap(), 1_000_000);
}

#[test]
fn test_placement_overlay_rejects_non_placement_evidence() {
    let mut non_placement = AttemptCapabilityEvidence::empty();
    non_placement.capability_floor_raise_bytes = 1;

    let error = AttemptCapabilityEvidence::empty()
        .with_test_placement_overlay(non_placement)
        .expect_err("test overlay must not attach capability evidence");

    assert_eq!(error.fact_value("reason"), Some("test_placement_overlay_not_placement_only"));

    let mut provenance = AttemptCapabilityEvidence::empty();
    provenance.bound_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
    let error = AttemptCapabilityEvidence::empty()
        .with_test_placement_overlay(provenance)
        .expect_err("test overlay must not attach bound provenance");
    assert_eq!(error.fact_value("reason"), Some("test_placement_overlay_not_placement_only"));
}

#[test]
fn legacy_band_is_an_interim_floor_and_hint() {
    let requirement = AttemptCapabilityEvidence::from_parts(10, 0, Some(20), EstimateQuality::Modeled, Vec::new());
    assert_eq!(requirement.required_floor_bytes().unwrap(), 10);
    assert_eq!(requirement.soft_working_set_hint_bytes, Some(20));
    assert_eq!(requirement.estimate_quality, EstimateQuality::Modeled);
}

#[test]
fn aggregate_sums_only_raises_and_unions_capabilities() {
    let capability = capability();
    let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
        10,
        [
            AttemptCapabilityEvidence::from_parts(99, 2, Some(12), EstimateQuality::Measured, vec![capability.clone()]),
            AttemptCapabilityEvidence::from_parts(88, 3, Some(20), EstimateQuality::Unknown, vec![capability]),
        ],
    )
    .unwrap();
    assert_eq!(result.configured_floor_bytes, 10);
    assert_eq!(result.capability_floor_raise_bytes, 5);
    assert_eq!(result.soft_working_set_hint_bytes, Some(20));
    assert_eq!(result.estimate_quality, EstimateQuality::Unknown);
    assert_eq!(result.required_capabilities.len(), 1);
}

#[test]
fn aggregate_clears_the_target_when_any_fragment_is_unmodeled() {
    let result = AttemptCapabilityEvidence::aggregate([
        AttemptCapabilityEvidence::from_parts(0, 0, Some(12), EstimateQuality::Modeled, Vec::new()),
        AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new()),
    ])
    .unwrap();

    assert_eq!(result.soft_working_set_hint_bytes(), None);
}

#[test]
fn aggregate_preserves_absent_source_stage_for_a_gpu_only_proof() {
    let proof = QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).expect("native plan issues a proof");
    let aggregate = QueryPlanCapabilityProof::combine(proof.clone(), std::iter::empty())
        .expect("one native proof still passes through the aggregate seam");

    assert_eq!(aggregate.as_requirement().source_stage_bytes, None);
    assert_eq!(aggregate, proof, "aggregation must not manufacture a zero local-Parquet stage");
}

#[test]
fn generic_gpu_sink_with_a_gpu_source_forwards_without_a_second_owner() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    let plan = builder.finish_gpu_resident(source);
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
    assert!(proof.as_requirement().owner_terms.is_empty());
    proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("a direct GPU-source sink forwards its upstream-owned handle");
}

#[test]
fn computed_gpu_sink_prices_concurrent_input_and_transformed_output() {
    let schema = Arc::new(Schema::new(vec![Field::new("keep", DataType::Boolean, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("input", Arc::clone(&schema)).with_row_count_upper_bound(3),
    ));
    let filter = builder.add_filter(source, Expr::column("keep"), Arc::clone(&schema));
    let plan = builder.finish_gpu_resident(filter);
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
    let retained_terms =
        proof.as_requirement().owner_terms.iter().filter(|term| term.is_retained_output()).collect::<Vec<_>>();
    assert_eq!(retained_terms.len(), 1);
    assert_eq!(retained_terms[0].retained_row_count_upper_bound(), Some(3));
    let source_terms =
        proof.as_requirement().owner_terms.iter().filter(|term| term.is_source_stage()).collect::<Vec<_>>();
    assert_eq!(source_terms.len(), 1);
    assert_eq!(source_terms[0].bytes(), retained_terms[0].bytes());
    assert_eq!(proof.as_requirement().conservative_owner_sum().unwrap(), source_terms[0].bytes() * 2);
    proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("computed GPU-source sink must price its concurrent input and output");
}

#[test]
fn decimal_projection_prices_native_rescale_workspace() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(20, 0), false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(20, -1), false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("input", input_schema).with_row_count_upper_bound(2),
    ));
    let projection = builder.add_projection(
        source,
        vec![ProjectionOutput::new(Expr::column("amount"), output_schema.field(0).as_ref().clone())],
    );
    let proof = QueryPlanCapabilityProof::for_query_plan(&builder.finish_gpu_resident(projection)).unwrap();
    let workspace = proof
        .as_requirement()
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::ConsumerProjectionCastWorkspace)
        .expect("decimal rescale must own its libcudf scalar workspace");

    assert_eq!(workspace.bytes(), 256);
    assert_eq!(workspace.id().owner, OwnerId::Consumer(projection));
}

#[test]
fn cugraph_side_input_disables_generic_gpu_output_ownership() {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("predicate", Arc::clone(&schema)).with_row_count_upper_bound(2),
    ));
    let plan = builder.finish_gpu_resident(source);

    let proof = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&plan)
        .expect("a bounded GPU-resident side input must issue a proof");
    assert!(
        proof.as_requirement().owner_terms.iter().all(|term| !matches!(term.id().owner, OwnerId::Producer(_))),
        "the side-input frame must be the sole owner for this native fragment"
    );
    assert_eq!(proof.as_requirement().side_input_frame_owner_term_bytes().len(), 1);
}

#[test]
fn cugraph_side_input_uses_bounded_literal_and_case_payload_evidence() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("selector", DataType::Int64, false)]));
    let input_batch =
        RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int64Array::from(vec![0, 1, 2])) as ArrayRef])
            .unwrap();
    let output_field = Field::new("vertex_id", DataType::Utf8, false);

    let side_input_proof = |expr: Expr| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
            Arc::clone(&input_schema),
            vec![input_batch.clone()],
        )));
        let projection = builder.add_projection(source, vec![ProjectionOutput::new(expr, output_field.clone())]);
        let plan = builder.finish_gpu_resident(projection);
        QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&plan)
    };

    let literal = side_input_proof(Expr::literal(LiteralValue::Utf8("seed".to_owned())))
        .expect("a bounded constant string projection must issue exact side-input evidence");
    let case = side_input_proof(Expr::case(
        vec![CaseWhen::new(
            Expr::binary(Expr::column("selector"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
            Expr::literal(LiteralValue::Utf8("edge".to_owned())),
        )],
        Expr::literal(LiteralValue::Utf8("node".to_owned())),
    ))
    .expect("a bounded CASE with complete literal arms must issue exact side-input evidence");

    let expected = crate::exec::join_output_upper_bound_bytes(
        3,
        &[crate::exec::JoinOutputColumnUpperBoundInput {
            data_type: DataType::Utf8,
            input_max_string_length_bytes: Some(4),
            input_total_string_bytes: Some(12),
        }],
    )
    .unwrap();
    assert_eq!(literal.as_requirement().side_input_frame_owner_term_bytes(), vec![expected]);
    assert_eq!(case.as_requirement().side_input_frame_owner_term_bytes(), vec![expected]);

    let unsupported =
        side_input_proof(Expr::substring(Expr::literal(LiteralValue::Utf8("vertex".to_owned())), 1, Some(3)))
            .expect_err("an unmodeled computed string expression must remain rejected");
    assert_eq!(unsupported.fact_value("schedule_reason"), Some("side_input_schema_unmodeled"));
    assert_eq!(unsupported.fact_value("column"), Some("vertex_id"));
    assert_eq!(unsupported.fact_value("data_type"), Some("Utf8"));
}

#[test]
fn composed_graph_schedule_charges_concurrent_side_input_with_graph_peak() {
    let side_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
    let mut side_builder = QueryPlanBuilder::new();
    let side_source = side_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("predicate", Arc::clone(&side_schema)).with_row_count_upper_bound(2),
    ));
    let side_plan = side_builder.finish_gpu_resident(side_source);
    let side_proof = QueryPlanCapabilityProof::try_for_cugraph_side_input_plan(&side_plan)
        .expect("bounded side input must issue a finite owner proof");
    let auxiliary_peak = side_proof.as_requirement().side_input_frame_owner_term_bytes().into_iter().sum::<u64>();
    let consumer_plan = graph_output_union_consumer(&["graph"]);

    let graph_only = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer proof must issue"),
        [("graph".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect("the graph-only ordered schedule must resolve");
    let graph_only_requirement = graph_only.as_requirement();
    let graph_only_shape = graph_only_requirement
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("graph-only consumer scope must be retained");
    let graph_only_peak = graph_only_shape
        .ordered_schedule()
        .expect("graph-only schedule must be retained")
        .checked_fold_with_inventory(&graph_only_requirement.reservation_shapes, &graph_only_requirement.owner_terms)
        .expect("graph-only schedule must remain complete");

    let graph_proof_with_auxiliary =
        QueryPlanCapabilityProof::combine(graph_bfs_proof_with_typed_output(0), [side_proof])
            .expect("graph and side-input proofs compose before producer normalization");
    let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer proof must issue"),
        [("graph".to_owned(), graph_proof_with_auxiliary)],
    )
    .expect("auxiliary side-input scopes must not invalidate the graph producer schedule");
    let requirement = composed.as_requirement();
    let consumer_shape = requirement
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("composed consumer scope must be retained");
    let schedule = consumer_shape.ordered_schedule().expect("ordered schedule must be retained");
    assert!(
        schedule
            .producer_phase_term_ids()
            .iter()
            .any(|term_id| term_id.role_label() == OwnerTermRole::SideInputFrame.as_str())
    );
    let ordered_peak = schedule
        .checked_fold_with_inventory(&requirement.reservation_shapes, &requirement.owner_terms)
        .expect("scheduled graph terms must remain complete");
    let envelope = composed
        .resolve_reservation_envelope(reservation_policy())
        .expect("ordered graph plus concurrent side-input scopes must resolve");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    let expected_peak = graph_only_peak.checked_add(auxiliary_peak).expect("test peak must fit");
    assert_eq!(ordered_peak, expected_peak);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected_peak));
}

#[test]
fn aggregate_rejects_raise_overflow() {
    let result = AttemptCapabilityEvidence::aggregate_with_configured_floor(
        0,
        [
            AttemptCapabilityEvidence::empty(),
            AttemptCapabilityEvidence::from_parts(0, u64::MAX, None, EstimateQuality::Unknown, Vec::new()),
            AttemptCapabilityEvidence::from_parts(0, 1, None, EstimateQuality::Unknown, Vec::new()),
        ],
    );
    assert_eq!(result.unwrap_err().fact_value("reason"), Some("attempt_requirement_capability_raise_overflow"));
}

fn finite_test_owner_terms(plan: &QueryPlan, first: u64, second: u64) -> Vec<AllocationOwnerTerm> {
    let scope = canonical_execution_scope_fingerprint(plan);
    vec![
        AllocationOwnerTerm::new(
            OwnerId::Source(plan.root_id()),
            OwnerTermRole::SourceStage,
            OccurrenceId::PlanLocal(plan.root_id()),
            scope.clone(),
            AllocationClass::ActivePhase,
            first,
            None,
        )
        .unwrap(),
        AllocationOwnerTerm::new(
            OwnerId::Consumer(plan.root_id()),
            OwnerTermRole::ConsumerOutput,
            OccurrenceId::PlanLocal(plan.root_id()),
            scope,
            AllocationClass::ActivePhase,
            second,
            None,
        )
        .unwrap(),
    ]
}

#[test]
fn conservative_owner_sum_charges_each_shared_inventory_term_once() {
    let plan = ordinary_native_plan();
    let mut evidence = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    evidence.owner_terms = finite_test_owner_terms(&plan, 11, 7);
    evidence.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
    assert_eq!(evidence.conservative_owner_sum().unwrap(), 18);
    let aggregate = AttemptCapabilityEvidence::aggregate([evidence]).unwrap();
    assert_eq!(aggregate.bound_derivation(), None, "provenance is selected when the envelope resolves");
}

#[test]
fn conservative_derivation_accepts_a_complete_zero_byte_inventory_but_not_missing_coverage() {
    let plan = ordinary_native_plan();
    let mut complete = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    complete.owner_terms = finite_test_owner_terms(&plan, 0, 0);
    complete.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
    let complete = AttemptCapabilityEvidence::aggregate([complete]).unwrap();
    assert_eq!(complete.bound_derivation(), None, "provenance is selected when the envelope resolves");

    let mut missing = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    missing.owner_terms = vec![finite_test_owner_terms(&plan, 0, 0).remove(0)];
    missing.reservation_shapes.push(AttemptReservationShape::for_query_plan(&plan));
    let missing = AttemptCapabilityEvidence::aggregate([missing]).unwrap();
    assert_eq!(missing.bound_derivation(), None);
}

#[test]
fn composed_resolver_rejects_graph_result_input_identity_mismatch_before_fallback() {
    let consumer_plan = graph_output_filter_consumer("producer");
    let producer = graph_bfs_proof_with_typed_output(0);
    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(
            Some("producer".to_owned()),
            Some("wrong-producer-name".to_owned()),
            Some(producer),
        )],
    )
    .expect_err("candidate normalization must reject before ordered or conservative derivation");
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_occurrence_identity_mismatch"));
}

#[test]
fn conservative_fallback_cannot_bypass_an_attested_producer_scope_mismatch() {
    let consumer_plan = graph_output_union_consumer(&["producer"]);
    let producer_plan = graph_bfs_plan_with_source_vertex(0);
    let different_scope_plan = graph_bfs_plan_with_source_vertex(1);
    let mut producer = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&producer_plan)
        .expect("producer graph plan must issue an attested proof");
    producer.requirement.reservation_shapes = vec![AttemptReservationShape::for_query_plan(&different_scope_plan)];

    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(Some("producer".to_owned()), None, Some(producer))],
    )
    .expect_err("conservative fallback must retain the attested producer scope check");
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_scope_coverage_mismatch"));
}

#[test]
fn conservative_owner_sum_rejects_duplicate_saturated_and_overflow_terms() {
    let plan = ordinary_native_plan();
    let terms = finite_test_owner_terms(&plan, 11, 7);
    let mut duplicate = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    duplicate.owner_terms = vec![terms[0].clone(), terms[0].clone()];
    assert_eq!(
        duplicate.conservative_owner_sum().unwrap_err().fact_value("reason"),
        Some("attempt_owner_term_duplicate")
    );

    let mut saturated = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    let mut saturated_term = terms[0].clone();
    saturated_term.bytes = FiniteBytes(u64::MAX);
    saturated.owner_terms = vec![saturated_term];
    assert_eq!(
        saturated.conservative_owner_sum().unwrap_err().fact_value("reason"),
        Some("attempt_owner_term_saturated")
    );

    let mut overflowing = AttemptCapabilityEvidence::from_parts(0, 0, None, EstimateQuality::Unknown, Vec::new());
    overflowing.owner_terms = finite_test_owner_terms(&plan, u64::MAX - 1, 2);
    assert_eq!(
        overflowing.conservative_owner_sum().unwrap_err().fact_value("reason"),
        Some("attempt_owner_term_sum_overflow")
    );
}

#[test]
fn composed_owner_saturation_rejects_first_and_later_producer_occurrences_without_fallback() {
    let consumer_plan = graph_output_union_consumer(&["first", "later"]);
    let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ("first".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("later".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .expect("two producer occurrences provide a complete owner inventory");

    for (occurrence_id, canonical_occurrence_label) in [("first", "composed:first"), ("later", "composed:later")] {
        let mut saturated = composed.clone().into_requirement();
        let term = saturated
            .owner_terms
            .iter_mut()
            .find(|term| term.is_retained_output() && term.id().occurrence_label() == canonical_occurrence_label)
            .expect("each producer occurrence owns one retained output term");
        term.bytes = FiniteBytes(u64::MAX);
        assert_eq!(
            term.id().occurrence_label(),
            canonical_occurrence_label,
            "{occurrence_id} producer must retain its canonical owner-term occurrence label",
        );
        assert_eq!(
            saturated.conservative_owner_sum().unwrap_err().fact_value("reason"),
            Some("attempt_owner_term_saturated")
        );

        let error = QueryPlanCapabilityProof { requirement: saturated }
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("saturation must make both derivations unavailable");
        assert_eq!(error.fact_value("reason"), Some("attempt_bound_derivation_unavailable"));
        assert_eq!(error.fact_value("conservative_reason"), Some("attempt_owner_term_saturated"));
    }
}

#[test]
fn cugraph_graph_scope_rejects_an_unattested_native_plan_with_stable_facts() {
    let attested_plan = graph_bfs_plan_with_source_vertex(1);
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&attested_plan)
        .expect("attested graph execution has capability requirements");
    let error = proof
        .as_requirement()
        .validate_covers_plan(&graph_bfs_plan())
        .expect_err("a plan-scoped cuGraph proof must not authorize a different native graph plan");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
    assert_eq!(
        error.fact_value("missing_capabilities"),
        Some("operator_output:unsupported_graph_algorithm_reservation_contract")
    );
}

#[test]
fn resident_output_payload_evidence_requires_bound_producer_scope() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("vertex", DataType::Utf8, false),
    ]));
    let producer_file = LocalParquetFile::new("/tmp/resident-string-producer.parquet")
        .with_row_groups(vec![0])
        .with_row_group_rows(vec![4])
        .with_variable_width_payload_evidence(std::collections::BTreeMap::from([(
            1,
            crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                1,
                vec![0],
                321,
                crate::parquet_stats::ParquetVariableWidthEncodingProof::Plain,
            ),
        )]));
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![producer_file],
        Arc::clone(&schema),
        vec!["id".to_owned(), "vertex".to_owned()],
    )));
    let producer_plan = producer_builder.finish_gpu_resident(producer_source);

    let mut consumer_builder = QueryPlanBuilder::new();
    let resident = consumer_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("codebook", Arc::clone(&schema))
            .with_row_count_upper_bound(4)
            .with_unique_key_columns(BTreeSet::from(["id".to_owned()])),
    ));
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int64Array::from(vec![0, 1])) as ArrayRef, Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef],
    )
    .unwrap();
    let edges =
        consumer_builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(edge_schema, vec![edge_batch])));
    let graph = consumer_builder.add_graph_algorithm(
        edges,
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::PageRank,
            GraphEdgeColumnBindings::new("src", "dst", None::<String>),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
        ),
    );
    let decoded = consumer_builder.add_join(
        graph,
        resident,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                crate::expr::ColumnRef::input(0, "vertex", "vertex"),
                crate::expr::ColumnRef::input(1, "id", "id"),
            )],
        )
        .with_at_most_one_match_per_bounded_row(JoinSide::Left)
        .with_output_projection(vec![JoinOutputColumn::right(
            crate::expr::ColumnRef::input(1, "vertex", "vertex"),
            "vertex",
        )]),
        Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)])),
    );
    let consumer_plan = consumer_builder.finish_gpu_resident(decoded);
    let consumer_scope = canonical_execution_scope_fingerprint(&consumer_plan);
    let consumer_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan_with_resident_inputs(
        &consumer_plan,
        [("codebook", &producer_plan)],
    )
    .unwrap()
    .expect("the resident consumer must issue a scoped proof");

    assert!(
        consumer_proof
            .as_requirement()
            .output_column_evidence_for_scope(&consumer_scope)
            .string_payload_bound("vertex")
            .is_none(),
        "detached resident evidence must remain unavailable"
    );

    let producer_proof = QueryPlanCapabilityProof::for_query_plan(&producer_plan)
        .expect("the exact producer must issue its own scoped proof");
    let combined =
        AttemptCapabilityEvidence::aggregate([consumer_proof.into_requirement(), producer_proof.into_requirement()])
            .expect("the bound producer and resident consumer proofs must combine");
    assert_eq!(
        combined
            .output_column_evidence_for_scope(&consumer_scope)
            .string_payload_bound("vertex")
            .and_then(|bound| bound.total_payload_upper_bound_bytes),
        Some(321),
    );
}

#[test]
fn exact_cugraph_plan_proof_replaces_only_the_engine_graph_blocker() {
    let plan = graph_bfs_plan();
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("graph execution has capability requirements");

    assert!(proof.required_capabilities().iter().any(|requirement| {
        requirement.kind == ReservationKind::GraphAlgorithm
            && requirement.required_contract == "cugraph_graph_execution_contract"
    }));
    assert!(!proof.required_capabilities().iter().any(is_generic_graph_execution_blocker));
    assert_eq!(proof.as_requirement().attested_cugraph_graph_execution_scopes.len(), 1);
    proof
        .as_requirement()
        .validate_covers_plan(&plan)
        .expect("the issued cuGraph proof must cover its exact internal plan");
}

#[test]
fn literal_string_source_owner_term_uses_exact_batch_statistics() {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vec!["alpha", "b"]))]).unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    let plan = builder.finish_gpu_resident(source);

    let terms = literal_batch_source_owner_terms(&plan).expect("literal strings have a finite GPU frame bound");

    assert_eq!(terms.len(), 1);
    assert!(terms[0].bytes() > 0);
    assert_eq!(terms[0].id.role, OwnerTermRole::SourceStage);
}

#[test]
fn combining_cugraph_proofs_retains_each_attested_graph_scope() {
    let native_plan = ordinary_native_plan();
    let first_plan = graph_bfs_plan_with_source_vertex(0);
    let second_plan = graph_bfs_plan_with_source_vertex(1);
    let unscoped_plan = graph_bfs_plan_with_source_vertex(2);
    let proof = QueryPlanCapabilityProof::combine(
        QueryPlanCapabilityProof::for_query_plan(&native_plan)
            .expect("native plan has an explicit sink capability requirement"),
        [
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&first_plan)
                .expect("first graph plan has capability requirements"),
            QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&second_plan)
                .expect("second graph plan has capability requirements"),
        ],
    )
    .expect("native and plan-scoped graph proofs must combine");

    let requirement = proof.as_requirement();
    assert_eq!(requirement.attested_cugraph_graph_execution_scopes.len(), 2);
    requirement.validate_covers_plan(&native_plan).expect("combined proof must retain the native plan capability");
    requirement.validate_covers_plan(&first_plan).expect("combined proof must retain the first graph scope");
    requirement.validate_covers_plan(&second_plan).expect("combined proof must retain the second graph scope");
    let error =
        requirement.validate_covers_plan(&unscoped_plan).expect_err("combined proof must not widen its graph scope");
    assert_eq!(error.fact_value("reason"), Some("admitted_capability_evidence_mismatch"));
    assert_eq!(
        error.fact_value("missing_capabilities"),
        Some("operator_output:unsupported_graph_algorithm_reservation_contract")
    );
}

#[test]
fn graph_owner_terms_reject_duplicate_standalone_proofs() {
    let proof = graph_bfs_proof_with_typed_output(0);
    let graph_persistent = proof
        .as_requirement()
        .owner_terms
        .iter()
        .find(|term| term.id.role == OwnerTermRole::GraphPersistent)
        .expect("graph proof carries a persistent graph owner");
    assert_eq!(graph_persistent.class, AllocationClass::ActivePhase);
    let duplicate =
        AttemptCapabilityEvidence::aggregate([proof.as_requirement().clone(), proof.as_requirement().clone()])
            .expect_err("standalone graph terms with one plan-local occurrence cannot be deduplicated");
    assert_eq!(duplicate.fact_value("reason"), Some("attempt_owner_term_duplicate"));
    assert_eq!(duplicate.fact_value("owner_term_role"), Some("graph_persistent"));
    assert!(duplicate.fact_value("occurrence").is_some_and(|value| value.starts_with("plan_local:n")));
}

#[test]
fn source_graph_and_consumer_terms_are_scoped_and_typed_in_one_inventory() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut source_builder = QueryPlanBuilder::new();
    let source_id = source_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/inventory-source.parquet")
                .with_row_groups(vec![0])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        Arc::clone(&schema),
        vec!["value".to_owned()],
    )));
    let source_plan = source_builder.finish_with_host_sink(source_id);
    let source_proof = QueryPlanCapabilityProof::for_query_plan(&source_plan).unwrap();
    let source_requirement = source_proof.as_requirement();
    let source_term = source_requirement
        .owner_terms
        .iter()
        .find(|term| matches!(term.id.owner, OwnerId::Source(id) if id == source_id))
        .expect("known local source has one NodeId-keyed owner term");
    assert_eq!(source_term.id.role, OwnerTermRole::SourceStage);
    assert_eq!(source_term.class, AllocationClass::ActivePhase);
    assert_eq!(source_term.id.execution_scope_fingerprint, canonical_execution_scope_fingerprint(&source_plan));

    let graph_plan = graph_bfs_plan();
    let graph_evidence =
        graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .unwrap();
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .unwrap()
        .with_graph_memory_admission_evidence(&graph_evidence)
        .unwrap();
    assert!(graph_proof.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Graph(_))
            && term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&graph_plan)
    }));

    let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, true);
    let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .unwrap();
    assert!(composed.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Consumer(_))
            && term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&consumer_plan)
    }));
}

#[test]
fn composed_schedule_keeps_zero_fixed_projection_with_explicit_passthrough() {
    let consumer_plan = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Int64);
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [("only".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .unwrap();
    assert!(combined.as_requirement().owner_terms.iter().all(|term| term.bytes.get() != 0));
    let consumer_shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("composed consumer scope is retained");
    let schedule = consumer_shape.ordered_schedule().expect("ordered schedule is attached");
    assert!(schedule.attempt_wide_fixed_term_ids().is_empty());
    assert_eq!(schedule.consumer_state(), AttemptReservationConsumerState::Passthrough);
    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("explicit passthrough has a finite ordered envelope");
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_schedule_folds_a_real_attempt_fixed_owner_term() {
    let consumer_plan = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Int64);
    let baseline = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [("only".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .unwrap()
    .resolve_reservation_envelope(reservation_policy())
    .unwrap();
    let baseline_peak = baseline.complete_phase_peak_bytes().expect("baseline schedule is finite");

    let producer_plan = graph_bfs_plan_with_source_vertex(0);
    let mut producer = graph_bfs_proof_with_typed_output(0);
    let fixed_bytes = baseline_peak.checked_add(1).expect("test fixed owner fits");
    producer.requirement.owner_terms.push(
        AllocationOwnerTerm::new(
            OwnerId::Attempt,
            OwnerTermRole::AttemptWideFixed,
            OccurrenceId::PlanLocal(producer_plan.root_id()),
            canonical_execution_scope_fingerprint(&producer_plan),
            AllocationClass::AttemptWideFixed,
            fixed_bytes,
            None,
        )
        .unwrap(),
    );

    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [("only".to_owned(), producer)],
    )
    .unwrap();
    let consumer_shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("composed consumer scope is retained");
    let schedule = consumer_shape.ordered_schedule().expect("ordered schedule is attached");
    assert_eq!(schedule.attempt_wide_fixed_term_ids().len(), 1);
    let envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();
    let projection = envelope
        .composed_schedule_projection(
            crate::observability::AttemptId::mint(),
            0,
            1,
            envelope.required_phase_floor_bytes(),
        )
        .unwrap();
    assert_eq!(projection.attempt_wide_fixed_bytes, fixed_bytes);
    assert!(projection.complete_peak_bytes >= fixed_bytes);
}

#[test]
fn attempt_owner_term_rejects_mismatched_owner_role_and_class() {
    let plan = graph_output_union_consumer(&["fixed-shape"]);
    AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::AttemptWideFixed,
        OccurrenceId::PlanLocal(plan.root_id()),
        canonical_execution_scope_fingerprint(&plan),
        AllocationClass::AttemptWideFixed,
        1,
        None,
    )
    .expect("attempt owner terms accept the dedicated fixed role and class");
    let error = AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::GraphPersistent,
        OccurrenceId::PlanLocal(plan.root_id()),
        canonical_execution_scope_fingerprint(&plan),
        AllocationClass::ActivePhase,
        1,
        None,
    )
    .expect_err("attempt owner terms must use the dedicated fixed role and class");
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_owner_role_class_mismatch"));
}

#[test]
fn ordered_schedule_validates_fixed_term_identity_and_checked_sum() {
    let plan = graph_bfs_plan_with_source_vertex(0);
    let scope = canonical_execution_scope_fingerprint(&plan);
    let mut owner_terms = graph_bfs_proof_with_typed_output(0)
        .as_requirement()
        .owner_terms
        .iter()
        .map(|term| term.rebind_composed_for_test("expected"))
        .collect::<Result<Vec<_>>>()
        .unwrap();
    let fixed = AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::AttemptWideFixed,
        OccurrenceId::Composed("fixed".to_owned()),
        scope.clone(),
        AllocationClass::AttemptWideFixed,
        3,
        None,
    )
    .unwrap();
    owner_terms.push(fixed.clone());
    let retained_output_term_id = owner_terms.iter().find(|term| term.is_retained_output()).unwrap().id().clone();
    let isolated_term_ids = owner_terms
        .iter()
        .find(|term| !term.is_retained_output() && !term.is_attempt_wide_fixed())
        .map(|term| term.id().clone())
        .unwrap();
    let all_isolated_term_ids = owner_terms
        .iter()
        .filter(|term| !term.is_retained_output() && !term.is_attempt_wide_fixed())
        .map(|term| term.id().clone())
        .collect::<Vec<_>>();

    let fold = |fixed_term_ids: Vec<OwnerTermId>, terms: &[AllocationOwnerTerm]| {
        let mut shape = AttemptReservationShape::for_query_plan(&plan);
        shape
            .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
                "expected",
                scope.clone(),
                retained_output_term_id.clone(),
            )])
            .unwrap();
        let schedule = AttemptReservationSchedule::from_phase_term_ids(
            fixed_term_ids,
            vec![AttemptReservationProducerPhase::from_term_groups(vec![all_isolated_term_ids.clone()])],
            AttemptReservationConsumerState::Passthrough,
            Vec::new(),
        );
        shape.attach_ordered_schedule(schedule.clone()).unwrap();
        schedule.checked_fold_with_inventory(&[shape, AttemptReservationShape::for_query_plan(&plan)], terms)
    };

    fold(vec![fixed.id().clone()], &owner_terms).expect("a real fixed owner term folds once");
    let duplicate = fold(vec![fixed.id().clone(), fixed.id().clone()], &owner_terms).unwrap_err();
    assert_eq!(duplicate.fact_value("schedule_reason"), Some("duplicate_owner_term_reference"));

    let wrong_role = fold(vec![isolated_term_ids.clone()], &owner_terms).unwrap_err();
    assert_eq!(wrong_role.fact_value("schedule_reason"), Some("owner_term_role_class_mismatch"));

    let missing = fold(vec![OwnerTermId::synthetic_for_test("missing", &scope)], &owner_terms).unwrap_err();
    assert_eq!(missing.fact_value("schedule_reason"), Some("missing_owner_term_reference"));

    let overflowing_first = AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::AttemptWideFixed,
        OccurrenceId::Composed("overflow-first".to_owned()),
        scope.clone(),
        AllocationClass::AttemptWideFixed,
        u64::MAX - 1,
        None,
    )
    .unwrap();
    let overflowing_second = AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::AttemptWideFixed,
        OccurrenceId::Composed("overflow-second".to_owned()),
        scope.clone(),
        AllocationClass::AttemptWideFixed,
        2,
        None,
    )
    .unwrap();
    let mut overflowing_terms = owner_terms;
    overflowing_terms.extend([overflowing_first.clone(), overflowing_second.clone()]);
    let overflow =
        fold(vec![overflowing_first.id().clone(), overflowing_second.id().clone()], &overflowing_terms).unwrap_err();
    assert_eq!(overflow.fact_value("schedule_reason"), Some("attempt_wide_fixed_sum_overflow"));
}

#[test]
fn ordered_envelope_charges_fixed_retained_and_consumer_terms_exactly_once() {
    const FIXED_BYTES: u64 = 11;
    const RETAINED_BYTES: u64 = 17;
    const CONSUMER_BYTES: u64 = 23;
    let consumer_plan = graph_output_union_consumer(&["producer"]);
    let producer_plan = graph_bfs_plan_with_source_vertex(0);
    let consumer_scope = canonical_execution_scope_fingerprint(&consumer_plan);
    let producer_scope = canonical_execution_scope_fingerprint(&producer_plan);

    let fixed = AllocationOwnerTerm::new(
        OwnerId::Attempt,
        OwnerTermRole::AttemptWideFixed,
        OccurrenceId::Composed("fixed".to_owned()),
        consumer_scope.clone(),
        AllocationClass::AttemptWideFixed,
        FIXED_BYTES,
        None,
    )
    .unwrap();
    let retained = AllocationOwnerTerm::new(
        OwnerId::Producer(producer_plan.root_id()),
        OwnerTermRole::RetainedOutput,
        OccurrenceId::Composed("producer".to_owned()),
        producer_scope.clone(),
        AllocationClass::AttemptRetained,
        RETAINED_BYTES,
        Some(20),
    )
    .unwrap();
    let consumer = AllocationOwnerTerm::new(
        OwnerId::Consumer(consumer_plan.root_id()),
        OwnerTermRole::ConsumerOutput,
        OccurrenceId::PlanLocal(consumer_plan.root_id()),
        consumer_scope.clone(),
        AllocationClass::ActivePhase,
        CONSUMER_BYTES,
        None,
    )
    .unwrap();

    let retained_term_id = retained.id().clone();
    let mut manifest = AttemptReservationShape::for_query_plan(&consumer_plan);
    manifest
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "producer",
            producer_scope.clone(),
            retained_term_id.clone(),
        )])
        .unwrap();
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        vec![fixed.id().clone()],
        vec![AttemptReservationProducerPhase::from_term_groups(vec![Vec::new()])],
        AttemptReservationConsumerState::Bounded,
        vec![consumer.id().clone()],
    );
    manifest.attach_ordered_schedule(schedule).unwrap();
    let producer_scope_shape = AttemptReservationShape::for_query_plan(&producer_plan);
    let scopes = vec![manifest, producer_scope_shape];
    let owner_terms = vec![fixed, retained.clone(), consumer];

    let envelope = AttemptReservationEnvelope::resolve_with_owner_terms(&scopes, &owner_terms, reservation_policy())
        .expect("the production ordered fold must resolve the complete owner inventory");
    let expected = FIXED_BYTES + RETAINED_BYTES + CONSUMER_BYTES;
    let resolved_floor = envelope.required_phase_floor_bytes();
    assert_eq!(resolved_floor, expected);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(expected));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    assert_ne!(resolved_floor, FIXED_BYTES + CONSUMER_BYTES, "retained output must not disappear from the floor");
    assert_ne!(
        resolved_floor,
        FIXED_BYTES + RETAINED_BYTES * 2 + CONSUMER_BYTES,
        "retained output must not be charged twice",
    );

    let occurrence = scopes[0].producer_occurrences().first().expect("manifest retains producer occurrence");
    assert_eq!(occurrence.retained_output_term_id(), &retained_term_id);
    assert_eq!(
        owner_terms.iter().filter(|term| term.id().role == OwnerTermRole::SourceStage).count(),
        0,
        "the graph-result source references the producer retained owner instead of minting a second source term",
    );
}

#[test]
fn graph_evidence_ignores_an_unrelated_relational_source_stage() {
    let graph_plan = graph_bfs_plan();
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&graph_plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("graph plan must produce bounded evidence");
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue a capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("graph evidence must attach to its exact plan");

    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut source_builder = QueryPlanBuilder::new();
    let source = source_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/unrelated-relational-source.parquet")
                .with_row_groups(vec![0])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        schema,
        vec!["value".to_owned()],
    )));
    let source_plan = source_builder.finish_with_host_sink(source);
    let source_proof = QueryPlanCapabilityProof::for_query_plan(&source_plan)
        .expect("relational source must issue its own capability proof");

    let combined = QueryPlanCapabilityProof::combine(graph_proof, [source_proof])
        .expect("graph and unrelated relational proofs compose");
    let graph_floor = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .filter_map(|shape| shape.graph_memory_admission_evidence())
        .map(GraphMemoryAdmissionEvidence::cold_model_requested_bytes)
        .next()
        .expect("combined proof retains graph scope");
    let requirement = combined.as_requirement();
    assert_eq!(requirement.source_stage_bytes, None, "an unrelated relational source must not become graph input");
    let relational_source = requirement
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::SourceStage)
        .expect("unrelated relational source retains its own owner");
    assert!(
        requirement.isolated_phase_term_groups().iter().any(|group| group == &vec![relational_source.id().clone()]),
        "unrelated relational source remains in its standalone active phase",
    );
    assert!(requirement.reservation_shapes.iter().any(|shape| {
        shape.has_local_parquet_source()
            && shape.has_modeled_source_stage()
            && !shape.is_cugraph_graph_input_scope()
            && !shape.is_accounted_by_graph_phase()
    }));
    assert!(requirement.reservation_shapes.iter().any(|shape| {
        shape.graph_memory_admission_evidence().is_some()
            && shape.graph_memory_admission_evidence().unwrap().cold_model_requested_bytes() == graph_floor
    }));
}

#[test]
fn equal_composed_graph_producers_retain_distinct_owner_term_occurrences() {
    let consumer_plan = graph_output_union_consumer(&["same-0", "same-1"]);
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ("same-0".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("same-1".to_owned(), graph_bfs_proof_with_typed_output(0)),
        ],
    )
    .expect("runtime occurrence rebinding keeps equal graph producers distinct");
    let requirement = combined.as_requirement();
    let graph_owner_top_up_count = requirement
        .owner_terms
        .iter()
        .filter(|term| term.id.role == OwnerTermRole::GraphLiveReservationFloorTopUp)
        .count();
    let expected_owner_term_count = (if cfg!(feature = "cugraph") { 11 } else { 9 }) + graph_owner_top_up_count;
    assert_eq!(requirement.owner_terms.len(), expected_owner_term_count);
    let retained_occurrences = requirement
        .owner_terms
        .iter()
        .filter(|term| term.id.role == OwnerTermRole::RetainedOutput)
        .map(|term| term.id.occurrence.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(retained_occurrences, HashSet::from(["composed:same-0".to_owned(), "composed:same-1".to_owned()]));
    assert_eq!(
        requirement.owner_terms.iter().filter(|term| term.id.role == OwnerTermRole::RetainedOutput).count(),
        2,
        "each producer retained term is represented exactly once"
    );
    assert!(requirement.conservative_owner_sum().is_ok());
}

#[test]
fn ordered_schedule_rejects_a_retained_term_bound_to_the_wrong_composed_occurrence() {
    let plan = graph_bfs_plan_with_source_vertex(0);
    let proof = graph_bfs_proof_with_typed_output(0);
    let scope = canonical_execution_scope_fingerprint(&plan);
    let owner_terms = proof
        .as_requirement()
        .owner_terms
        .iter()
        .map(|term| term.rebind_composed_for_test(if term.is_retained_output() { "wrong" } else { "expected" }))
        .collect::<Result<Vec<_>>>()
        .expect("test terms can be rebound to composed occurrences");
    let retained_output_term_id = owner_terms
        .iter()
        .find(|term| term.is_retained_output())
        .expect("graph proof carries one retained output term")
        .id()
        .clone();
    let isolated_term_id = owner_terms
        .iter()
        .find(|term| !term.is_retained_output())
        .expect("graph proof carries an isolated phase term")
        .id()
        .clone();
    let mut shape = AttemptReservationShape::for_query_plan(&plan);
    shape
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "expected",
            scope.clone(),
            retained_output_term_id,
        )])
        .unwrap();
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![isolated_term_id]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    shape.attach_ordered_schedule(schedule.clone()).unwrap();
    let scopes = vec![shape, AttemptReservationShape::for_query_plan(&plan)];

    let error = schedule
        .checked_fold_with_inventory(&scopes, &owner_terms)
        .expect_err("ordered admission must bind retained output to its runtime occurrence");
    assert_eq!(error.fact_value("schedule_reason"), Some("retained_output_term_occurrence_mismatch"));
    assert_eq!(error.fact_value("expected_occurrence"), Some("composed:expected"));
    assert_eq!(error.fact_value("actual_occurrence"), Some("composed:wrong"));
}

#[test]
fn ordered_schedule_rejects_a_retained_term_bound_to_the_wrong_producer_scope() {
    let plan = graph_bfs_plan_with_source_vertex(0);
    let different_scope_plan = graph_bfs_plan_with_source_vertex(1);
    let proof = graph_bfs_proof_with_typed_output(0);
    let scope = canonical_execution_scope_fingerprint(&plan);
    let different_scope = canonical_execution_scope_fingerprint(&different_scope_plan);
    let owner_terms = proof
        .as_requirement()
        .owner_terms
        .iter()
        .map(|term| term.rebind_composed_for_test("expected"))
        .collect::<Result<Vec<_>>>()
        .expect("test terms can bind to one composed occurrence");
    let retained_output_term_id = owner_terms
        .iter()
        .find(|term| term.is_retained_output())
        .expect("graph proof carries one retained output term")
        .id()
        .clone();
    let isolated_term_id = owner_terms
        .iter()
        .find(|term| !term.is_retained_output())
        .expect("graph proof carries an isolated phase term")
        .id()
        .clone();
    let mut shape = AttemptReservationShape::for_query_plan(&plan);
    shape
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "expected",
            different_scope.clone(),
            retained_output_term_id,
        )])
        .unwrap();
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![isolated_term_id]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    shape.attach_ordered_schedule(schedule.clone()).unwrap();
    let scopes = vec![shape, AttemptReservationShape::for_query_plan(&different_scope_plan)];

    let error = schedule
        .checked_fold_with_inventory(&scopes, &owner_terms)
        .expect_err("ordered admission must bind retained output to its producer scope");
    assert_eq!(error.fact_value("schedule_reason"), Some("retained_output_term_scope_mismatch"));
    assert_eq!(error.fact_value("expected_scope"), Some(different_scope.as_str()));
    assert_eq!(error.fact_value("actual_scope"), Some(scope.as_str()));
}

#[test]
fn composed_consumer_rejects_an_extra_unconsumed_producer_occurrence() {
    let consumer_plan = graph_output_union_consumer(&["only"]);
    let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ("only".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("extra".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .expect_err("an extra producer occurrence must not be silently dropped");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_producer_occurrence_unconsumed"));
    assert_eq!(error.fact_value("occurrence_id"), Some("extra"));
}

#[test]
fn composed_graph_schedule_seam_derives_complete_typed_union_evidence() {
    let consumer_plan =
        graph_output_union_consumer(&["zeta-occurrence", "alpha-occurrence", "mu-occurrence", "beta-occurrence"]);
    let first = graph_bfs_proof_with_typed_output(0);
    let second = graph_bfs_proof_with_typed_output(1);
    let third = graph_bfs_proof_with_typed_output(2);
    let fourth = graph_bfs_proof_with_typed_output(3);
    let consumer =
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).expect("consumer proof must be engine-issued");

    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [
            ("zeta-occurrence".to_owned(), first),
            ("alpha-occurrence".to_owned(), second),
            ("mu-occurrence".to_owned(), third),
            ("beta-occurrence".to_owned(), fourth),
        ],
    )
    .expect("typed producer and engine consumer evidence must derive one complete schedule");
    assert_eq!(combined.as_requirement().reservation_shapes.len(), 5);
    let consumer_shape = combined
        .as_requirement()
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("the composed consumer scope must be present");
    let binding_occurrences =
        consumer_shape.producer_occurrences().iter().map(|occurrence| occurrence.occurrence_id()).collect::<Vec<_>>();
    assert_eq!(binding_occurrences, vec!["zeta-occurrence", "alpha-occurrence", "mu-occurrence", "beta-occurrence"]);
    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("complete typed graph and union evidence must resolve before placement");
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_graph_schedule_prices_the_canonical_join_and_materialized_unary_branch() {
    let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, true);
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .expect("fixed-width non-null graph join must have complete engine-owned evidence");
    let envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert!(envelope.complete_phase_peak_bytes().is_some());
    assert!(envelope.required_phase_floor_bytes() > 0);
}

#[test]
fn composed_resolver_keeps_graph_derivation_when_a_relational_gpu_producer_is_added() {
    let graph_only_consumer = graph_output_union_consumer(&["graph-left", "graph-right"]);
    let graph_only = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &graph_only_consumer,
        QueryPlanCapabilityProof::for_query_plan(&graph_only_consumer).unwrap(),
        [
            ComposedGraphProducerCandidate::new(
                Some("graph-left".to_owned()),
                Some("graph-left".to_owned()),
                Some(graph_bfs_proof_with_typed_output(0)),
            ),
            ComposedGraphProducerCandidate::new(
                Some("graph-right".to_owned()),
                Some("graph-right".to_owned()),
                Some(graph_bfs_proof_with_typed_output(1)),
            ),
        ],
    )
    .expect("two graph producers must resolve through the production seam");
    let graph_only_envelope = graph_only
        .resolve_reservation_envelope(reservation_policy())
        .expect("two graph producers must resolve a finite envelope");
    let graph_only_derivation =
        graph_only_envelope.bound_derivation().expect("two graph producers must select a finite bound derivation");

    let mixed_consumer = graph_output_union_consumer(&["graph-left", "graph-right", "relational"]);
    let relational = relational_gpu_producer_proof();
    let mixed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &mixed_consumer,
        QueryPlanCapabilityProof::for_query_plan(&mixed_consumer).unwrap(),
        [
            ComposedGraphProducerCandidate::new(
                Some("graph-left".to_owned()),
                Some("graph-left".to_owned()),
                Some(graph_bfs_proof_with_typed_output(0)),
            ),
            ComposedGraphProducerCandidate::new(
                Some("graph-right".to_owned()),
                Some("graph-right".to_owned()),
                Some(graph_bfs_proof_with_typed_output(1)),
            ),
            ComposedGraphProducerCandidate::new(Some("relational".to_owned()), None, Some(relational)),
        ],
    )
    .expect("adding a complete relational GPU producer must not change admissibility");
    let mixed_requirement = mixed.as_requirement();
    let relational_terms = mixed_requirement
        .owner_terms
        .iter()
        .filter(|term| term.id().occurrence_label() == "composed:relational")
        .collect::<Vec<_>>();
    assert!(!relational_terms.is_empty(), "the relational producer must retain engine-owned terms");
    assert!(relational_terms.iter().all(|term| term.bytes() > 0));
    let mixed_envelope = mixed
        .resolve_reservation_envelope(reservation_policy())
        .expect("the mixed producer set must resolve a finite envelope");
    assert_eq!(mixed_envelope.bound_derivation(), Some(graph_only_derivation));
    assert!(mixed_envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_graph_schedule_routes_n1_and_equal_n3_occurrences_through_one_path() {
    let n1_consumer = graph_output_union_consumer(&["only"]);
    let n1 = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &n1_consumer,
        QueryPlanCapabilityProof::for_query_plan(&n1_consumer).unwrap(),
        [ComposedGraphProducerCandidate::new(
            Some("only".to_owned()),
            Some("only".to_owned()),
            Some(graph_bfs_proof_with_typed_output(0)),
        )],
    )
    .unwrap();
    let n1_envelope = n1
        .resolve_reservation_envelope(reservation_policy())
        .expect("one plain graph producer must resolve a finite envelope");
    assert!(n1_envelope.complete_phase_peak_bytes().is_some());
    assert_eq!(n1_envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));

    let n3_consumer = graph_output_union_consumer(&["same-0", "same-1", "same-2"]);
    let equal = graph_bfs_proof_with_typed_output(0);
    let n3 = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &n3_consumer,
        QueryPlanCapabilityProof::for_query_plan(&n3_consumer).unwrap(),
        [("same-0".to_owned(), equal.clone()), ("same-1".to_owned(), equal.clone()), ("same-2".to_owned(), equal)],
    )
    .unwrap();
    let envelope = n3.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert!(envelope.complete_phase_peak_bytes().is_some());
    assert_eq!(n3.as_requirement().reservation_shapes.len(), 4, "equal producer scopes remain three occurrences");
    let floor_ids = n3
        .as_requirement()
        .owner_terms
        .iter()
        .filter(|term| term.id.role == OwnerTermRole::GraphLiveReservationFloorTopUp)
        .map(|term| term.id.clone())
        .collect::<HashSet<_>>();
    assert_eq!(floor_ids.len(), 3, "each tiny producer owns one distinct graph floor top-up");
    let scheduled_ids = envelope
        .ordered_schedule()
        .expect("equal graph producers use the ordered derivation")
        .producer_phase_term_ids()
        .into_iter()
        .collect::<HashSet<_>>();
    assert!(floor_ids.is_subset(&scheduled_ids), "ordered graph phases must reference each floor top-up");
}

#[test]
fn production_pagerank_forceatlas2_proof_selects_ordered_timeline() {
    let consumer_plan = pagerank_forceatlas2_join_consumer();
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("the production consumer IR must issue a composed proof"),
        [
            ("pagerank".to_owned(), production_graph_proof(GraphAlgorithm::PageRank)),
            ("forceatlas2".to_owned(), production_graph_proof(GraphAlgorithm::ForceAtlas2)),
        ],
    )
    .expect("the production PageRank + ForceAtlas2 topology must compose");
    let envelope = combined
        .resolve_reservation_envelope(reservation_policy())
        .expect("the production graph proof must resolve a finite envelope");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));

    let mut incomplete = combined.into_requirement();
    let consumer_shape_index = incomplete
        .reservation_shapes
        .iter()
        .position(|shape| shape.matches(&consumer_plan))
        .expect("the composed consumer scope must remain in the proof");
    let occurrences = incomplete.reservation_shapes[consumer_shape_index].producer_occurrences().to_vec();
    let mut consumer_shape = AttemptReservationShape::for_query_plan(&consumer_plan);
    consumer_shape
        .set_producer_occurrences(occurrences)
        .expect("the existing producer manifest remains valid without ordered rows");
    incomplete.reservation_shapes[consumer_shape_index] = consumer_shape;
    let fallback = QueryPlanCapabilityProof { requirement: incomplete }
        .resolve_reservation_envelope(reservation_policy())
        .expect("the complete owner inventory must provide the conservative fallback");
    assert_eq!(fallback.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
}

#[test]
fn incomplete_ordered_metadata_falls_back_to_rebound_conservative_owner_terms() {
    let consumer_plan = graph_output_union_consumer(&["only"]);
    let proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(
            Some("only".to_owned()),
            None,
            Some(graph_bfs_proof_with_typed_output(0)),
        )],
    )
    .expect("complete typed proof can conservatively compose without ordered metadata");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn graph_result_global_aggregate_supports_ordered_and_conservative_owner_bounds() {
    let consumer_plan = graph_output_global_aggregate_consumer();
    let combined = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("the aggregate consumer must issue a composed proof"),
        [ComposedGraphProducerCandidate::new(
            Some("only".to_owned()),
            Some("only".to_owned()),
            Some(graph_bfs_proof_with_typed_output(0)),
        )],
    )
    .expect("the graph-result global aggregate must compose before admission");
    assert!(combined.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Aggregate(_)) && term.id.role == OwnerTermRole::AggregatePartialState
    }));
    assert!(combined.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Aggregate(_)) && term.id.role == OwnerTermRole::AggregateOutput
    }));
    assert!(combined.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Consumer(_)) && term.id.role == OwnerTermRole::ConsumerOutput
    }));
    let ordered = combined
        .clone()
        .resolve_reservation_envelope(reservation_policy())
        .expect("the global aggregate owner terms must resolve an ordered bound");
    assert_eq!(ordered.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));

    let mut incomplete = combined.into_requirement();
    let consumer_shape_index = incomplete
        .reservation_shapes
        .iter()
        .position(|shape| shape.matches(&consumer_plan))
        .expect("the aggregate consumer scope must remain in the proof");
    let occurrences = incomplete.reservation_shapes[consumer_shape_index].producer_occurrences().to_vec();
    let mut consumer_shape = AttemptReservationShape::for_query_plan(&consumer_plan);
    consumer_shape
        .set_producer_occurrences(occurrences)
        .expect("the producer manifest remains valid without ordered rows");
    incomplete.reservation_shapes[consumer_shape_index] = consumer_shape;
    let conservative = QueryPlanCapabilityProof { requirement: incomplete }
        .resolve_reservation_envelope(reservation_policy())
        .expect("the aggregate owner inventory must support the conservative fallback");
    assert_eq!(conservative.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
}

#[test]
fn graph_result_global_count_and_nunique_use_the_native_distinct_workspace_owner() {
    let consumer_plan = graph_output_global_nunique_consumer(DataType::Int64, false, false);
    let producer = graph_bfs_proof_with_typed_output_and_rows(0, 64);
    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
            .expect("bounded global NUnique must issue a composed consumer proof"),
        [("only".to_owned(), producer)],
    )
    .expect("one non-null fixed-width global NUnique must compose");
    let requirement = combined.as_requirement();
    let workspace = requirement
        .owner_terms
        .iter()
        .find(|term| term.id.role == OwnerTermRole::AggregateDistinctWorkspace)
        .expect("the native distinct workspace has one explicit owner term");
    assert!(workspace.bytes() > 0);
    let shape = requirement
        .reservation_shapes
        .iter()
        .find(|shape| shape.matches(&consumer_plan))
        .expect("the consumer scope is retained")
        .ordered_schedule()
        .expect("the mixed global aggregate has an ordered schedule");
    let groups = shape.consumer_peak_term_groups();
    assert!(groups.iter().any(|group| group.contains(workspace.id())));
    assert!(groups.iter().any(|group| {
        group.iter().any(|term| term.role == OwnerTermRole::AggregatePartialState)
            && group.iter().any(|term| term.role == OwnerTermRole::AggregateOutput)
            && !group.contains(workspace.id())
    }));
    let envelope =
        combined.resolve_reservation_envelope(reservation_policy()).expect("the exact NUnique owner schedule resolves");
    let peak = envelope.complete_phase_peak_bytes().expect("the exact NUnique owner schedule has one complete peak");
    let error = envelope
        .validate_composed_execution(&consumer_plan, &["only".to_owned()], peak - 1)
        .expect_err("a grant one byte below the exact NUnique peak must reject before execution");
    assert_eq!(error.fact_value("schedule_reason"), Some("immutable_grant_below_schedule_peak"));
    assert_eq!(error.fact_value("immutable_grant_bytes"), Some((peak - 1).to_string().as_str()));
    assert_eq!(error.fact_value("complete_phase_peak_bytes"), Some(peak.to_string().as_str()));
}

#[test]
fn graph_result_nunique_keeps_grouped_nullable_and_variable_width_shapes_unmodeled() {
    for (data_type, nullable, grouped, expected_reason) in [
        (DataType::Int64, false, true, "consumer_aggregate_state_unmodeled"),
        (DataType::Int64, true, false, "consumer_aggregate_nunique_owner_preflight_unavailable"),
        (DataType::Utf8, false, false, "consumer_aggregate_nunique_owner_preflight_unavailable"),
    ] {
        let consumer_plan = graph_output_global_nunique_consumer(data_type, nullable, grouped);
        let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer_plan,
            QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
                .expect("the consumer proof stays incomplete until its producer is bound"),
            [("only".to_owned(), graph_bfs_proof_with_typed_output_and_rows(0, 64))],
        )
        .expect_err("the unsupported NUnique owner shape must fail closed");
        assert_eq!(error.fact_value("schedule_reason"), Some(expected_reason));
    }
}

#[test]
fn conservative_resolver_rejects_a_non_graph_source_proof_without_retained_output_term() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let source_bytes = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    let mut producer_builder = QueryPlanBuilder::new();
    let producer_source = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/conservative-producer.parquet").with_file_size_bytes(source_bytes)],
        Arc::clone(&schema),
        vec!["value".to_owned()],
    )));
    let producer_plan = producer_builder.finish_with_host_sink(producer_source);
    let mut consumer_builder = QueryPlanBuilder::new();
    let consumer_source = consumer_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("only", Arc::clone(&schema)).with_row_count_upper_bound(1),
    ));
    let consumer_plan = consumer_builder.finish_with_host_sink(consumer_source);

    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(
            Some("only".to_owned()),
            None,
            Some(QueryPlanCapabilityProof::for_query_plan(&producer_plan).unwrap()),
        )],
    )
    .expect_err("a non-graph producer without an AttemptRetained output must reject before placement");
    assert_eq!(error.fact_value("conservative_reason"), Some("producer_retained_output_term_missing"));
}

#[test]
fn incomplete_single_graph_candidate_fails_closed_with_structured_schedule_facts() {
    let consumer_plan = graph_output_union_consumer(&["only"]);
    let incomplete_graph_proof =
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_bfs_plan_with_source_vertex(0))
            .expect("the graph plan issues its scoped capability proof before evidence is hydrated");

    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(
            Some("only".to_owned()),
            Some("only".to_owned()),
            Some(incomplete_graph_proof),
        )],
    )
    .expect_err("an incomplete N=1 candidate must not retain an unscheduled admission proof");

    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_retained_output_term_missing"));
    assert_eq!(error.fact_value("occurrence_id"), Some("only"));
}

#[test]
fn missing_single_graph_proof_fails_closed_with_structured_schedule_facts() {
    let consumer_plan = graph_output_union_consumer(&["only"]);
    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [ComposedGraphProducerCandidate::new(Some("only".to_owned()), Some("only".to_owned()), None)],
    )
    .expect_err("missing N=1 proof evidence must fail before admission");

    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
    assert_eq!(error.fact_value("occurrence_id"), Some("only"));
}

#[test]
fn incomplete_multi_graph_candidate_remains_fail_closed() {
    let consumer_plan = graph_output_union_consumer(&["first", "second"]);
    let complete_graph_proof = graph_bfs_proof_with_typed_output(0);
    let error = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [
            ComposedGraphProducerCandidate::new(
                Some("first".to_owned()),
                Some("first".to_owned()),
                Some(complete_graph_proof),
            ),
            ComposedGraphProducerCandidate::new(Some("second".to_owned()), Some("second".to_owned()), None),
        ],
    )
    .expect_err("two retained graph outputs require complete evidence");

    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_admission_proof_missing_or_ambiguous"));
}

#[test]
fn composed_consumer_alias_column_projection_reuses_its_input_allocation() {
    let consumer_plan = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Int64);
    let output = ComposedGraphProducerOutputEvidence {
        occurrence_id: "only".to_owned(),
        row_count_upper_bound: 128,
        column_evidence: PlanColumnEvidence::default(),
    };
    let (bytes, owner_terms) = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &[output])
        .expect("an alias-only projection must have a bounded zero-copy contract");
    assert_eq!(bytes, 0, "an alias-only projection is a zero-copy select/rename");
    assert!(owner_terms.is_empty(), "an alias-only projection must not mint a zero-byte owner term");

    let combined = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        QueryPlanCapabilityProof::for_query_plan(&consumer_plan).unwrap(),
        [("only".to_owned(), graph_bfs_proof_with_typed_output(0))],
    )
    .expect("an exact-type alias projection must issue complete composed evidence");
    assert!(combined.resolve_reservation_envelope(reservation_policy()).unwrap().complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_consumer_rejects_projected_type_changes_and_missing_string_payload_evidence() {
    let output = [ComposedGraphProducerOutputEvidence {
        occurrence_id: "only".to_owned(),
        row_count_upper_bound: 128,
        column_evidence: PlanColumnEvidence::default(),
    }];
    let type_changed = graph_output_projection_consumer(Expr::column("value").alias("projected"), DataType::Float64);
    let error = composed_consumer_additional_peak_bytes_for_plan(&type_changed, &output)
        .expect_err("a type-changing projection would allocate a new output frame");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_output_type_mismatch"));

    let representation_cast = graph_output_projection_consumer_with_input(
        DataType::Utf8,
        Expr::cast(Expr::column("value"), DataType::Utf8View),
        DataType::Utf8View,
    );
    let error = composed_consumer_additional_peak_bytes_for_plan(&representation_cast, &output)
        .expect_err("a string representation cast without source-coupled payload evidence must fail closed");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_string_payload_evidence_missing"));
    assert_eq!(error.fact_value("column"), Some("projected"));

    let consumer_plan =
        graph_output_projection_consumer(Expr::cast(Expr::column("value"), DataType::Float64), DataType::Int64);
    let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &output)
        .expect_err("an unmodeled cast must remain rejected");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_projection_expression_workspace_unmodeled"));
}

#[test]
fn composed_consumer_case_projection_uses_cudf_owner_preflight_group() {
    let plan = graph_output_case_projection_consumer();
    let output = ComposedGraphProducerOutputEvidence {
        occurrence_id: "only".to_owned(),
        row_count_upper_bound: 128,
        column_evidence: PlanColumnEvidence::default(),
    };
    let outputs = HashMap::from([("only", &output)]);
    let mut used_occurrences = HashSet::new();
    let mut memo = HashMap::new();
    let column_evidence = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new())
        .expect("CASE projection payload evidence analysis must succeed");
    let bound = composed_consumer_branch_bound(
        &plan,
        plan.root_id(),
        &outputs,
        &mut used_occurrences,
        &[],
        &column_evidence,
        &mut memo,
    )
    .expect("supported asset-score CASE must use the cudf owner preflight");

    let workspace = bound
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::ConsumerProjectionExpressionWorkspace)
        .expect("CASE preflight must expose one expression workspace owner term");
    let output_term = bound
        .owner_terms
        .iter()
        .find(|term| term.id().role == OwnerTermRole::ConsumerOutput)
        .expect("CASE projection must retain its final output frame");
    assert_eq!(workspace.bytes(), 128 + (128 * 8 + 64));
    assert!(
        bound
            .peak_owner_term_groups
            .iter()
            .any(|group| group.contains(workspace.id()) && group.contains(output_term.id()))
    );
}

#[test]
fn composed_graph_schedule_prices_nullable_keys_and_rejects_missing_string_payload_evidence() {
    let nullable_plan = graph_output_inner_join_consumer(DataType::Int64, true, false);
    QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &nullable_plan,
        QueryPlanCapabilityProof::for_query_plan(&nullable_plan).unwrap(),
        [
            ("left".to_owned(), graph_bfs_proof_with_typed_output(0)),
            ("right".to_owned(), graph_bfs_proof_with_typed_output(1)),
        ],
    )
    .expect("nullable fixed-width keys must use the owner preflight null-mask component");

    for data_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
        let string_plan = graph_output_inner_join_consumer(data_type, false, false);
        let error = composed_consumer_additional_peak_bytes_for_plan(
            &string_plan,
            &[
                ComposedGraphProducerOutputEvidence {
                    occurrence_id: "left".to_owned(),
                    row_count_upper_bound: 16,
                    column_evidence: PlanColumnEvidence::default(),
                },
                ComposedGraphProducerOutputEvidence {
                    occurrence_id: "right".to_owned(),
                    row_count_upper_bound: 8,
                    column_evidence: PlanColumnEvidence::default(),
                },
            ],
        )
        .expect_err("string key without payload evidence must fail closed before fixed-width hash preflight");
        assert_eq!(error.fact_value("schedule_reason"), Some("consumer_join_variable_width_key"));
        assert!(error.fact_value("node_id").is_some());
        assert_eq!(error.fact_value("left_rows"), Some("16"));
        assert_eq!(error.fact_value("right_rows"), Some("8"));
        assert_eq!(error.fact_value("column"), Some("vertex"));
    }
}

#[test]
fn production_shaped_string_lookup_caps_payload_without_row_multiplication() {
    let consumer_plan = graph_output_inner_join_consumer_with_contract(DataType::Utf8, false, false, true);
    let endpoint_lineages = BTreeSet::from([
        crate::admission::string_payload::StringPayloadLineage::Parquet {
            stable_id: "production-edges".to_owned(),
            physical_column_index: 0,
            selected_row_groups: (0..36).collect(),
        },
        crate::admission::string_payload::StringPayloadLineage::Parquet {
            stable_id: "production-edges".to_owned(),
            physical_column_index: 1,
            selected_row_groups: (0..36).collect(),
        },
    ]);
    let column_evidence = PlanColumnEvidence {
        string_payload_bounds: std::collections::BTreeMap::from([(
            "vertex".to_owned(),
            crate::admission::string_payload::StringPayloadBound {
                total_payload_upper_bound_bytes: Some(4_706_637_611),
                max_single_value_length_bytes: None,
                lineages: endpoint_lineages,
            },
        )]),
        unique_keys: vec![BTreeSet::from(["vertex".to_owned()])],
    };
    let producer_outputs = [
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "left".to_owned(),
            row_count_upper_bound: 109_914_910,
            column_evidence: column_evidence.clone(),
        },
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "right".to_owned(),
            row_count_upper_bound: 109_914_910,
            column_evidence,
        },
    ];

    let (additional_peak_bytes, owner_terms) =
        composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &producer_outputs)
            .expect("footer-coupled payload evidence must price the production-shaped string lookup");
    assert!(owner_terms.iter().any(|term| term.id().role == OwnerTermRole::ConsumerHashTable));
    assert!(additional_peak_bytes < 1_u64 << 40, "the composed lookup must remain below one TiB");
    assert!(owner_terms.iter().all(|term| term.bytes() < 1_u64 << 40));
}

fn pagerank_like_string_producer() -> QueryPlanCapabilityProof {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let file = LocalParquetFile::new("/tmp/pagerank-like-string-output.parquet")
        .with_row_groups(vec![0])
        .with_row_group_rows(vec![64])
        .with_row_group_bytes(vec![4_096])
        .with_projected_row_group_bytes(vec![4_096], vec![4_096])
        .with_variable_width_payload_evidence(std::collections::BTreeMap::from([(
            0,
            crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                0,
                vec![0],
                1_024,
                crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
            ),
        )]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        Arc::clone(&schema),
        vec!["vertex".to_owned(), "value".to_owned()],
    )));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            vec![Expr::column("vertex")],
            vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("value"), "value")],
        ),
        schema,
    );
    let plan = builder.finish_gpu_resident(aggregate);
    QueryPlanCapabilityProof::for_query_plan(&plan).expect("PageRank-like output must issue a finite producer proof")
}

fn graph_metadata_join_consumer(
    join_type: JoinType,
    bounded_side: Option<JoinSide>,
    top_k_before_enrichment: bool,
    hydrated: bool,
) -> QueryPlan {
    graph_metadata_join_consumer_with_rows(join_type, bounded_side, top_k_before_enrichment, hydrated, 96)
}

fn graph_metadata_join_consumer_with_rows(
    join_type: JoinType,
    bounded_side: Option<JoinSide>,
    top_k_before_enrichment: bool,
    hydrated: bool,
    metadata_rows: u64,
) -> QueryPlan {
    let graph_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let metadata_schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexName", DataType::Utf8, false),
        Field::new("assetType", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let metadata_file = LocalParquetFile::new("/tmp/graph-metadata.parquet");
    let metadata_file = if hydrated {
        metadata_file
            .with_row_groups(vec![0])
            .with_row_group_rows(vec![metadata_rows])
            .with_row_group_bytes(vec![metadata_rows.saturating_mul(128)])
            .with_projected_row_group_bytes(
                vec![metadata_rows.saturating_mul(128)],
                vec![metadata_rows.saturating_mul(128)],
            )
            .with_variable_width_payload_evidence(std::collections::BTreeMap::from([
                (
                    0,
                    crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                        0,
                        vec![0],
                        metadata_rows.saturating_mul(40),
                        crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                    ),
                ),
                (
                    1,
                    crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                        1,
                        vec![0],
                        metadata_rows.saturating_mul(64),
                        crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                    ),
                ),
                (
                    2,
                    crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                        2,
                        vec![0],
                        metadata_rows.saturating_mul(16),
                        crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                    ),
                ),
            ]))
    } else {
        metadata_file
    };
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("vertexName", DataType::Utf8, false),
        Field::new("assetType", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let graph =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("pagerank", Arc::clone(&graph_schema))));
    let graph = if top_k_before_enrichment {
        let sort = builder.add_sort(
            graph,
            SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
            Arc::clone(&graph_schema),
        );
        builder.add_limit(sort, LimitSpec::fetch(20), Arc::clone(&graph_schema))
    } else {
        graph
    };
    let metadata = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![metadata_file],
        metadata_schema,
        vec!["vertexId".to_owned(), "vertexName".to_owned(), "assetType".to_owned(), "vertexPriority".to_owned()],
    )));
    let mut join_spec = JoinSpec::new(
        join_type,
        vec![JoinKey::new(
            crate::expr::ColumnRef::input(0, "vertex", "vertex"),
            crate::expr::ColumnRef::input(1, "vertexId", "vertexId"),
        )],
    )
    .with_build_side(JoinSide::Right)
    .with_output_projection(vec![
        JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "vertex", "vertex"), "vertex"),
        JoinOutputColumn::left(crate::expr::ColumnRef::input(0, "value", "value"), "value"),
        JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "vertexName", "vertexName"), "vertexName"),
        JoinOutputColumn::right(crate::expr::ColumnRef::input(1, "assetType", "assetType"), "assetType"),
        JoinOutputColumn::right(
            crate::expr::ColumnRef::input(1, "vertexPriority", "vertexPriority"),
            "vertexPriority",
        ),
    ]);
    if let Some(bounded_side) = bounded_side {
        join_spec = join_spec.with_at_most_one_match_per_bounded_row(bounded_side);
    }
    let join = builder.add_join(graph, metadata, join_spec, Arc::clone(&output_schema));
    if top_k_before_enrichment {
        return builder.finish_with_host_sink(join);
    }
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("value"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(20), output_schema);
    builder.finish_with_host_sink(limit)
}

#[test]
fn graph_to_metadata_inner_join_uses_bounded_key_payload_evidence() {
    let producer = pagerank_like_string_producer();
    let inner = graph_metadata_join_consumer(JoinType::Inner, Some(JoinSide::Right), false, true);
    let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &inner,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&inner)
            .expect("mixed LocalParquet and graph consumer must issue a structural proof"),
        [("pagerank".to_owned(), producer)],
    )
    .expect("the bounded metadata key must prove the equal inner-join output key payload");
    assert!(proof.resolve_reservation_envelope(reservation_policy()).is_ok());

    let left = graph_metadata_join_consumer(JoinType::Left, None, false, true);
    let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &left,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&left)
            .expect("mixed LocalParquet and graph consumer must issue a structural proof"),
        [("pagerank".to_owned(), pagerank_like_string_producer())],
    )
    .expect("finite row and payload bounds must prove a non-unique left enrichment without claiming uniqueness");
    proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("the finite non-unique left enrichment must retain an ordered envelope");
}

#[test]
fn graph_to_metadata_left_join_uses_uniqueness_only_for_the_tight_bound() {
    for top_k_before_enrichment in [true, false] {
        let qualified =
            graph_metadata_join_consumer(JoinType::Left, Some(JoinSide::Left), top_k_before_enrichment, true);
        let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &qualified,
            QueryPlanCapabilityProof::for_composed_consumer_plan(&qualified).unwrap(),
            [("pagerank".to_owned(), pagerank_like_string_producer())],
        )
        .expect("authoritative metadata uniqueness must bound each left graph row to one match");
        proof
            .resolve_reservation_envelope(reservation_policy())
            .expect("qualified left enrichment must resolve a finite reservation envelope");
        crate::admission::bounded_pipeline_admission(&qualified)
            .expect("qualified left enrichment capability admission must not error")
            .expect("qualified left enrichment must select a compiled bounded pipeline");

        let unqualified = graph_metadata_join_consumer(JoinType::Left, None, top_k_before_enrichment, true);
        let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &unqualified,
            QueryPlanCapabilityProof::for_composed_consumer_plan(&unqualified).unwrap(),
            [("pagerank".to_owned(), pagerank_like_string_producer())],
        )
        .expect("finite cross-product evidence must bound an unqualified lookup without treating it as unique");
        proof
            .resolve_reservation_envelope(reservation_policy())
            .expect("the conservative non-unique lookup must retain an ordered envelope");
    }
}

#[test]
fn production_shaped_metadata_joins_retain_finite_and_authoritative_bounds() {
    let mut output_bytes = Vec::new();
    for (join_type, bounded_side) in
        [(JoinType::Inner, Some(JoinSide::Right)), (JoinType::Left, Some(JoinSide::Left)), (JoinType::Left, None)]
    {
        let consumer = graph_metadata_join_consumer_with_rows(join_type, bounded_side, true, true, 10_408_932);
        let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &consumer,
            QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer)
                .expect("the production-shaped composed topology must issue a structural proof"),
            [("pagerank".to_owned(), pagerank_like_string_producer())],
        )
        .expect("10M-scale metadata must use finite multiplicity instead of missing-schedule rejection");
        let output = proof
            .as_requirement()
            .owner_terms
            .iter()
            .find(|term| {
                term.id().owner == OwnerId::Consumer(consumer.root_id())
                    && term.id().role == OwnerTermRole::ConsumerOutput
            })
            .expect("the composed join must own its output allocation");
        output_bytes.push(output.bytes());
        proof
            .resolve_reservation_envelope(reservation_policy())
            .expect("the production-shaped proof must be complete even when a device may reject its capacity floor");
    }
    assert!(output_bytes[2] > output_bytes[0]);
    assert!(output_bytes[2] > output_bytes[1], "non-unique LEFT cannot reuse the authoritative one-match bound");
}

#[test]
fn graph_to_metadata_planning_defers_footer_dependent_admission_until_hydration() {
    let planning = graph_metadata_join_consumer(JoinType::Inner, Some(JoinSide::Right), false, false);
    let structural = QueryPlanCapabilityProof::for_composed_consumer_plan(&planning)
        .expect("planning must accept the supported mixed graph/metadata topology without footer I/O");
    let error = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &planning,
        structural,
        [("pagerank".to_owned(), pagerank_like_string_producer())],
    )
    .expect_err("an unhydrated metadata source cannot promise a live device grant");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_local_source_owner_missing"));

    let prepared = graph_metadata_join_consumer(JoinType::Inner, Some(JoinSide::Right), false, true);
    let proof = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &prepared,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&prepared).unwrap(),
        [("pagerank".to_owned(), pagerank_like_string_producer())],
    )
    .expect("hydrated source evidence must complete the same engine-owned proof");
    proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("prepared proof must resolve a finite reservation envelope");
}

#[test]
fn composed_consumer_rejects_join_output_row_overflow() {
    let consumer_plan = graph_output_inner_join_consumer(DataType::Int64, false, false);
    let producer_outputs = [
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "left".to_owned(),
            row_count_upper_bound: u64::MAX,
            column_evidence: PlanColumnEvidence::default(),
        },
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "right".to_owned(),
            row_count_upper_bound: 2,
            column_evidence: PlanColumnEvidence::default(),
        },
    ];
    let error = composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &producer_outputs)
        .expect_err("overflowing inner-join output rows cannot become a finite admission contract");
    assert_eq!(error.fact_value("schedule_reason"), Some("consumer_join_output_rows_overflow"));
}

#[test]
fn composed_consumer_prices_large_unique_lookup_without_cartesian_pair_indices() {
    let consumer_plan = graph_output_inner_join_consumer_with_contract(DataType::Int64, false, false, true);
    let producer_outputs = [
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "left".to_owned(),
            row_count_upper_bound: 48_322_510,
            column_evidence: PlanColumnEvidence::default(),
        },
        ComposedGraphProducerOutputEvidence {
            occurrence_id: "right".to_owned(),
            row_count_upper_bound: 96_645_020,
            column_evidence: PlanColumnEvidence::default(),
        },
    ];
    let (additional_peak_bytes, owner_terms) =
        composed_consumer_additional_peak_bytes_for_plan(&consumer_plan, &producer_outputs)
            .expect("unique lookup contract must keep the composed join schedule finite");

    assert!(additional_peak_bytes > 0);
    assert!(owner_terms.iter().any(|term| term.id().role == OwnerTermRole::ConsumerPairIndices));
}

#[test]
fn composed_consumer_uses_the_linked_cudf_metadata_hash_preflight() {
    let keys = [cudf::ops::join::HashJoinKeyMetadata::fixed_width(cudf::TypeId::Int64, false)];
    let preflight = cudf::ops::join::hash_join_allocation_preflight_for_metadata(
        &keys,
        10,
        0.5,
        cudf::ops::join::HashJoinPreprocessingMode::FlatKeys,
    )
    .expect("the linked cuDF metadata owner preflight must accept a fixed-width non-null key");
    assert!(preflight.retained_bytes() > 0);
}

fn graph_result_sort_plan(data_type: DataType, nullable: bool, row_bound: Option<u64>, key_count: usize) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, nullable)]));
    let mut builder = QueryPlanBuilder::new();
    let mut source = GpuResidentSource::new("graph_result", Arc::clone(&schema));
    if let Some(row_bound) = row_bound {
        source = source.with_row_count_upper_bound(row_bound);
    }
    let source = builder.add_source(SourceNode::GpuResident(source));
    let expressions = (0..key_count).map(|_| SortExpr::new(Expr::column("value"), false, false)).collect();
    let sort = builder.add_sort(source, SortSpec::new(expressions), schema);
    builder.finish_with_host_sink(sort)
}

fn graph_result_sort_limit_plan(row_bound: u64, skip: usize, fetch: usize) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("distance", DataType::Int64, false),
        Field::new("predecessor", DataType::Int64, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("graph_result", Arc::clone(&schema)).with_row_count_upper_bound(row_bound),
    ));
    let sort = builder.add_sort(
        source,
        SortSpec::new(vec![SortExpr::new(Expr::column("distance"), false, false)]),
        Arc::clone(&schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::new(skip, Some(fetch)), schema);
    builder.finish_with_host_sink(limit)
}

fn graph_result_filter_sort_plan(row_bound: Option<u64>) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let mut source = GpuResidentSource::new("graph_result", Arc::clone(&schema));
    if let Some(row_bound) = row_bound {
        source = source.with_row_count_upper_bound(row_bound);
    }
    let source = builder.add_source(SourceNode::GpuResident(source));
    let filter = builder.add_filter(source, Expr::column("value"), Arc::clone(&schema));
    let sort =
        builder.add_sort(filter, SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]), schema);
    builder.finish_with_host_sink(sort)
}

#[test]
fn supported_graph_result_sort_resolves_exact_owner_components_per_device() {
    let plan = graph_result_sort_plan(DataType::Int32, false, Some(128), 1);
    let proof = QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
        .expect("supported sort metadata must issue a typed proof");
    let resolved = proof
        .requirement_for_device_with_preflight(7, |descriptor, ordinal| {
            assert_eq!(ordinal, 7);
            assert_eq!(descriptor.num_rows, 128);
            Ok(SortRadixPreflight::new_for_test(128, 512, 1_024, 512))
        })
        .expect("fake owner preflight resolves without CUDA");
    let terms = resolved
        .as_requirement()
        .owner_terms
        .iter()
        .filter(|term| matches!(term.id.owner, OwnerId::Sort(_)))
        .collect::<Vec<_>>();
    assert_eq!(terms.len(), 7);
    assert_eq!(
        terms
            .iter()
            .find(|term| term.id.role == OwnerTermRole::SortKeyInput)
            .expect("sort key input owner must be present")
            .bytes(),
        0
    );
    assert!(terms.iter().filter(|term| term.id.role != OwnerTermRole::SortKeyInput).all(|term| term.bytes() > 0));
    assert!(
        terms.iter().all(|term| term.id.execution_scope_fingerprint == canonical_execution_scope_fingerprint(&plan))
    );
}

#[test]
fn floating_sort_keys_issue_key_input_owner_terms() {
    for data_type in [DataType::Float32, DataType::Float64] {
        let plan = graph_result_sort_plan(data_type, false, Some(128), 1);
        let proof =
            QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
                .expect("floating radix sort metadata must be accepted");
        let resolved = proof
            .requirement_for_device_with_preflight(0, |_, _| {
                Ok(SortRadixPreflight::new_for_test_with_key_input(512, 128, 256, 512, 256))
            })
            .expect("floating sort owner preflight must resolve");
        let key_input = resolved
            .as_requirement()
            .owner_terms
            .iter()
            .find(|term| term.id.role == OwnerTermRole::SortKeyInput)
            .expect("floating sort key input owner must be present");
        assert_eq!(key_input.bytes(), 512);
    }
}

#[test]
fn graph_result_sort_limit_uses_alternative_consumer_peak_groups() {
    let cases = [
        (
            SortRadixPreflight::new_for_test_with_key_input(1_000, 1_000, 1_000, 1_000, 1_000),
            LimitRangeGatherPreflight::new_for_test(1, 1, 1, 1),
        ),
        (
            SortRadixPreflight::new_for_test_with_key_input(1, 1, 1, 1, 1),
            LimitRangeGatherPreflight::new_for_test(1_000, 1_000, 1_000, 1_000),
        ),
    ];
    for (sort_preflight, limit_preflight) in cases {
        let plan = graph_result_sort_limit_plan(100, 7, 2);
        let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &plan,
            "graph_result",
            &graph_bfs_proof_with_typed_output_and_rows(0, 50),
        )
        .expect("bounded sort-limit consumer must issue a proof");
        let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
            &plan,
            consumer,
            [("graph_result".to_owned(), graph_bfs_proof_with_typed_output_and_rows(0, 50))],
        )
        .expect("sort-limit consumer must bind one graph producer");
        let resolved = composed
            .requirement_for_device_with_preflight_and_limit(
                0,
                |descriptor, _| {
                    assert_eq!(descriptor.num_rows, 100);
                    Ok(sort_preflight)
                },
                |descriptor, _| {
                    assert_eq!(descriptor.input_rows, 100);
                    assert_eq!(descriptor.output_rows, 2);
                    assert_eq!(
                        descriptor.output_columns,
                        vec![
                            (cudf::DataType::new(cudf::TypeId::Int64), false),
                            (cudf::DataType::new(cudf::TypeId::Int64), false),
                            (cudf::DataType::new(cudf::TypeId::Int64), true),
                        ]
                    );
                    Ok(limit_preflight)
                },
            )
            .expect("sort and limit owner preflights must resolve");
        let shape = resolved
            .as_requirement()
            .reservation_shapes
            .iter()
            .find(|shape| shape.matches(&plan))
            .expect("consumer reservation scope must remain attached");
        let schedule = shape.ordered_schedule().expect("consumer schedule must be attached");
        let groups = schedule.consumer_peak_term_groups();
        assert_eq!(groups.len(), 2);
        let owner_terms = &resolved.as_requirement().owner_terms;
        let sum_group = |group: &[OwnerTermId]| {
            group
                .iter()
                .map(|term_id| {
                    owner_terms
                        .iter()
                        .find(|term| term.id() == term_id)
                        .expect("scheduled term must have an owner inventory entry")
                        .bytes()
                })
                .sum::<u64>()
        };
        let group_a = sum_group(&groups[0]);
        let group_b = sum_group(&groups[1]);
        assert_eq!(schedule.consumer_peak_bytes_for_test(owner_terms).unwrap(), group_a.max(group_b));
        assert_ne!(group_a, group_b, "the fixtures must exercise both peak alternatives");
        assert!(
            groups[1].iter().all(|term_id| !matches!(
                term_id.role,
                OwnerTermRole::SortKeyInput
                    | OwnerTermRole::SortKeyOutput
                    | OwnerTermRole::SortSequence
                    | OwnerTermRole::SortTemporaryWorkspace
                    | OwnerTermRole::SortRetainedOrder
            )),
            "the limit phase retains only the full sorted frame"
        );
        let sort_frame = owner_terms
            .iter()
            .find(|term| matches!(term.id.owner, OwnerId::Sort(_)) && term.id.role == OwnerTermRole::SortOutputFrame)
            .expect("full sorted frame owner must be present");
        let limit_frame = owner_terms
            .iter()
            .find(|term| matches!(term.id.owner, OwnerId::Limit(_)) && term.id.role == OwnerTermRole::LimitOutputFrame)
            .expect("bounded limit output frame owner must be present");
        assert_ne!(sort_frame.id(), limit_frame.id());
        assert_ne!(sort_frame.bytes(), limit_frame.bytes());
        let limit_null_mask = owner_terms
            .iter()
            .filter(|term| matches!(term.id.owner, OwnerId::Limit(_)))
            .filter(|term| term.id.role == OwnerTermRole::LimitOutputNullMask)
            .collect::<Vec<_>>();
        assert_eq!(limit_null_mask.len(), 1);
        assert_eq!(limit_null_mask[0].bytes(), limit_preflight.output_null_mask_bytes);
        let limit_workspace = owner_terms
            .iter()
            .filter(|term| matches!(term.id.owner, OwnerId::Limit(_)))
            .filter(|term| term.id.role == OwnerTermRole::LimitNativeWorkspace)
            .collect::<Vec<_>>();
        assert_eq!(limit_workspace.len(), 1);
        assert_eq!(limit_workspace[0].bytes(), limit_preflight.native_workspace_bytes);
    }
}

#[test]
fn composed_graph_result_sort_resolves_through_the_production_schedule_seam() {
    let consumer_plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
    let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &consumer_plan,
        "graph_result",
        &graph_bfs_proof_with_typed_output(0),
    )
    .expect("supported sort metadata must issue a typed consumer proof");
    let composed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [ComposedGraphProducerCandidate::new(
            Some("graph_result".to_owned()),
            Some("graph_result".to_owned()),
            Some(graph_bfs_proof_with_typed_output(0)),
        )],
    )
    .expect("one graph producer and its supported sort consumer must compose");
    let resolved = composed
        .requirement_for_device_with_preflight(0, |descriptor, _| {
            assert!(descriptor.num_rows > 0);
            Ok(SortRadixPreflight::new_for_test(128, 256, 512, 256))
        })
        .expect("the candidate device preflight must hydrate sort owner terms");
    let sort_owner_terms =
        resolved.as_requirement().owner_terms.iter().filter(|term| matches!(term.id.owner, OwnerId::Sort(_))).count();
    assert_eq!(sort_owner_terms, 7);
    let envelope = resolved
        .resolve_reservation_envelope(reservation_policy())
        .expect("the composed sort proof must resolve a finite ordered envelope");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    assert!(envelope.complete_phase_peak_bytes().is_some());
}

#[test]
fn composed_filter_then_sort_references_filter_and_sort_owner_terms() {
    let consumer_plan = graph_result_filter_sort_plan(None);
    let producer = graph_bfs_proof_with_typed_output(0);
    let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &consumer_plan,
        "graph_result",
        &producer,
    )
    .expect("Filter then one-key integral Sort has a bounded consumer proof");
    let composed = QueryPlanCapabilityProof::combine_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [("graph_result".to_owned(), producer)],
    )
    .expect("Filter then Sort must consume one producer exactly once");
    let resolved = composed
        .requirement_for_device_with_preflight(0, |descriptor, _| {
            assert!(descriptor.num_rows > 0);
            Ok(SortRadixPreflight::new_for_test(64, 128, 256, 128))
        })
        .expect("the existing cuDF sort owner preflight resolves the consumer terms");
    assert!(resolved.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Consumer(_)) && term.id.role == OwnerTermRole::ConsumerPredicateMask
    }));
    assert!(resolved.as_requirement().owner_terms.iter().any(|term| {
        matches!(term.id.owner, OwnerId::Sort(_)) && term.id.role == OwnerTermRole::SortTemporaryWorkspace
    }));
    let envelope = resolved
        .resolve_reservation_envelope(reservation_policy())
        .expect("Filter and Sort owner terms share one ordered consumer inventory");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
}

#[test]
fn graph_result_sort_uses_the_producer_owned_row_bound_when_consumer_source_is_unbounded() {
    let plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
    let producer_plan = graph_bfs_plan();
    let producer_evidence =
        graph_memory_bounded_admission_evidence_for_plan(&producer_plan, GraphMemoryCalibrationOverride::none(), None)
            .unwrap();
    let expected_rows = usize::try_from(producer_evidence.retained_output_rows_upper_bound()).unwrap();
    let producer = graph_bfs_proof_with_typed_output(0);
    let proof = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &plan,
        "graph_result",
        &producer,
    )
    .expect("producer-owned output rows must complete the bounded sort proof");
    let resolved = proof
        .requirement_for_device_with_preflight(3, |descriptor, ordinal| {
            assert_eq!(ordinal, 3);
            assert_eq!(descriptor.num_rows, expected_rows);
            Ok(SortRadixPreflight::new_for_test(64, 128, 256, 128))
        })
        .expect("producer-owned row bound must reach candidate preflight");
    assert!(
        resolved.as_requirement().owner_terms.iter().any(|term| {
            matches!(term.id.owner, OwnerId::Sort(_)) && term.id.role == OwnerTermRole::SortOutputFrame
        })
    );
}

#[test]
fn graph_result_sort_producer_row_bound_failures_remain_sticky() {
    let plan = graph_result_sort_plan(DataType::Int32, false, None, 1);
    let ordinary = QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).unwrap();
    let missing = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &plan,
        "graph_result",
        &ordinary,
    )
    .unwrap();
    let missing_error = missing.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(missing_error.fact_value("sort_rejection_reason"), Some("sort_owner_producer_graph_scope_missing"));

    let ambiguous_producer =
        QueryPlanCapabilityProof::combine(graph_bfs_proof_with_typed_output(0), [graph_bfs_proof_with_typed_output(1)])
            .unwrap();
    let ambiguous = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &plan,
        "graph_result",
        &ambiguous_producer,
    )
    .unwrap();
    let ambiguous_error = ambiguous.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(ambiguous_error.fact_value("sort_rejection_reason"), Some("sort_owner_producer_graph_scope_ambiguous"));

    let mismatched_plan = graph_result_sort_plan(DataType::Int32, false, Some(2), 1);
    let mismatched = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
        &mismatched_plan,
        "graph_result",
        &graph_bfs_proof_with_typed_output(0),
    )
    .unwrap();
    let mismatch_error = mismatched.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(mismatch_error.fact_value("sort_rejection_reason"), Some("sort_owner_graph_result_row_bound_mismatch"));
}

#[test]
fn composed_resolver_preserves_sticky_sort_rejection_until_envelope_resolution() {
    let graph_producer = graph_bfs_proof_with_typed_output(0);
    let cases = [
        (
            graph_result_sort_plan(DataType::Utf8, false, None, 1),
            graph_producer.clone(),
            "sort_owner_key_type_not_radix_dispatchable",
        ),
        (
            graph_result_sort_plan(DataType::Int32, false, None, 1),
            QueryPlanCapabilityProof::for_query_plan(&ordinary_native_plan()).unwrap(),
            "sort_owner_producer_graph_scope_missing",
        ),
    ];
    for (consumer_plan, producer_for_sort, expected_sort_reason) in cases {
        let consumer = QueryPlanCapabilityProof::for_attested_graph_result_bounded_consumer_plan_with_producer(
            &consumer_plan,
            "graph_result",
            &producer_for_sort,
        )
        .expect("the attested sort shape must retain a typed rejection");
        let composed = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
            &consumer_plan,
            consumer,
            [ComposedGraphProducerCandidate::new(
                Some("graph_result".to_owned()),
                Some("graph_result".to_owned()),
                Some(graph_producer.clone()),
            )],
        )
        .expect("sticky sort rejection remains inside the production proof until envelope resolution");
        let error = composed
            .resolve_reservation_envelope(reservation_policy())
            .expect_err("unsupported sort must fail before an admission envelope is issued");
        assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
        assert_eq!(error.fact_value("schedule_reason"), Some("sort_metadata_rejected"));
        assert_eq!(error.fact_value("sort_rejection_reason"), Some(expected_sort_reason));
    }
}

#[test]
fn unsupported_graph_result_sort_shapes_reject_before_preflight() {
    for (plan, reason) in [
        (graph_result_sort_plan(DataType::Utf8, false, Some(8), 1), "sort_owner_key_type_not_radix_dispatchable"),
        (graph_result_sort_plan(DataType::Int32, true, Some(8), 1), "sort_owner_nullable_key_unsupported"),
        (graph_result_sort_plan(DataType::Int32, false, Some(8), 2), "sort_owner_sort_requires_exactly_one_key"),
        (graph_result_sort_plan(DataType::Int32, false, None, 1), "sort_owner_row_bound_missing"),
    ] {
        let error =
            QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result")
                .expect_err("unsupported sort shape must reject before native preflight");
        assert_eq!(error.fact_value("reason"), Some(reason));
    }
}

#[test]
fn stable_sort_metadata_rejects_before_the_fake_native_preflight() {
    let plan = graph_result_sort_plan(DataType::Int32, false, Some(8), 1);
    let mut proof =
        QueryPlanCapabilityProof::try_for_attested_graph_result_bounded_consumer_plan(&plan, "graph_result").unwrap();
    proof.requirement.reservation_shapes.first_mut().unwrap().set_sort_stable_for_test(true);
    let error = proof
        .requirement_for_device_with_preflight(0, |_, _| {
            panic!("unsupported stable sort must not call native preflight")
        })
        .expect_err("stable sort metadata must reject before preflight");
    assert_eq!(error.fact_value("reason"), Some("sort_owner_stable_sort_unsupported"));
}
