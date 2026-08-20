use std::{collections::HashMap, sync::Arc};

use arrow::datatypes::{Field, Schema};

use super::*;
#[cfg(feature = "cugraph")]
use crate::plan::{
    GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphEdgePredicateInput,
    GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams, GraphPredicateInput,
};
use crate::{
    expr::{AggregateSpec, ColumnRef, Expr, LiteralValue},
    plan::{
        GraphBfsParams, GraphBfsPathOutput, GraphConstructionOptions, GraphForceAtlas2Params, GraphPageRankParams,
        GraphSimilarityParams, GraphSsspParams, GraphSsspSource, GraphTriangleCountAllParams, JoinKey, JoinSpec,
        QueryPlanBuilder,
    },
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

fn edge_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]))
}

fn bounded_evidence_for_spec(spec: &GraphAlgorithmSpec, rows: u64) -> GraphMemoryAdmissionEvidence {
    bounded_evidence_for_spec_with_vertex_bound(spec, rows, None)
}

fn bounded_evidence_for_spec_with_vertex_bound(
    spec: &GraphAlgorithmSpec,
    rows: u64,
    graph_vertex_upper_bound: Option<u64>,
) -> GraphMemoryAdmissionEvidence {
    let mut builder = QueryPlanBuilder::new();
    let mut source =
        crate::source::GpuResidentSource::new("graph-test-input", edge_schema()).with_row_count_upper_bound(rows);
    if let Some(graph_vertex_upper_bound) = graph_vertex_upper_bound {
        source = source.with_graph_vertex_upper_bound(graph_vertex_upper_bound);
    }
    let source = builder.add_source(SourceNode::GpuResident(source));
    let graph = builder.add_graph_algorithm(source, spec.clone());
    let plan = builder.finish_gpu_resident(graph);
    graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("bounded graph test plan must produce evidence")
}

#[test]
fn resident_row_evidence_replaces_only_the_matching_gpu_input_cardinality() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("edges", edge_schema()).with_row_count_upper_bound(100_000_000),
    ));
    let graph = builder.add_graph_algorithm(source, spec);
    let plan = builder.finish_gpu_resident(graph);
    let static_evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("the static source bound must remain modelable");
    let resident_evidence = graph_memory_bounded_admission_evidence_for_plan_with_resident_rows(
        &plan,
        GraphMemoryCalibrationOverride::none(),
        None,
        &HashMap::from([("edges".to_owned(), 10)]),
    )
    .expect("the matching resident frame supplies exact runtime rows");

    assert!(resident_evidence.cold_model_requested_bytes() < static_evidence.cold_model_requested_bytes());
    assert!(resident_evidence.retained_output_rows_upper_bound() < static_evidence.retained_output_rows_upper_bound());
}

#[cfg(feature = "cugraph")]
fn bfs_side_input_plan<F>(side_schema: SchemaRef, side_rows: Option<u64>, configure: F) -> QueryPlan
where F: FnOnce(NodeId) -> GraphBfsParams {
    let mut builder = QueryPlanBuilder::new();
    let edge = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("bfs-side-edges", edge_schema()).with_row_count_upper_bound(20),
    ));
    let mut side = crate::source::GpuResidentSource::new("bfs-side-input", side_schema);
    if let Some(side_rows) = side_rows {
        side = side.with_row_count_upper_bound(side_rows);
    }
    let side = builder.add_source(SourceNode::GpuResident(side));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(configure(side))),
    );
    let graph = builder.add_graph_algorithm(edge, spec);
    builder.finish_gpu_resident(graph)
}

#[cfg(feature = "cugraph")]
fn personalized_pagerank_side_input_plan(side_rows: Option<u64>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let edge = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("ppr-side-edges", edge_schema()).with_row_count_upper_bound(20),
    ));
    let side_schema = Arc::new(Schema::new(vec![
        Field::new("vertex", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    let mut side = crate::source::GpuResidentSource::new("ppr-side-input", side_schema);
    if let Some(side_rows) = side_rows {
        side = side.with_row_count_upper_bound(side_rows);
    }
    let side = builder.add_source(SourceNode::GpuResident(side));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PersonalizedPageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
            pagerank: GraphPageRankParams::default(),
            personalization: GraphPageRankPersonalizationInput::new(side, "vertex", "value"),
        }),
    );
    let graph = builder.add_graph_algorithm(edge, spec);
    builder.finish_gpu_resident(graph)
}

#[cfg(feature = "cugraph")]
#[test]
fn bounded_bfs_path_evidence_prices_only_the_bounded_retained_frame() {
    let plan = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("target", DataType::Int64, false)])),
        Some(1),
        |target| {
            let mut params = GraphBfsParams::new(0, Some(6));
            params.predicates.target_vertices = Some(GraphPredicateInput::new(target, "target"));
            params.path_output = Some(GraphBfsPathOutput::new(7));
            params
        },
    );
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded BFS path must produce graph evidence");

    assert_eq!(evidence.retained_output_rows_upper_bound(), 7);
    assert_eq!(evidence.retained_output_bytes(), 7 * 40);
    assert!(evidence.warm_model_requested_bytes() > evidence.retained_output_bytes());
}

#[cfg(feature = "cugraph")]
#[test]
fn bounded_bfs_path_rejects_provably_empty_target_before_execution() {
    let plan = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("target", DataType::Int64, false)])),
        Some(0),
        |target| {
            let mut params = GraphBfsParams::new(0, Some(0));
            params.predicates.target_vertices = Some(GraphPredicateInput::new(target, "target"));
            params.path_output = Some(GraphBfsPathOutput::new(1));
            params
        },
    );
    let error = graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect_err("a provably empty path target must fail closed");
    assert_eq!(error.fact_value("side_input_reason"), Some("bfs_path_target_cardinality_invalid"));
}

#[cfg(feature = "cugraph")]
#[test]
fn bounded_bfs_path_accepts_a_target_relation_upper_bound_above_one() {
    let plan = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("target", DataType::Int64, false)])),
        Some(2),
        |target| {
            let mut params = GraphBfsParams::new(0, Some(6));
            params.predicates.target_vertices = Some(GraphPredicateInput::new(target, "target"));
            params.path_output = Some(GraphBfsPathOutput::new(7));
            params
        },
    );
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("a relation upper bound must price every possible target row");

    assert_eq!(evidence.retained_output_rows_upper_bound(), 7);
}

#[cfg(feature = "cugraph")]
#[test]
fn bounded_bfs_path_rejects_output_byte_overflow() {
    let plan = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("target", DataType::Int64, false)])),
        Some(1),
        |target| {
            let mut params = GraphBfsParams::new(0, None);
            params.predicates.target_vertices = Some(GraphPredicateInput::new(target, "target"));
            params.path_output = Some(GraphBfsPathOutput::new(u64::MAX));
            params
        },
    );
    let error = graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect_err("path byte overflow must fail closed");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("bfs_path_owner_preflight_unavailable"));
}

#[cfg(feature = "cugraph")]
fn betweenness_seed_side_input_plan(seed_row_bound: u64) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let edge = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("betweenness-side-edges", edge_schema()).with_row_count_upper_bound(20),
    ));
    let seed_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let seeds = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("betweenness-side-seeds", seed_schema)
            .with_row_count_upper_bound(seed_row_bound),
    ));
    let params = GraphBetweennessCentralityParams {
        seeds: Some(GraphBetweennessSeeds::Input {
            input: GraphPredicateInput::new(seeds, "vertex"),
            row_count_upper_bound: seed_row_bound,
        }),
        ..Default::default()
    };
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::BetweennessCentrality(params),
    );
    let graph = builder.add_graph_algorithm(edge, spec);
    builder.finish_gpu_resident(graph)
}

fn bounded_join_plan(
    left_rows: Option<u64>,
    right_rows: Option<u64>,
    join_type: JoinType,
    keyed: bool,
    residual: bool,
) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let mut left_source = GpuResidentSource::new("bounded-join-left", Arc::clone(&schema));
    if let Some(rows) = left_rows {
        left_source = left_source.with_row_count_upper_bound(rows);
    }
    let mut right_source = GpuResidentSource::new("bounded-join-right", Arc::clone(&schema));
    if let Some(rows) = right_rows {
        right_source = right_source.with_row_count_upper_bound(rows);
    }
    let left = builder.add_source(SourceNode::GpuResident(left_source));
    let right = builder.add_source(SourceNode::GpuResident(right_source));
    let keys = keyed
        .then(|| JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key")))
        .into_iter()
        .collect();
    let mut spec = JoinSpec::new(join_type, keys);
    if residual {
        spec = spec.with_residual_predicate(Expr::literal(LiteralValue::Boolean(true)));
    }
    let join = builder.add_join(left, right, spec, schema);
    builder.finish_gpu_resident(join)
}

fn bounded_lookup_mapping_builder(edge_rows: u64, at_most_one_match: bool) -> (QueryPlanBuilder, NodeId, NodeId) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let codebook_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let edge = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("logical-mapping-edges", Arc::clone(&edge_schema)).with_row_count_upper_bound(edge_rows),
    ));
    let codebook = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("logical-mapping-codebook", Arc::clone(&codebook_schema))
            .with_row_count_upper_bound(edge_rows.checked_mul(2).expect("test codebook bound fits")),
    ));
    let lookup_spec = |column: &str| {
        let mut spec = JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, column, column), ColumnRef::input(1, "vertex", "vertex"))],
        );
        if at_most_one_match {
            spec = spec.with_at_most_one_match_per_bounded_row(JoinSide::Left);
        }
        spec
    };
    let first = builder.add_join(edge, codebook, lookup_spec("src"), Arc::clone(&edge_schema));
    let second = builder.add_join(first, codebook, lookup_spec("dst"), Arc::clone(&edge_schema));
    (builder, first, second)
}

fn bounded_lookup_mapping_plan(edge_rows: u64, at_most_one_match: bool) -> (QueryPlan, NodeId, NodeId) {
    let (builder, first, second) = bounded_lookup_mapping_builder(edge_rows, at_most_one_match);
    (builder.finish_gpu_resident(second), first, second)
}

fn bounded_lookup_mapping_graph_plan(edge_rows: u64) -> QueryPlan {
    let (mut builder, _, encoded_edges) = bounded_lookup_mapping_builder(edge_rows, true);
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let graph = builder.add_graph_algorithm(encoded_edges, spec);
    builder.finish_gpu_resident(graph)
}

#[test]
fn join_hard_row_bounds_are_checked_and_fail_closed() {
    let bounded = bounded_join_plan(Some(3), Some(4), JoinType::Inner, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&bounded), Some(12));

    let empty = bounded_join_plan(Some(0), Some(4), JoinType::Inner, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&empty), Some(0));

    let unknown = bounded_join_plan(None, Some(4), JoinType::Inner, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&unknown), None);

    let overflow = bounded_join_plan(Some(u64::MAX), Some(2), JoinType::Inner, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&overflow), None);

    let residual = bounded_join_plan(Some(3), Some(4), JoinType::Inner, true, true);
    assert_eq!(query_plan_hard_row_upper_bound(&residual), Some(12));

    let keyless = bounded_join_plan(Some(3), Some(4), JoinType::Inner, false, false);
    assert_eq!(query_plan_hard_row_upper_bound(&keyless), Some(12));

    for (join_type, keyed, expected) in [
        (JoinType::Left, true, 12),
        (JoinType::Full, true, 19),
        (JoinType::LeftSemi, true, 3),
        (JoinType::LeftAnti, true, 3),
        (JoinType::Cross, false, 12),
        (JoinType::LeftMark, true, 3),
    ] {
        let plan = bounded_join_plan(Some(3), Some(4), join_type, keyed, false);
        assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(expected), "join type {join_type}");
    }

    let empty_right_left = bounded_join_plan(Some(3), Some(0), JoinType::Left, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&empty_right_left), Some(3));

    let empty_left_full = bounded_join_plan(Some(0), Some(4), JoinType::Full, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&empty_left_full), Some(4));

    let unknown_right_semi = bounded_join_plan(Some(3), None, JoinType::LeftSemi, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&unknown_right_semi), Some(3));

    let full_overflow = bounded_join_plan(Some(u64::MAX), Some(1), JoinType::Full, true, false);
    assert_eq!(query_plan_hard_row_upper_bound(&full_overflow), None);
}

#[test]
fn at_most_one_match_bounds_both_logical_endpoint_joins() {
    let edge_rows = 48_322_510;
    let (plan, first, second) = bounded_lookup_mapping_plan(edge_rows, true);

    assert_eq!(query_plan_node_hard_row_upper_bound(&plan, first), Some(edge_rows));
    assert_eq!(query_plan_node_hard_row_upper_bound(&plan, second), Some(edge_rows));
    assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(edge_rows));

    let estimates = graph_memory_estimates_for_plan(
        &bounded_lookup_mapping_graph_plan(edge_rows),
        GraphMemoryCalibrationOverride::none(),
        None,
    )
    .expect("large logical mapping graph plan must remain priceable");
    assert_eq!(estimates.len(), 1);
    assert_eq!(estimates[0].estimate.input_edges, edge_rows);

    let evidence = graph_memory_bounded_admission_evidence_for_plan(
        &bounded_lookup_mapping_graph_plan(edge_rows),
        GraphMemoryCalibrationOverride::none(),
        None,
    )
    .expect("large logical mapping graph plan must produce one admission token");
    assert_eq!(evidence.owner_term_facts().iter().filter(|fact| fact.row_count_upper_bound().is_some()).count(), 1);
}

#[test]
fn lookup_contract_preserves_bounds_past_generic_checked_multiplication_boundary() {
    let last_safe_rows = 1_664_510;
    let first_overflow_rows = last_safe_rows + 1;

    let (safe_generic, _, safe_root) = bounded_lookup_mapping_plan(last_safe_rows, false);
    assert!(query_plan_node_hard_row_upper_bound(&safe_generic, safe_root).is_some());

    let (overflow_generic, _, overflow_root) = bounded_lookup_mapping_plan(first_overflow_rows, false);
    assert_eq!(query_plan_node_hard_row_upper_bound(&overflow_generic, overflow_root), None);

    let (overflow_lookup, _, lookup_root) = bounded_lookup_mapping_plan(first_overflow_rows, true);
    assert_eq!(query_plan_node_hard_row_upper_bound(&overflow_lookup, lookup_root), Some(first_overflow_rows));
}

#[test]
fn grouped_aggregate_hard_row_bound_cannot_exceed_its_input() {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("bounded-grouped-aggregate", Arc::clone(&schema)).with_row_count_upper_bound(3),
    ));
    let aggregate = builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column("key")], Vec::new()), schema);
    let plan = builder.finish_gpu_resident(aggregate);

    assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(3));
}

#[test]
fn graph_edge_output_hard_bound_can_feed_a_later_graph() {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("bounded-k-core-edges", edge_schema()).with_row_count_upper_bound(4),
    ));
    let k_core = builder.add_graph_algorithm(
        source,
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::KCore,
            GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::KCore(crate::plan::GraphKCoreParams::default()),
        ),
    );
    let plan = builder.finish_gpu_resident(k_core);

    assert_eq!(query_plan_hard_row_upper_bound(&plan), Some(4));
}

#[test]
fn bounded_graph_evidence_exposes_exclusive_owner_terms() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("graph-owner-input", edge_schema()).with_row_count_upper_bound(20),
    ));
    let graph = builder.add_graph_algorithm(source, spec);
    let plan = builder.finish_gpu_resident(graph);
    let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("bounded graph estimate")
        .pop()
        .expect("one graph estimate");
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded graph evidence");

    let facts = evidence.owner_term_facts();
    let expected_owner_term_count = 4
        + usize::from(evidence.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
        + usize::from(cfg!(feature = "cugraph"));
    assert_eq!(facts.len(), expected_owner_term_count);
    assert_eq!(facts.iter().filter(|fact| fact.row_count_upper_bound().is_some()).count(), 1);
    assert_eq!(
        facts
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphPersistent)
            .expect("persistent owner")
            .bytes(),
        estimate.estimate.owner_components.graph_persistent_bytes
    );
    assert_eq!(
        facts
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::ConstructionTransient)
            .expect("construction transient owner")
            .bytes(),
        estimate.estimate.owner_components.construction_transient_bytes
    );
    assert_eq!(
        facts
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::ExecutionWorkspace)
            .expect("execution workspace owner")
            .bytes(),
        estimate.estimate.owner_components.execution_workspace_bytes
    );
    assert_eq!(
        facts
            .iter()
            .find(|fact| fact.role() == GraphMemoryOwnerTermRole::RetainedOutput)
            .expect("retained output owner")
            .bytes(),
        estimate.estimate.retained_output_bytes.bytes()
    );
    let floor_top_up = facts
        .iter()
        .find(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
        .map(GraphMemoryOwnerTermFact::bytes);
    assert_eq!(
        floor_top_up,
        (evidence.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES)
            .then(|| GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - evidence.cold_model_requested_bytes())
    );
    if cfg!(feature = "cugraph") {
        assert!(
            facts
                .iter()
                .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
                .is_some_and(|fact| fact.bytes() > 0)
        );
    }
    assert_eq!(
        estimate
            .estimate
            .owner_components
            .execution_workspace_bytes
            .checked_add(estimate.estimate.retained_output_bytes.bytes()),
        Some(estimate.estimate.execution_bytes)
    );
    assert!(facts.iter().all(|fact| fact.role() != GraphMemoryOwnerTermRole::GraphPersistent
        || fact.bytes() != estimate.estimate.construction_bytes));
}

#[cfg(feature = "cugraph")]
#[test]
fn sssp_unit_weights_are_an_exact_active_phase_owner_term() {
    let spec = |weight: Option<&str>| {
        GraphAlgorithmSpec::with_stable_schema(
            GraphAlgorithm::Sssp,
            GraphEdgeColumnBindings::new("src", "dst", weight),
            GraphConstructionOptions::default(),
            GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: GraphSsspSource::I64(0), cutoff: None }),
        )
    };
    let unweighted = bounded_evidence_for_spec(&spec(None), 3);
    let weighted = bounded_evidence_for_spec(&spec(Some("weight")), 3);
    let unit_weights = unweighted
        .owner_term_facts()
        .iter()
        .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SsspUnitWeights)
        .expect("unweighted SSSP must price the native unit-weight property");
    assert_eq!(unit_weights.bytes(), 12);
    assert!(
        weighted.owner_term_facts().iter().all(|fact| fact.role() != GraphMemoryOwnerTermRole::SsspUnitWeights),
        "weighted SSSP borrows its graph-owned weight property"
    );
}

#[test]
fn bounded_graph_live_floor_is_a_checked_owner_top_up_over_raw_model() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let tiny = bounded_evidence_for_spec(&spec, 1);
    assert!(tiny.cold_model_requested_bytes() < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    assert_eq!(
        tiny.bounded_live_graph_peak_bytes().expect("tiny graph floor fits"),
        GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES
    );
    assert_eq!(
        tiny.owner_term_facts()
            .iter()
            .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
            .map(GraphMemoryOwnerTermFact::bytes)
            .collect::<Vec<_>>(),
        vec![GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - tiny.cold_model_requested_bytes()]
    );

    let large = bounded_evidence_for_spec(&spec, 100_000);
    assert!(large.cold_model_requested_bytes() >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);
    assert_eq!(
        large.bounded_live_graph_peak_bytes().expect("large graph model fits"),
        large.cold_model_requested_bytes(),
        "the canonical floor must not alter sufficiently large graph models"
    );
    assert!(
        large
            .owner_term_facts()
            .iter()
            .all(|fact| fact.role() != GraphMemoryOwnerTermRole::GraphLiveReservationFloorTopUp)
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn bfs_side_input_owner_uses_the_owner_peak_not_component_sum() {
    let side_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let plan = bfs_side_input_plan(side_schema, Some(3), |side| {
        let mut params = GraphBfsParams::new(0, None);
        params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
        params
    });
    let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("bounded BFS estimate")
        .pop()
        .expect("one BFS estimate");
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded BFS side-input evidence");
    let owner_peak = evidence
        .owner_term_facts()
        .iter()
        .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
        .expect("side-input owner term")
        .bytes();
    let preflight = cugraph::cudf_interop::bfs_side_input_workspace_preflight(
        &cugraph::cudf_interop::BfsSideInputPreflightInput::new(
            cugraph::types::DataType::Int64,
            cugraph::types::DataType::Int64,
            usize::try_from(estimate.estimate.vertex_estimate).expect("vertex estimate fits usize"),
            usize::try_from(estimate.estimate.input_edges).expect("edge estimate fits usize"),
            1,
            Some(3),
            None,
            None,
            None,
        )
        .expect("BFS side-input preflight input"),
    )
    .expect("BFS side-input preflight");
    assert_eq!(owner_peak, preflight.peak_bytes());
    assert_ne!(
        owner_peak,
        preflight.source_copy_bytes() + preflight.predicate_bitmap_bytes() + preflight.edge_mask_bytes(),
        "the owner term must preserve the co-resident peak rather than sum sequential components"
    );
}

#[cfg(feature = "cugraph")]
#[test]
fn bfs_empty_source_side_input_owner_is_zero() {
    let side_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
    let plan = bfs_side_input_plan(side_schema, Some(0), |side| {
        let mut params = GraphBfsParams::new(0, None);
        params.source_vertices = Some(GraphPredicateInput::new(side, "vertex"));
        params.endpoint_policy = crate::plan::GraphBfsEndpointPolicy::EmptyOnMissingPathEndpoint;
        params
    });
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("zero-row BFS source must retain finite graph-memory evidence");
    let side_input_terms = evidence
        .owner_term_facts()
        .iter()
        .filter(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
        .collect::<Vec<_>>();
    assert_eq!(side_input_terms.len(), 1);
    assert_eq!(side_input_terms[0].bytes(), 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn personalized_pagerank_side_input_owner_requires_a_hard_row_bound() {
    let plan = personalized_pagerank_side_input_plan(Some(3));
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded PPR side-input evidence");
    let owner_peak = evidence
        .owner_term_facts()
        .iter()
        .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
        .expect("PPR side-input owner term")
        .bytes();
    assert!(owner_peak > 0);

    let missing_rows = personalized_pagerank_side_input_plan(None);
    let error =
        graph_memory_bounded_admission_evidence_for_plan(&missing_rows, GraphMemoryCalibrationOverride::none(), None)
            .expect_err("PPR side input without a hard row bound must fail closed");
    assert_eq!(error.fact_value("reason"), Some("side_input_workspace_preflight_unavailable"));
    assert_eq!(error.fact_value("side_input_reason"), Some("missing_row_upper_bound"));
    assert_eq!(error.fact_value("side_input_role"), Some("personalization"));
}

#[cfg(feature = "cugraph")]
#[test]
fn relation_backed_betweenness_seeds_use_the_proven_row_bound_for_admission() {
    let estimate = graph_memory_estimates_for_plan(
        &betweenness_seed_side_input_plan(3),
        GraphMemoryCalibrationOverride::none(),
        None,
    )
    .expect("bounded betweenness seed relation must remain priceable")
    .pop()
    .expect("one graph estimate");
    assert_eq!(estimate.estimate.seed_count, Some(3));
    assert_eq!(estimate.estimate.seed_mode, Some(GraphSeedMode::ExplicitSeedList));

    let evidence = graph_memory_bounded_admission_evidence_for_plan(
        &betweenness_seed_side_input_plan(3),
        GraphMemoryCalibrationOverride::none(),
        None,
    )
    .expect("bounded betweenness seed relation must produce admission evidence");
    assert!(evidence.cold_model_requested_bytes() > 0);
}

#[cfg(feature = "cugraph")]
#[test]
fn bfs_side_input_preflight_rejects_unsupported_dtype_and_aliases_and_prices_include_edges() {
    let unsupported = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("vertex", DataType::UInt64, false)])),
        Some(3),
        |side| {
            let mut params = GraphBfsParams::new(0, None);
            params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
            params
        },
    );
    let error =
        graph_memory_bounded_admission_evidence_for_plan(&unsupported, GraphMemoryCalibrationOverride::none(), None)
            .expect_err("unsupported BFS side-input dtype must fail closed");
    assert_eq!(error.kind(), crate::error::QueryErrorKind::Unsupported);
    assert_eq!(error.fact_value("side_input_reason"), Some("side_input_vertex_dtype_unsupported"));

    let include_edges = bfs_side_input_plan(edge_schema(), Some(3), |side| {
        let mut params = GraphBfsParams::new(0, None);
        params.predicates.include_edges = Some(GraphEdgePredicateInput::new(side, "src", "dst", None::<&str>));
        params
    });
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&include_edges, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded include_edges must use the filtered-join owner contract");
    let owner_peak = evidence
        .owner_term_facts()
        .iter()
        .find(|fact| fact.role() == GraphMemoryOwnerTermRole::SideInputWorkspace)
        .expect("include_edges side-input owner term")
        .bytes();
    assert!(owner_peak > 0);

    let duplicate = bfs_side_input_plan(
        Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)])),
        Some(3),
        |side| {
            let mut params = GraphBfsParams::new(0, None);
            params.predicates.include_vertices = Some(GraphPredicateInput::new(side, "vertex"));
            params.predicates.exclude_vertices = Some(GraphPredicateInput::new(side, "vertex"));
            params
        },
    );
    let error =
        graph_memory_bounded_admission_evidence_for_plan(&duplicate, GraphMemoryCalibrationOverride::none(), None)
            .expect_err("reused BFS side-input nodes must fail closed");
    assert_eq!(error.fact_value("side_input_reason"), Some("side_input_node_reused"));
}

#[test]
fn malformed_graph_owner_evidence_rejects_saturation_and_missing_terms() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("graph-owner-input", edge_schema()).with_row_count_upper_bound(20),
    ));
    let graph = builder.add_graph_algorithm(source, spec);
    let plan = builder.finish_gpu_resident(graph);
    let evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("bounded graph evidence");
    let mut saturated = evidence.clone();
    saturated.owner_terms[0].bytes = u64::MAX;
    let proof = crate::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
    let error = proof.with_graph_memory_admission_evidence(&saturated).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_saturated"));

    let mut incomplete = evidence;
    incomplete.owner_terms.remove(0);
    let proof = crate::admission::QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan).unwrap();
    let error = proof.with_graph_memory_admission_evidence(&incomplete).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_owner_term_role_missing"));
}

fn triangle_plan(source: LocalParquetSource) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::TriangleCountAll,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::TriangleCountAll(GraphTriangleCountAllParams),
    );
    let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
    builder.finish_with_host_sink(graph)
}

fn raw_bfs_plan(source: LocalParquetSource) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let graph = builder.try_add_graph_algorithm(source, spec).unwrap();
    builder.finish_with_host_sink(graph)
}

#[cfg(feature = "cugraph")]
#[test]
fn graph_memory_estimates_for_plan_raw_bfs_consumes_owner_construction_workspace_bounds() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![100])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string()],
    );
    let plan = raw_bfs_plan(source);

    let planned = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("the owner preflight supplies raw BFS bounded-admission evidence");
    let graph = plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::GraphAlgorithm(graph) => Some(graph),
            _ => None,
        })
        .expect("raw BFS plan contains a graph node");
    let input = graph_memory_model_input_for_plan_node(&plan, graph, GraphMemoryCalibrationOverride::none())
        .unwrap()
        .expect("literal row count is hard graph-input evidence");
    let workspace_bounds = graph_construction_workspace_bounds_from_owner(&graph.spec, input)
        .expect("same typed owner query must produce the construction bounds");
    let expected = GraphMemoryModel
        .estimate_for_bounded(&graph.spec, input.with_construction_workspace_bounds(workspace_bounds))
        .unwrap();

    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].estimate, expected);
}

#[cfg(feature = "cugraph")]
#[test]
fn raw_bfs_owner_preflight_query_failure_leaves_the_bound_unavailable() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None);

    assert_eq!(
        graph_construction_workspace_bounds_from_owner_query(&spec, input, |_preflight| None),
        None,
        "an owner query error must remain an absent bound"
    );
    let error = GraphMemoryModel
        .estimate_for_bounded(&spec, input)
        .expect_err("an absent owner bound must retain the raw BFS rejection");
    assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
}

#[cfg(feature = "cugraph")]
#[test]
fn raw_bfs_owner_preflight_passes_renumber_and_refined_vertex_bound() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None)
        .with_vertex_estimate(GraphVertexEstimate::runtime_exact_distinct(37));
    let bounds = graph_construction_workspace_bounds_from_owner_query(&spec, input, |preflight| {
        assert!(preflight.renumber());
        assert_eq!(preflight.num_edges(), 100);
        assert_eq!(preflight.num_vertices(), 37);
        Some(GraphConstructionWorkspaceBounds::new(11, 17))
    })
    .expect("callback returns owner bounds");
    assert_eq!(bounds.renumber_workspace_bytes(), 11);
    assert_eq!(bounds.compression_workspace_bytes(), 17);
}

#[cfg(feature = "cugraph")]
#[test]
fn raw_bfs_owner_preflight_falls_back_to_endpoint_vertex_bound() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let input = graph_memory_model_input(100, 8, &spec.edge_columns, false, None);
    let bounds = graph_construction_workspace_bounds_from_owner_query(&spec, input, |preflight| {
        assert!(preflight.renumber());
        assert_eq!(preflight.num_edges(), 100);
        assert_eq!(preflight.num_vertices(), 200);
        Some(GraphConstructionWorkspaceBounds::new(13, 19))
    })
    .expect("callback returns owner bounds");
    assert_eq!(bounds, GraphConstructionWorkspaceBounds::new(13, 19));
}

#[cfg(not(feature = "cugraph"))]
#[test]
fn graph_memory_estimates_for_plan_raw_bfs_refuses_without_cugraph_preflight() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![100])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string()],
    );

    let error = graph_memory_estimates_for_plan(&raw_bfs_plan(source), GraphMemoryCalibrationOverride::none(), None)
        .expect_err("feature-off planning has no owner preflight contract");

    assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
}

#[test]
fn graph_memory_estimates_for_plan_use_row_group_rows_and_edge_normalize_upper_bound() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![10, 5])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let plan = triangle_plan(source);

    let estimates =
        graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none().with_raw_input_edges(100), None)
            .unwrap();

    assert_eq!(estimates.len(), 1);
    let estimate = &estimates[0].estimate;
    assert_eq!(estimate.raw_input_edges, Some(15));
    assert_eq!(estimate.input_edges, 30);
    assert_eq!(estimate.vertex_width_bytes, 8);
    assert_eq!(estimate.weight_width_bytes, None);
    assert!(estimate.edge_normalize_inserted);
    assert_eq!(estimate.calibration_source, GraphCalibrationSource::RawInputEdges);
    assert_eq!(estimate.requested_bytes, estimate.construction_bytes.saturating_add(estimate.execution_bytes));
    assert!(estimate.construction_bytes > 0);
    assert!(estimate.execution_bytes < estimate.requested_bytes);
}

#[test]
fn graph_memory_evidence_keeps_typed_retained_output_distinct_from_execution_workspace() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::ForceAtlas2,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
    );
    let evidence = bounded_evidence_for_spec(&spec, 10);

    assert!(evidence.retained_output_bytes() > 0);
    assert!(evidence.retained_output_rows_upper_bound() > 0);
    assert!(evidence.warm_model_requested_bytes() > evidence.retained_output_bytes());
    assert_ne!(
        evidence.retained_output_bytes(),
        evidence.warm_model_requested_bytes(),
        "retained output must not be recovered from the rounded execution peak"
    );
}

#[test]
fn graph_memory_source_lifetime_composes_from_owner_evidence() {
    let base_spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let source_bytes = 2_048;
    let shared = bounded_evidence_for_spec(&base_spec, 20);
    let normalized = bounded_evidence_for_spec(&base_spec.clone().with_edge_normalize_inserted(true), 20);

    assert_eq!(
        shared.cold_model_with_source_stage_bytes(source_bytes).unwrap(),
        shared.cold_model_requested_bytes().max(source_bytes),
        "a prepared edge frame is already owned by the shared graph peak"
    );
    assert_eq!(
        normalized.cold_model_with_source_stage_bytes(source_bytes).unwrap(),
        source_bytes.checked_add(normalized.cold_model_requested_bytes()).unwrap(),
        "explicit normalization retains source and normalized graph input together"
    );
}

#[test]
fn bounded_model_rejects_exact_u64_max_projection() {
    let error = ModelBytes(u128::from(u64::MAX)).bounded_u64("bounded_test_term").unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
    assert_eq!(error.fact_value("term"), Some("bounded_test_term"));
    assert_eq!(error.fact_value("operation"), Some("saturated"));
}

#[test]
fn shared_graph_source_composition_rejects_exact_u64_max() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let evidence = bounded_evidence_for_spec(&spec, 1);
    let error = evidence.cold_model_with_source_stage_bytes(u64::MAX).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
    assert_eq!(error.fact_value("term"), Some("source_stage_graph_peak"));
    assert_eq!(error.fact_value("operation"), Some("saturated"));
}

#[test]
fn bounded_graph_reservation_credits_only_its_live_normalized_input_owner() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![1_000_000])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let estimate =
        graph_memory_estimates_for_plan(&triangle_plan(source), GraphMemoryCalibrationOverride::none(), None)
            .unwrap()
            .into_iter()
            .next()
            .expect("graph plan yields one model estimate");
    let full_model_bytes = estimate.estimate.requested_bytes;
    let semantic_edge_row_width = semantic_edge_row_width_bytes(
        estimate.estimate.vertex_width_bytes,
        estimate.estimate.weight_width_bytes,
        estimate.estimate.has_edge_id,
    );
    let modeled_input_bytes = estimate.estimate.input_edges.saturating_mul(semantic_edge_row_width);
    let raw_source_provenance_bytes = estimate
        .estimate
        .raw_input_edges
        .expect("complete normalized source retains an exact raw row count")
        .saturating_mul(semantic_edge_row_width);
    let remaining_budget_bytes = full_model_bytes.saturating_sub(modeled_input_bytes);
    assert!(remaining_budget_bytes >= GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);

    let admitted = GraphReservationReport::from_estimate(
        estimate.node_id,
        estimate.estimate.clone(),
        Some(remaining_budget_bytes),
    )
    .credit_live_graph_input(modeled_input_bytes, None, 0);
    assert_eq!(admitted.model_requested_bytes, full_model_bytes);
    assert_eq!(admitted.requested_bytes, remaining_budget_bytes);
    assert_eq!(admitted.outcome, GraphReservationOutcome::Admitted);

    let capped = GraphReservationReport::from_estimate(
        estimate.node_id,
        estimate.estimate.clone(),
        Some(remaining_budget_bytes),
    )
    .credit_live_graph_input(u64::MAX, None, 0);
    assert_eq!(capped.model_requested_bytes, full_model_bytes);
    assert_eq!(capped.requested_bytes, remaining_budget_bytes);
    assert_eq!(capped.outcome, GraphReservationOutcome::Admitted);

    let provenance_only = GraphReservationReport::from_estimate(
        estimate.node_id,
        estimate.estimate.clone(),
        Some(remaining_budget_bytes),
    )
    .credit_live_graph_input(modeled_input_bytes.saturating_add(raw_source_provenance_bytes), None, 0);
    assert_eq!(
        provenance_only.requested_bytes, remaining_budget_bytes,
        "raw source rows are provenance only; they cannot increase graph-input credit"
    );

    let denied = GraphReservationReport::from_estimate(
        estimate.node_id,
        estimate.estimate.clone(),
        Some(remaining_budget_bytes),
    )
    .credit_live_graph_input(modeled_input_bytes.saturating_sub(1), None, 0);
    assert_eq!(denied.model_requested_bytes, full_model_bytes);
    assert_eq!(denied.requested_bytes, remaining_budget_bytes.saturating_add(1));
    assert_eq!(denied.outcome, GraphReservationOutcome::Denied);

    // EdgeNormalize is the exact owner of the logical graph edge frame.
    // Its native output can retain aliases beyond the semantic column
    // widths, so a caller that proves this ownership may credit the
    // tracker charge rather than pretending those bytes are a second raw
    // source frame.
    let exact_normalized_frame_charge_bytes = modeled_input_bytes.saturating_add(4_096);
    let exact_owner_budget_bytes = full_model_bytes.saturating_sub(exact_normalized_frame_charge_bytes);
    let normalized_and_source_budget_bytes = full_model_bytes
        .checked_sub(modeled_input_bytes.checked_mul(2).expect("test input bytes fit u64"))
        .expect("graph model covers normalized and retained source frames");
    let normalized_and_source = GraphReservationReport::from_estimate(
        estimate.node_id,
        estimate.estimate.clone(),
        Some(normalized_and_source_budget_bytes),
    )
    .credit_live_graph_input(modeled_input_bytes, Some(modeled_input_bytes), modeled_input_bytes);
    assert_eq!(normalized_and_source.requested_bytes, normalized_and_source_budget_bytes);
    assert_eq!(normalized_and_source.outcome, GraphReservationOutcome::Admitted);

    let exact_owner =
        GraphReservationReport::from_estimate(estimate.node_id, estimate.estimate, Some(exact_owner_budget_bytes))
            .credit_live_graph_input(u64::MAX, Some(exact_normalized_frame_charge_bytes), 0);
    assert_eq!(exact_owner.requested_bytes, exact_owner_budget_bytes);
    assert_eq!(exact_owner.outcome, GraphReservationOutcome::Admitted);
}

#[test]
fn normalized_graph_raw_source_count_is_provenance_not_physical_residency() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::ForceAtlas2,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
    )
    .with_edge_normalize_inserted(true);
    let normalized_input = GraphMemoryModelInput {
        raw_input_edges: Some(200),
        input_edges: 400,
        vertex_width_bytes: 8,
        weight_width_bytes: Some(8),
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::graph_metadata(50)),
        edge_normalize_inserted: true,
        construction_workspace_bounds: None,
        statistics: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };
    let without_raw = GraphMemoryModel.estimate_for_bounded(&spec, normalized_input).unwrap();
    let with_raw = GraphMemoryModel.estimate_for_bounded(&spec, normalized_input.with_raw_input_edges(100)).unwrap();
    assert_eq!(with_raw.raw_input_edges, Some(100));
    assert_eq!(with_raw.construction_bytes, without_raw.construction_bytes);
    assert_eq!(with_raw.execution_bytes, without_raw.execution_bytes);
    assert_eq!(with_raw.requested_bytes, without_raw.requested_bytes);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        crate::source::GpuResidentSource::new("normalized-graph-input", edge_schema())
            .with_row_count_upper_bound(200)
            .with_graph_vertex_upper_bound(50),
    ));
    let graph = builder.add_graph_algorithm(source, spec.clone());
    let plan = builder.finish_gpu_resident(graph);
    let canonical_estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("canonical lowered graph estimate")
        .pop()
        .expect("one canonical graph estimate");
    assert_eq!(canonical_estimate.estimate.raw_input_edges, Some(200));
    assert_eq!(canonical_estimate.estimate.input_edges, 400);
    assert_eq!(canonical_estimate.estimate.vertex_estimate, 50);
    let cold_evidence =
        graph_memory_bounded_admission_evidence_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
            .expect("canonical bounded graph evidence");
    assert_eq!(cold_evidence.cold_model_requested_bytes(), canonical_estimate.estimate.requested_bytes);
}

#[test]
fn graph_memory_estimates_for_plan_can_use_configured_raw_edges_when_plan_rows_are_unknown() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet")],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let plan = triangle_plan(source);
    let calibration = GraphMemoryCalibrationOverride::none().with_raw_input_edges(100);

    let estimates = graph_memory_estimates_for_plan(&plan, calibration, None).unwrap();

    assert_eq!(estimates.len(), 1);
    let estimate = &estimates[0].estimate;
    assert_eq!(estimate.raw_input_edges, Some(100));
    assert_eq!(estimate.input_edges, 200);
    assert_eq!(estimate.calibration_source, GraphCalibrationSource::RawInputEdges);
}

#[test]
fn graph_memory_estimates_for_plan_skip_graph_nodes_without_edge_count_evidence() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet")],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let plan = triangle_plan(source);

    let estimates = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None).unwrap();

    assert!(estimates.is_empty());
}

#[test]
fn graph_plan_memory_uses_footer_vertex_bound_only_for_complete_rowsets() {
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("first.parquet").with_row_group_rows(vec![10]).with_graph_vertex_upper_bound(30),
            LocalParquetFile::new("second.parquet").with_row_group_rows(vec![20]).with_graph_vertex_upper_bound(50),
        ],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let plan = triangle_plan(source);

    let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .unwrap()
        .into_iter()
        .next()
        .expect("footer-backed graph plan has a model estimate")
        .estimate;
    assert_eq!(estimate.vertex_estimate, 80);
    assert_eq!(estimate.vertex_estimate_source, GraphVertexEstimateSource::GraphMetadata);

    let partial_source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("first.parquet")
                .with_row_group_rows(vec![10, 20])
                .with_row_groups(vec![0])
                .with_graph_vertex_upper_bound(300),
        ],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let partial_estimate =
        graph_memory_estimates_for_plan(&triangle_plan(partial_source), GraphMemoryCalibrationOverride::none(), None)
            .unwrap()
            .into_iter()
            .next()
            .expect("selected row groups still have a hard row bound")
            .estimate;
    assert_eq!(partial_estimate.vertex_estimate_source, GraphVertexEstimateSource::EdgeEndpointUpperBound);
}

#[test]
fn force_atlas2_workspace_is_linear_and_uses_the_dispatch_path() {
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 0,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        construction_workspace_bounds: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };
    let output_sized = 4_096;
    let exact_params = GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default());
    let barnes_hut = GraphForceAtlas2Params { barnes_hut_optimize: true, ..Default::default() };
    let barnes_hut_params = GraphAlgorithmParams::ForceAtlas2(barnes_hut);

    assert_eq!(force_atlas2_workspace_bytes(input, 10, &exact_params, output_sized), output_sized + 10 * 44);
    assert_eq!(
        force_atlas2_workspace_bytes(input, 10, &barnes_hut_params, output_sized),
        output_sized + 10 * (120 + 16),
        "Barnes-Hut includes its Int64 degree vector"
    );
    assert_eq!(
        force_atlas2_workspace_bytes(input, 20, &exact_params, output_sized)
            - force_atlas2_workspace_bytes(input, 10, &exact_params, output_sized),
        10 * 44,
        "the exact path must stay linear in vertices"
    );

    let weighted_edges = GraphMemoryModelInput { input_edges: 10, weight_width_bytes: Some(8), ..input };
    assert_eq!(
        force_atlas2_workspace_bytes(weighted_edges, 0, &exact_params, 0),
        10 * 25,
        "the legacy COO stable-sort workspace is one weighted record plus its CUB spine"
    );

    let unweighted_edges = GraphMemoryModelInput { input_edges: 10, weight_width_bytes: None, ..input };
    assert_eq!(
        force_atlas2_workspace_bytes(unweighted_edges, 0, &exact_params, 0),
        10 * 21,
        "the C-API materializes Float32 unit weights for unweighted ForceAtlas2"
    );
}

#[test]
fn force_atlas2_phase_peaks_keep_sequential_allocations_separate() {
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(50),
        input_edges: 100,
        vertex_width_bytes: 8,
        weight_width_bytes: Some(8),
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::graph_metadata(10)),
        statistics: None,
        edge_normalize_inserted: true,
        construction_workspace_bounds: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };
    let barnes_hut = GraphForceAtlas2Params { barnes_hut_optimize: true, ..Default::default() };
    let params = GraphAlgorithmParams::ForceAtlas2(barnes_hut);
    let construction_bytes = 4_000;
    let result_conversion_bytes = 160;
    let phases = force_atlas2_phase_peaks(input, 10, &params, construction_bytes, result_conversion_bytes);
    let workspace = force_atlas2_workspace_phase_bytes(input, 10, &params, 2_640);

    assert_eq!(
        phases.legacy_coo_sort_before_degree_bytes,
        phases.persistent_base_bytes.saturating_add(workspace.legacy_coo_sort_workspace_bytes),
        "COO sorting completes before the Barnes-Hut degree vector is allocated"
    );
    assert_eq!(
        phases.barnes_hut_with_degree_bytes,
        phases.persistent_base_bytes.saturating_add(workspace.barnes_hut_workspace_bytes),
        "the degree vector belongs only to the Barnes-Hut phase"
    );
    assert_eq!(
        phases.post_barnes_result_conversion_bytes,
        phases.persistent_base_bytes.saturating_add(result_conversion_bytes),
        "result vectors are created only after Barnes-Hut returns"
    );
    assert_eq!(
        phases.peak_bytes(),
        phases
            .legacy_coo_sort_before_degree_bytes
            .max(phases.barnes_hut_with_degree_bytes)
            .max(phases.post_barnes_result_conversion_bytes)
    );
}

#[test]
fn graph_construction_prices_sequential_renumber_and_property_compression_phases() {
    let mut input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 10,
        vertex_width_bytes: 8,
        weight_width_bytes: Some(8),
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        construction_workspace_bounds: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 10 * 8);

    input.weight_width_bytes = None;
    assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 0);

    input.has_edge_id = true;
    assert_eq!(raw_libcugraph_property_output_staging(input).unwrap().diagnostic_u64(), 10 * 8);

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::ForceAtlas2,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: false, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::ForceAtlas2(GraphForceAtlas2Params::default()),
    );
    input.has_edge_id = false;
    input.weight_width_bytes = Some(8);
    let construction_with_edges = GraphMemoryModel.estimate(&spec, input).unwrap().construction_bytes;
    input.input_edges = 0;
    let construction_without_edges = GraphMemoryModel.estimate(&spec, input).unwrap().construction_bytes;
    assert_eq!(construction_with_edges - construction_without_edges, 10 * (24 + 24 + 8));

    let weighted_pagerank = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", Some("weight")),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    input.input_edges = 10;
    input.vertex_estimate = Some(GraphVertexEstimate::graph_metadata(5));
    let error = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
    assert_eq!(error.fact_value("algorithm"), Some("pagerank"));

    input.construction_workspace_bounds = Some(GraphConstructionWorkspaceBounds::new(200, 0));
    let renumber_dominant = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap();
    assert_eq!(renumber_dominant.construction_bytes, 240 + 240 + 200);

    input.construction_workspace_bounds = Some(GraphConstructionWorkspaceBounds::new(0, 200));
    let compression_dominant = GraphMemoryModel.estimate_for_bounded(&weighted_pagerank, input).unwrap();
    assert_eq!(compression_dominant.construction_bytes, 240 + 240 + 40 + 80 + 200);
}

#[test]
fn raw_bfs_phase_price_retains_normalization_source_and_prepared_input() {
    const EDGES: u64 = 45_564_149;
    const VERTICES: u64 = 4_894_081;
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(1_091, None))),
    )
    .with_edge_normalize_inserted(true);
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(EDGES),
        input_edges: EDGES,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::graph_metadata(VERTICES)),
        statistics: None,
        edge_normalize_inserted: true,
        construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    let estimate = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap();
    let normalized_input_bytes = EDGES * 16;

    assert_eq!(estimate.construction_bytes, 2_226_231_800);
    assert_eq!(estimate.execution_bytes, 1_328_455_464);
    assert_eq!(estimate.requested_bytes, 3_515_534_616);
    assert_eq!(
        estimate.requested_bytes,
        2 * normalized_input_bytes + normalized_input_bytes + estimate.execution_bytes,
        "the source and normalized edge owners remain live for BFS, while construction-only work does not"
    );
}

#[test]
fn bounded_graph_arithmetic_rejects_early_cardinality_overflow() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: false, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let input = graph_memory_model_input(u64::MAX, u64::MAX, &spec.edge_columns, false, None)
        .with_construction_workspace_bounds(GraphConstructionWorkspaceBounds::new(0, 0));
    let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
    assert_eq!(error.fact_value("term"), Some("vertex_upper_bound"));
}

#[test]
fn bounded_graph_arithmetic_rejects_late_phase_overflow() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: false, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let input = graph_memory_model_input(u64::MAX / 24, 1, &spec.edge_columns, false, None)
        .with_construction_workspace_bounds(GraphConstructionWorkspaceBounds::new(0, 0));
    let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
    assert_eq!(error.fact_value("term"), Some("execution_workspace_bytes"));
}

#[test]
fn bounded_normalized_source_and_graph_peak_reject_addition_overflow() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: false, policy: GraphConstructionPolicy::PythonCugraph },
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default()),
    );
    let evidence = bounded_evidence_for_spec(&spec, 1);
    let error = evidence.cold_model_with_source_stage_bytes(u64::MAX).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_memory_arithmetic_overflow"));
    assert_eq!(error.fact_value("term"), Some("source_stage_graph_peak"));
}

#[test]
fn bounded_raw_bfs_requires_a_preflight_workspace_contract() {
    let mut spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: Some(100),
        input_edges: 100,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: Some(GraphVertexEstimate::graph_metadata(50)),
        edge_normalize_inserted: false,
        construction_workspace_bounds: None,
        statistics: None,
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    let error = GraphMemoryModel.estimate_for_bounded(&spec, input).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("graph_construction_workspace_preflight_unavailable"));
    assert_eq!(error.fact_value("algorithm"), Some("bfs"));

    let renumber_workspace_bytes = 10_000;
    let compression_workspace_bytes = 0;
    let workspace_bounds = GraphConstructionWorkspaceBounds::new(renumber_workspace_bytes, compression_workspace_bytes);
    let admitted = GraphMemoryModel
        .estimate_for_bounded(&spec, input.with_construction_workspace_bounds(workspace_bounds))
        .expect("the owner-provided bound completes raw BFS admission evidence");
    let persistent_graph_and_input_bytes = input.input_edges * input.vertex_width_bytes * 2;

    assert_eq!(admitted.construction_bytes, 13_200);
    assert_eq!(admitted.execution_bytes, 4_800);
    assert_eq!(
        admitted.requested_bytes,
        admitted.construction_bytes.max(persistent_graph_and_input_bytes.saturating_add(admitted.execution_bytes)),
        "construction-only workspace must not remain live during execution"
    );
    assert_eq!(admitted.requested_bytes, admitted.construction_bytes);

    spec.edge_normalize_inserted = true;
    let normalized = input.with_edge_normalize_inserted(true).with_construction_workspace_bounds(workspace_bounds);
    assert!(GraphMemoryModel.estimate_for_bounded(&spec, normalized).is_ok());
}

#[test]
fn raw_bfs_owner_bounds_retain_renumber_map_across_compression_and_execution() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: true, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    );
    let input = graph_memory_model_input(1_000, 8, &spec.edge_columns, false, None)
        .with_vertex_estimate(GraphVertexEstimate::graph_metadata(50));
    let bounds = GraphConstructionWorkspaceBounds::new(6_000, 10_000);
    let estimate = GraphMemoryModel
        .estimate_for_bounded(&spec, input.with_construction_workspace_bounds(bounds))
        .expect("owner bounds complete bounded raw BFS pricing");

    assert_eq!(estimate.owner_components.graph_persistent_bytes, 16_000);
    assert_eq!(estimate.owner_components.construction_transient_bytes, 26_400);
    assert_eq!(estimate.construction_bytes, 42_400);
    assert_eq!(estimate.execution_bytes, 26_400);
    assert_eq!(estimate.requested_bytes, 42_400);
}

#[test]
fn graph_claim_calibration_row_keeps_claim_and_peak_provenance_distinct() {
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new("edges.parquet").with_row_group_rows(vec![10, 5])],
        edge_schema(),
        vec!["src".to_string(), "dst".to_string(), "weight".to_string()],
    );
    let plan = triangle_plan(source);
    let estimate = graph_memory_estimates_for_plan(&plan, GraphMemoryCalibrationOverride::none(), None)
        .unwrap()
        .into_iter()
        .next()
        .expect("graph plan yields one model estimate");
    let report = GraphReservationReport::from_estimate(estimate.node_id, estimate.estimate, Some(64 * 1024 * 1024));
    let mut row = GraphClaimCalibrationRow::from_reservation_report(&report, estimate.node_id);

    assert_eq!(row.node_id, estimate.node_id.to_string());
    assert_eq!(row.node_index, estimate.node_id.index());
    assert_eq!(row.algorithm, GraphAlgorithm::TriangleCountAll);
    assert_eq!(row.model_requested_bytes, report.model_requested_bytes);
    assert_eq!(row.requested_bytes, report.requested_bytes);
    assert_eq!(row.budget_bytes, report.budget_bytes);
    assert_eq!(row.execution_state, GraphClaimExecutionState::Pending);
    assert_eq!(row.measured_peak_bytes, None);
    assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
    assert_eq!(row.graph_model_error_ratio, None);

    row.record_measured_peak(Some(4096), GraphMeasuredPeakProvenance::ReservationHighWater);
    assert_eq!(row.measured_peak_bytes, Some(4096));
    assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::ReservationHighWater);
    assert_eq!(
        row.graph_model_error_ratio,
        GraphModelErrorRatio::from_requested_and_measured(report.model_requested_bytes, 4096)
    );

    row.record_measured_peak(Some(8192), GraphMeasuredPeakProvenance::Unavailable);
    assert_eq!(row.measured_peak_bytes, None, "unavailable provenance cannot retain a guessed numeric peak");
    assert_eq!(row.graph_model_error_ratio, None, "unavailable provenance cannot retain a ratio");
    assert_eq!(row.measured_peak_provenance, GraphMeasuredPeakProvenance::Unavailable);
}

#[test]
fn bounded_similarity_model_accepts_only_pre_admission_pair_evidence() {
    assert!(GraphPairCardinalitySource::PlanExact.is_pre_admission_bound());
    assert!(GraphPairCardinalitySource::PlanHardUpperBound.is_pre_admission_bound());
    assert!(!GraphPairCardinalitySource::RuntimeMaterialized.is_pre_admission_bound());

    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 20,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: false,
        construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    };

    let runtime_only = GraphMemoryModel
        .estimate_for_bounded(
            &spec,
            input.with_similarity_pair_input(100, 8, GraphPairCardinalitySource::RuntimeMaterialized),
        )
        .unwrap_err();
    assert_eq!(runtime_only.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(runtime_only.fact_value("reason"), Some("bounded_pair_cardinality_evidence_not_plan_bound"));

    for source in [GraphPairCardinalitySource::PlanExact, GraphPairCardinalitySource::PlanHardUpperBound] {
        let estimate =
            GraphMemoryModel.estimate_for_bounded(&spec, input.with_similarity_pair_input(100, 8, source)).unwrap();
        assert_eq!(estimate.pair_cardinality_source, Some(source));
        assert_eq!(estimate.pair_cardinality_admission_mode, Some(GraphPairCardinalityAdmissionMode::Bounded));
    }
}

#[test]
fn bounded_graph_floor_is_applied_before_live_input_credits() {
    let spec = GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Jaccard,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed: false, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::default()),
    );
    let input = GraphMemoryModelInput {
        raw_input_edges: None,
        input_edges: 6,
        vertex_width_bytes: 8,
        weight_width_bytes: None,
        has_edge_id: false,
        device_total_memory_bytes: None,
        vertex_estimate: None,
        statistics: None,
        edge_normalize_inserted: true,
        construction_workspace_bounds: Some(GraphConstructionWorkspaceBounds::new(0, 0)),
        pair_rows: None,
        pair_vertex_width_bytes: None,
        pair_cardinality_source: None,
    }
    .with_similarity_pair_input(1, 8, GraphPairCardinalitySource::PlanHardUpperBound);
    let estimate = GraphMemoryModel.estimate_for_bounded(&spec, input).expect("small similarity model");
    assert!(estimate.requested_bytes < GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);

    let report =
        GraphReservationReport::from_estimate("n3", estimate, Some(GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES))
            .credit_live_graph_input(6 * 8, None, 0)
            .credit_live_similarity_pair_input(16);
    assert_eq!(
        report.requested_bytes,
        GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES - 6 * 8 - 16,
        "the floor applies to the complete live graph peak, so accounted input bytes reduce the incremental ask"
    );
    assert_eq!(report.outcome, GraphReservationOutcome::Admitted);
}
