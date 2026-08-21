use std::sync::Arc;

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use super::*;
use crate::{
    admission::QueryPlanCapabilityProof,
    chunk::SourceChunkPlanningOptions,
    expr::{AggregateExpr, AggregateFunction, AggregateSpec, Expr, SortExpr, SortSpec},
    graph_memory::{
        GraphMemoryAdmissionEvidence, GraphMemoryCalibrationOverride, graph_memory_bounded_admission_evidence_for_plan,
    },
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBfsParams, GraphConstructionOptions,
        GraphEdgeColumnBindings, QueryPlanBuilder,
    },
    source::{
        GpuResidentSource, LiteralBatchSource, LocalParquetFile, LocalParquetSource, SourceNode,
        ZERO_COLUMN_COUNT_COLUMN,
    },
};

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

fn zero_column_count_plan(row_group_count: usize) -> QueryPlan {
    let row_groups = (0..row_group_count).collect::<Vec<_>>();
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/admission-envelope-count.parquet")
            .with_row_groups(row_groups)
            .with_row_group_bytes(vec![4_096; row_group_count])
            .with_projected_row_group_bytes(vec![0; row_group_count], vec![0; row_group_count]),
    ]);
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(
        source,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count")],
        ),
        Arc::new(Schema::new(vec![Field::new("count", DataType::UInt64, false)])),
    );
    builder.finish_with_host_sink(aggregate)
}

fn unmodeled_native_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "unmodeled",
        Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)])),
    )));
    builder.finish_with_host_sink(source)
}

fn graph_edge_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)]))
}

fn bfs_spec() -> GraphAlgorithmSpec {
    GraphAlgorithmSpec::with_stable_schema(
        GraphAlgorithm::Bfs,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions::default(),
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(0, None))),
    )
}

fn local_graph_source(schema: Arc<Schema>) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/admission-envelope-graph.parquet")
                .with_row_groups(vec![0])
                .with_row_group_rows(vec![10])
                .with_row_group_bytes(vec![64])
                .with_projected_row_group_bytes(vec![32], vec![96]),
        ],
        schema,
        vec!["src".to_owned(), "dst".to_owned()],
    )
}

fn graph_bfs_plan_with_local_source(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
    let graph = builder.add_graph_algorithm(source, bfs_spec());
    builder.finish_with_host_sink(graph)
}

fn graph_bfs_plan_with_gpu_resident_source(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("prepared-edges", schema).with_row_count_upper_bound(10),
    ));
    let graph = builder.add_graph_algorithm(source, bfs_spec());
    builder.finish_with_host_sink(graph)
}

fn graph_memory_evidence(plan: &QueryPlan) -> GraphMemoryAdmissionEvidence {
    graph_memory_bounded_admission_evidence_for_plan(plan, GraphMemoryCalibrationOverride::none(), None)
        .expect("graph plan must produce bounded memory evidence")
}

fn source_stage_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(local_graph_source(schema)));
    builder.finish_gpu_resident(source)
}

fn unknown_source_stage_plan(schema: Arc<Schema>) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/admission-envelope-graph-unknown.parquet",
        schema,
        vec!["src".to_owned(), "dst".to_owned()],
    )));
    builder.finish_gpu_resident(source)
}

fn capability_free_literal_plan(data_type: DataType) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", data_type, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(schema)));
    builder.finish_gpu_resident(source)
}

fn one_row_literal_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![1]))])
        .expect("one-row literal must be typed");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    builder.finish_gpu_resident(source)
}

fn placeholder_literal_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::placeholder_row()));
    builder.finish_gpu_resident(source)
}

fn nullable_literal_side_input_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![Some(1), None]))])
        .expect("nullable literal side input must be typed");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
    builder.finish_gpu_resident(source)
}

fn unknown_row_bound_side_input_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("predicate", schema)));
    builder.finish_gpu_resident(source)
}

fn overflow_row_bound_side_input_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("predicate-overflow", schema).with_row_count_upper_bound(u64::MAX),
    ));
    builder.finish_gpu_resident(source)
}

fn graph_result_passthrough_plan(input_name: &str) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(input_name, schema)));
    builder.finish_with_host_sink(source)
}

fn graph_result_sort_consumer_plan(input_name: &str) -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new(input_name, Arc::clone(&schema)).with_row_count_upper_bound(20),
    ));
    let sort =
        builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("value"), false, false)]), schema);
    builder.finish_with_host_sink(sort)
}

#[test]
fn capability_free_native_fragment_keeps_an_exact_reservation_scope() {
    let plan = capability_free_literal_plan(DataType::Int64);
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan)
        .expect("an executed native fragment must issue an exact-scope proof");

    assert!(proof.required_capabilities().is_empty(), "the scope-only proof must not add a capability allowance");
    assert_eq!(proof.required_floor_bytes_for(0).unwrap(), 0, "the scope-only proof must not raise admission");

    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert_eq!(envelope.required_phase_floor_bytes(), 0, "a literal input has no source-read reservation");
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&plan)));

    let different_plan = capability_free_literal_plan(DataType::Int32);
    assert!(
        !envelope.scopes.iter().any(|scope| scope.matches(&different_plan)),
        "a scope-only proof must not authorize a different capability-free native fragment",
    );
}

#[test]
fn one_row_zero_column_literal_keeps_zero_floor() {
    let plan = placeholder_literal_plan();
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
    let envelope = proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("one-row zero-column placeholders do not require a literal byte owner");
    assert_eq!(envelope.required_phase_floor_bytes(), 0);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(0));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
}

#[test]
fn nonempty_literal_owns_its_source_stage_bound() {
    let proof = QueryPlanCapabilityProof::for_query_plan(&one_row_literal_plan()).unwrap();
    let source_terms =
        proof.as_requirement().owner_terms.iter().filter(|term| term.is_source_stage()).collect::<Vec<_>>();
    assert_eq!(source_terms.len(), 1);
    let envelope = proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("a reachable nonempty literal must carry its checked source owner");
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(source_terms[0].bytes()));
}

#[test]
fn unreachable_nonempty_literal_does_not_reject_a_reachable_empty_plan() {
    let mut builder = QueryPlanBuilder::new();
    let reachable =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
        ])))));
    let unreachable_schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let unreachable_batch =
        RecordBatch::try_new(Arc::clone(&unreachable_schema), vec![Arc::new(Int64Array::from(vec![1]))])
            .expect("unreachable literal must be typed");
    let _unreachable = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(unreachable_schema, vec![unreachable_batch])));
    let plan = builder.finish_gpu_resident(reachable);
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).unwrap();
    proof
        .resolve_reservation_envelope(reservation_policy())
        .expect("unreachable literal sources are not executed and must not reject admission");
}

#[test]
fn resident_side_input_frame_is_additive_and_retains_both_exact_scopes() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    let side_input = nullable_literal_side_input_plan();
    let side_proof = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&side_input)
        .expect("the exact nullable predicate fragment must issue a bounded side-input proof");
    let side_frame_bytes = crate::exec::join_output_upper_bound_bytes(
        2,
        &[JoinOutputColumnUpperBoundInput {
            data_type: DataType::Int64,
            input_max_string_length_bytes: None,
            input_total_string_bytes: None,
        }],
    )
    .expect("fixed-width side-input frame is finite");
    assert_eq!(side_proof.as_requirement().side_input_frame_owner_term_bytes(), vec![side_frame_bytes]);
    let proof = QueryPlanCapabilityProof::combine(graph_proof, [side_proof]).expect("proofs compose");
    let owner_sum = proof.as_requirement().conservative_owner_sum().expect("owner inventory is finite");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).expect("envelope resolves");

    assert_eq!(envelope.required_phase_floor_bytes(), owner_sum);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_sum));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&graph_plan)));
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&side_input)));

    let generic_side_proof = QueryPlanCapabilityProof::for_query_plan(&nullable_literal_side_input_plan())
        .expect("generic side-input proof issues an exact scope");
    QueryPlanCapabilityProof::combine(
        QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
            .expect("graph plan must issue its cuGraph capability proof")
            .with_graph_memory_admission_evidence(&evidence)
            .expect("finite graph evidence"),
        [generic_side_proof],
    )
    .expect("generic proof composition remains structurally valid")
    .resolve_reservation_envelope(reservation_policy())
    .expect("the literal source must remain priced when its proof is composed");
}

#[test]
fn resident_side_input_proof_prices_literal_strings_and_rejects_unbounded_string_frames() {
    let unknown = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&unknown_row_bound_side_input_plan())
        .expect_err("an unknown resident row bound cannot price a side-input frame");
    assert_eq!(unknown.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    assert_eq!(unknown.fact_value("schedule_reason"), Some("side_input_row_bound_unavailable"));
    assert_eq!(unknown.fact_value("phase"), Some("side_input_frame"));

    let variable_schema = Arc::new(Schema::new(vec![Field::new("vertex_id", DataType::Utf8, false)]));
    let mut variable_builder = QueryPlanBuilder::new();
    let variable_source =
        variable_builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::empty(Arc::clone(&variable_schema))));
    let literal = variable_builder.finish_gpu_resident(variable_source);
    QueryPlanCapabilityProof::try_for_resident_side_input_plan(&literal)
        .expect("an exact literal string batch has finite byte statistics");

    let mut resident_builder = QueryPlanBuilder::new();
    let resident_source = resident_builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("string_side_input", variable_schema).with_row_count_upper_bound(1),
    ));
    let resident = resident_builder.finish_gpu_resident(resident_source);
    let variable_error = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&resident)
        .expect_err("a resident variable-width predicate lacks an exact byte bound");
    assert_eq!(variable_error.fact_value("schedule_reason"), Some("side_input_schema_unmodeled"));

    let overflow = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&overflow_row_bound_side_input_plan())
        .expect_err("saturated side-input frame bytes must fail closed");
    assert_eq!(overflow.fact_value("schedule_reason"), Some("side_input_frame_bytes_overflow"));
}

#[test]
fn duplicate_resident_side_input_identity_is_rejected() {
    let side_input = nullable_literal_side_input_plan();
    let first = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&side_input).unwrap();
    let second = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&side_input).unwrap();
    let error = QueryPlanCapabilityProof::combine(first, [second])
        .expect_err("one exact predicate identity cannot authorize two ambiguous side-input occurrences");
    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
    assert_eq!(error.fact_value("schedule_reason"), Some("duplicate_predicate_identity"));
    assert_eq!(error.fact_value("phase"), Some("side_input_frame"));
}

#[test]
fn two_row_group_zero_column_count_envelope_covers_a_retained_partial_and_next_source_read() {
    let plan = zero_column_count_plan(2);
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("host-sink plan issues a proof");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    let minimum_read = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();

    assert_eq!(envelope.complete_phase_peak_bytes(), Some(envelope.required_phase_floor_bytes()));
    assert!(
        envelope.required_phase_floor_bytes() > minimum_read * 2,
        "the second source read must coexist with the retained first partial and scalar finalization state"
    );
}

#[test]
fn combining_an_unmodeled_fragment_clears_the_complete_phase_target() {
    let known = QueryPlanCapabilityProof::for_query_plan(&zero_column_count_plan(2))
        .expect("zero-column aggregate issues a proof");
    let known_envelope = known.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert!(known_envelope.complete_phase_peak_bytes().is_some());

    let unmodeled =
        QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
    let combined = QueryPlanCapabilityProof::combine(known, [unmodeled]).unwrap();
    let combined_envelope = combined.resolve_reservation_envelope(reservation_policy()).unwrap();

    assert_eq!(combined_envelope.complete_phase_peak_bytes(), None);
    assert!(combined_envelope.required_phase_floor_bytes() >= known_envelope.required_phase_floor_bytes());
}

#[test]
fn graph_phase_envelope_is_complete_and_bound_to_its_exact_scope() {
    let schema = graph_edge_schema();
    let plan = graph_bfs_plan_with_local_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&plan);
    let proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&plan)
        .expect("graph plan must issue its cuGraph capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    let owner_sum = proof.as_requirement().conservative_owner_sum().expect("owner inventory is finite");

    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert_eq!(envelope.required_phase_floor_bytes(), owner_sum);
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_sum));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));

    let unmodeled =
        QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("host-sink plan issues a proof");
    let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).unwrap();
    let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
}

#[test]
fn graph_phase_accounts_for_one_composed_native_source_stage_without_losing_its_exact_scope() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof");
    let source_plan = source_stage_plan(schema);
    let source_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&source_plan)
        .expect("graph-input source plan must issue its graph-input capability proof");
    let proof = QueryPlanCapabilityProof::combine(graph_proof, [source_proof])
        .expect("graph and exact source proofs compose")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    let owner_sum = proof.as_requirement().conservative_owner_sum().expect("owner inventory is finite");

    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_sum));
    assert_eq!(envelope.required_phase_floor_bytes(), owner_sum);
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
}

#[test]
fn graph_phase_rejects_an_unmodeled_marked_local_graph_input() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof");
    let known_input = source_stage_plan(Arc::clone(&schema));
    let known_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&known_input)
        .expect("known graph-input source must issue its graph-input capability proof");
    let unknown_input = unknown_source_stage_plan(schema);
    let unknown_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&unknown_input)
        .expect("unknown graph-input source must issue its graph-input capability proof");
    let proof = QueryPlanCapabilityProof::combine(graph_proof, [known_input_proof, unknown_input_proof])
        .expect("graph and graph-input proofs compose")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");

    let requirement = proof.as_requirement();
    assert_eq!(requirement.source_stage_bytes, None);
    let unknown_shape = requirement
        .reservation_shapes
        .iter()
        .find(|shape| shape.has_local_parquet_source() && !shape.has_modeled_source_stage())
        .expect("the unknown graph-input source remains an exact reservation scope");
    assert!(!unknown_shape.is_accounted_by_producer_phase());

    let error = proof
        .resolve_reservation_envelope(reservation_policy())
        .expect_err("an unmodeled local graph input must keep the graph envelope incomplete");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("attempt_owner_term_local_source_bytes_unavailable"));
}

#[test]
fn graph_phase_accounts_for_one_attested_literal_graph_input_without_widening_other_scopes() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof");
    let graph_input = one_row_literal_plan();
    let graph_input_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&graph_input)
        .expect("the exact graph-input fragment must issue a proof");
    let proof = QueryPlanCapabilityProof::combine(graph_proof, [graph_input_proof])
        .expect("graph and exact graph-input proofs compose")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    let owner_sum = proof.as_requirement().conservative_owner_sum().expect("owner inventory is finite");

    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_sum));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&graph_input)));

    let unmodeled =
        QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("unmodeled fragment has exact scope");
    let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
    let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
}

#[test]
fn graph_phase_accounts_for_one_composed_result_passthrough_without_widening_other_scopes() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof");
    let passthrough = graph_result_passthrough_plan("graph_result");
    let passthrough_proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&passthrough)
        .expect("the graph-result source handoff must issue a composed proof");
    let graph_proof = graph_proof.with_graph_memory_admission_evidence(&evidence).expect("finite graph evidence");
    let proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &passthrough,
        passthrough_proof,
        [crate::admission::ComposedGraphProducerCandidate::new(
            Some("graph_result".to_owned()),
            Some("graph_result".to_owned()),
            Some(graph_proof.clone()),
        )],
    )
    .expect("a refined graph proof and its graph-result passthrough proof compose through one schedule");
    let owner_sum = proof.as_requirement().conservative_owner_sum().expect("owner inventory is finite");

    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert!(envelope.complete_phase_peak_bytes().is_some_and(|peak| peak <= owner_sum));
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&passthrough)));

    let mismatch = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &passthrough,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&passthrough).unwrap(),
        [crate::admission::ComposedGraphProducerCandidate::new(
            Some("different_graph_result".to_owned()),
            Some("different_graph_result".to_owned()),
            Some(graph_proof),
        )],
    )
    .expect_err("the producer occurrence must match the consumer source");
    assert_eq!(mismatch.fact_value("schedule_reason"), Some("consumer_source_occurrence_missing"));

    let unmodeled =
        QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("unmodeled fragment has exact scope");
    let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
    let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
}

#[test]
fn graph_phase_accounts_for_one_composed_bounded_sort_consumer_without_widening_other_scopes() {
    let schema = graph_edge_schema();
    let graph_plan = graph_bfs_plan_with_gpu_resident_source(Arc::clone(&schema));
    let evidence = graph_memory_evidence(&graph_plan);
    let graph_proof = QueryPlanCapabilityProof::for_cugraph_graph_execution_plan(&graph_plan)
        .expect("graph plan must issue its cuGraph capability proof")
        .with_graph_memory_admission_evidence(&evidence)
        .expect("finite graph evidence");
    let sort_consumer = graph_result_sort_consumer_plan("graph_result");
    let sort_consumer_proof = QueryPlanCapabilityProof::for_composed_consumer_plan(&sort_consumer)
        .expect("the bounded graph-result sort must issue a composed proof");
    let proof = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &sort_consumer,
        sort_consumer_proof,
        [crate::admission::ComposedGraphProducerCandidate::new(
            Some("graph_result".to_owned()),
            Some("graph_result".to_owned()),
            Some(graph_proof.clone()),
        )],
    )
    .expect("a refined graph proof and its bounded sort proof compose through one schedule");

    let proof = proof
        .requirement_for_device_with_preflight(0, |_, _| {
            Ok(crate::admission::requirement::SortPreflight::new_for_test(128, 80, 256, 80))
        })
        .expect("the deterministic test preflight resolves the sort owner terms");
    let envelope = proof.resolve_reservation_envelope(reservation_policy()).unwrap();
    assert!(envelope.complete_phase_peak_bytes().unwrap() >= evidence.cold_model_requested_bytes());
    assert!(envelope.scopes.iter().any(|scope| scope.matches(&sort_consumer)));

    let unrelated = graph_result_passthrough_plan("different_graph_result");
    let mismatch = QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &unrelated,
        QueryPlanCapabilityProof::for_composed_consumer_plan(&unrelated).unwrap(),
        [crate::admission::ComposedGraphProducerCandidate::new(
            Some("graph_result".to_owned()),
            Some("graph_result".to_owned()),
            Some(graph_proof),
        )],
    )
    .expect_err("the producer occurrence must match the consumer source");
    assert_eq!(mismatch.fact_value("schedule_reason"), Some("consumer_source_occurrence_missing"));

    let unmodeled =
        QueryPlanCapabilityProof::for_query_plan(&unmodeled_native_plan()).expect("unmodeled fragment has exact scope");
    let combined = QueryPlanCapabilityProof::combine(proof, [unmodeled]).expect("proofs compose");
    let error = combined.resolve_reservation_envelope(reservation_policy()).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("graph_attempt_complete_phase_unmodeled"));
}

fn schedule_occurrence(
    occurrence_id: &str,
    scope: &str,
    isolated_peak_bytes: Option<u64>,
    retained_output_bytes: Option<u64>,
) -> (AttemptReservationProducerOccurrence, AttemptReservationProducerPhase) {
    (
        AttemptReservationProducerOccurrence::from_term_id(
            occurrence_id,
            scope,
            OwnerTermId::synthetic_for_test(occurrence_id, scope),
        ),
        AttemptReservationProducerPhase::new(isolated_peak_bytes, retained_output_bytes),
    )
}

struct ScheduleFixture {
    shape: AttemptReservationShape,
    schedule: AttemptReservationSchedule,
}

fn schedule(
    fixed_bytes: Option<u64>,
    producers: Vec<(AttemptReservationProducerOccurrence, AttemptReservationProducerPhase)>,
    consumer_scope: Option<&str>,
    consumer_peak_bytes: Option<u64>,
) -> ScheduleFixture {
    let (producer_occurrences, phases): (Vec<_>, Vec<_>) = producers.into_iter().unzip();
    let mut shape = schedule_scope(consumer_scope.unwrap_or_default());
    shape.producer_occurrences = producer_occurrences;
    let schedule = AttemptReservationSchedule::new(fixed_bytes, phases, consumer_peak_bytes);
    shape.ordered_schedule = Some(schedule.clone());
    ScheduleFixture { shape, schedule }
}

impl ScheduleFixture {
    fn producer_occurrences(&self) -> &[AttemptReservationProducerOccurrence] {
        self.shape.producer_occurrences()
    }

    fn checked_fold(&self) -> Result<u64> {
        let mut scopes = vec![self.shape.clone()];
        scopes.extend(
            self.shape
                .producer_occurrences()
                .iter()
                .map(|occurrence| schedule_scope(&occurrence.execution_scope_fingerprint)),
        );
        self.schedule.checked_fold_with_scopes(&scopes)
    }

    fn checked_fold_with_scopes(&self, scopes: &[AttemptReservationShape]) -> Result<u64> {
        self.schedule.checked_fold_with_scopes(scopes)
    }
}

fn schedule_scope(scope: &str) -> AttemptReservationShape {
    AttemptReservationShape {
        execution_scope_fingerprint: scope.to_owned(),
        output_column_evidence: crate::admission::string_payload::PlanColumnEvidence::default(),
        output_column_evidence_input_scopes: Vec::new(),
        has_local_parquet_source: false,
        complete_literal_source: false,
        complete_zero_byte_literal_source: false,
        source_stage_bytes: None,
        complete_zero_column_global_aggregate: None,
        complete_graph_memory_phase: None,
        scope_role: AttemptReservationScopeRole::Ordinary,
        sort_metadata: Vec::new(),
        sort_rejection_reason: None,
        owner_term_rejection_reason: None,
        producer_occurrences: Vec::new(),
        nested_consumer_occurrence_ids: None,
        deferred_owner_term_occurrence: None,
        ordered_schedule: None,
    }
}

fn assert_schedule_rejected(schedule: &ScheduleFixture, expected_reason: &str) {
    let error = schedule.checked_fold().expect_err("incomplete schedule must reject closed");
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some(expected_reason));
}

#[test]
fn ordered_schedule_n1_folds_fixed_producer_output_and_consumer() {
    let schedule =
        schedule(Some(10), vec![schedule_occurrence("p0", "producer", Some(20), Some(5))], Some("consumer"), Some(7));
    assert_eq!(schedule.checked_fold().unwrap(), 30, "max(10 + 20, 10 + 5 + 7)");
}

#[test]
fn ordered_schedule_folds_heterogeneous_two_producers() {
    let schedule = schedule(
        Some(10),
        vec![
            schedule_occurrence("pagerank", "p", Some(20), Some(5)),
            schedule_occurrence("forceatlas", "f", Some(40), Some(8)),
        ],
        Some("consumer"),
        Some(7),
    );
    assert_eq!(schedule.checked_fold().unwrap(), 55, "the second producer coexists with the retained first output");
    assert_ne!(schedule.checked_fold().unwrap(), 80, "scheduled producers must not be blindly summed");
}

#[test]
fn ordered_schedule_retains_equal_structural_occurrences() {
    let schedule = schedule(
        Some(4),
        vec![
            schedule_occurrence("same-0", "equal", Some(10), Some(6)),
            schedule_occurrence("same-1", "equal", Some(10), Some(6)),
        ],
        Some("consumer"),
        Some(1),
    );
    assert_eq!(schedule.producer_occurrences().len(), 2);
    assert_eq!(schedule.checked_fold().unwrap(), 20);
}

#[test]
fn ordered_schedule_folds_three_producers() {
    let schedule = schedule(
        Some(3),
        vec![
            schedule_occurrence("p0", "p0", Some(4), Some(5)),
            schedule_occurrence("p1", "p1", Some(8), Some(6)),
            schedule_occurrence("p2", "p2", Some(2), Some(7)),
        ],
        Some("consumer"),
        Some(9),
    );
    assert_eq!(schedule.checked_fold().unwrap(), 30);
}

#[test]
fn ordered_schedule_is_order_sensitive() {
    let first = schedule(
        Some(5),
        vec![
            schedule_occurrence("large", "large", Some(40), Some(1)),
            schedule_occurrence("small", "small", Some(10), Some(20)),
        ],
        Some("consumer"),
        Some(1),
    );
    let second = schedule(
        Some(5),
        vec![
            schedule_occurrence("small", "small", Some(10), Some(20)),
            schedule_occurrence("large", "large", Some(40), Some(1)),
        ],
        Some("consumer"),
        Some(1),
    );
    assert_eq!(first.checked_fold().unwrap(), 45);
    assert_eq!(second.checked_fold().unwrap(), 65);
}

#[test]
fn ordered_schedule_rejects_missing_evidence() {
    assert_schedule_rejected(
        &schedule(None, vec![schedule_occurrence("p", "p", Some(1), Some(1))], Some("c"), Some(1)),
        "missing_attempt_wide_fixed_bytes",
    );
    assert_schedule_rejected(&schedule(Some(0), Vec::new(), Some("c"), Some(1)), "missing_producer_evidence");
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), Some(1))], None, Some(1)),
        "missing_consumer_scope",
    );
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), Some(1))], Some("c"), None),
        "missing_consumer_additional_peak",
    );
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("p", "p", None, Some(1))], Some("c"), Some(1)),
        "missing_producer_isolated_peak",
    );
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("p", "p", Some(1), None)], Some("c"), Some(1)),
        "missing_producer_retained_output",
    );
}

#[test]
fn ordered_schedule_rejects_duplicate_occurrence_ids() {
    assert_schedule_rejected(
        &schedule(
            Some(0),
            vec![
                schedule_occurrence("duplicate", "p0", Some(1), Some(1)),
                schedule_occurrence("duplicate", "p1", Some(1), Some(1)),
            ],
            Some("c"),
            Some(1),
        ),
        "duplicate_producer_occurrence_id",
    );
}

#[test]
fn manifest_rejects_duplicate_occurrence_ids_even_when_retained_terms_differ() {
    let mut shape = schedule_scope("consumer");
    let error = shape
        .set_producer_occurrences([
            AttemptReservationProducerOccurrence::from_term_id(
                "duplicate",
                "producer",
                OwnerTermId::synthetic_for_test("retained-a", "producer"),
            ),
            AttemptReservationProducerOccurrence::from_term_id(
                "duplicate",
                "producer",
                OwnerTermId::synthetic_for_test("retained-b", "producer"),
            ),
        ])
        .expect_err("occurrence identity is mandatory even when retained terms differ");
    assert_eq!(error.fact_value("schedule_reason"), Some("duplicate_producer_occurrence_id"));
}

#[test]
fn manifest_rejects_a_phase_row_count_mismatch_before_schedule_attachment() {
    let mut shape = schedule_scope("consumer");
    let (occurrence, _) = schedule_occurrence("producer", "producer", Some(1), Some(1));
    shape.set_producer_occurrences([occurrence]).unwrap();
    let error = shape
        .attach_ordered_schedule(AttemptReservationSchedule::new(Some(0), Vec::new(), Some(0)))
        .expect_err("ordered phase rows must match mandatory producer occurrences");
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_phase_row_count_mismatch"));
}

#[test]
fn ordered_schedule_preserves_equal_producer_scope_multiplicity() {
    let fixture = schedule(
        Some(0),
        vec![
            schedule_occurrence("same-0", "equal", Some(1), Some(1)),
            schedule_occurrence("same-1", "equal", Some(1), Some(1)),
        ],
        Some("consumer"),
        Some(1),
    );
    let complete_scopes = vec![fixture.shape.clone(), schedule_scope("equal"), schedule_scope("equal")];
    AttemptReservationEnvelope::resolve(&complete_scopes, reservation_policy())
        .expect("equal producer scopes are separate runtime occurrences");

    let incomplete_scopes = vec![fixture.shape, schedule_scope("equal")];
    let error = AttemptReservationEnvelope::resolve(&incomplete_scopes, reservation_policy())
        .expect_err("scope multiplicity must not be collapsed during admission");
    assert_eq!(error.fact_value("schedule_reason"), Some("schedule_scope_coverage_mismatch"));
}

#[test]
fn ordered_schedule_rejects_duplicate_manifest_core() {
    let fixture = schedule(
        Some(0),
        vec![schedule_occurrence("producer", "producer", Some(1), Some(1))],
        Some("consumer"),
        Some(1),
    );
    let mut duplicate_manifest = schedule_scope("consumer");
    duplicate_manifest.producer_occurrences = fixture.shape.producer_occurrences.clone();
    let scopes = vec![fixture.shape, duplicate_manifest, schedule_scope("producer")];
    let error = AttemptReservationEnvelope::resolve(&scopes, reservation_policy())
        .expect_err("one composed attempt cannot carry two manifest cores");
    assert_eq!(error.fact_value("schedule_reason"), Some("duplicate_admitted_manifest"));
}

#[test]
fn ordered_schedule_rejects_scope_mismatch_but_allows_auxiliary_scopes() {
    let fixture =
        schedule(Some(0), vec![schedule_occurrence("p", "producer", Some(1), Some(1))], Some("consumer"), Some(1));
    let scopes = vec![fixture.shape.clone(), schedule_scope("producer"), schedule_scope("auxiliary")];
    assert_eq!(fixture.checked_fold_with_scopes(&scopes).unwrap(), 2);

    let mismatched = vec![fixture.shape.clone(), schedule_scope("different"), schedule_scope("consumer")];
    let error = fixture.checked_fold_with_scopes(&mismatched).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("schedule_scope_coverage_mismatch"));
}

#[test]
fn envelope_uses_schedule_peak_as_the_composed_floor_without_deduplicating_scopes() {
    let fixture =
        schedule(Some(10), vec![schedule_occurrence("p", "producer", Some(20), Some(5))], Some("consumer"), Some(7));
    let producer = fixture.shape;
    let mut auxiliary = schedule_scope("auxiliary");
    auxiliary.account_for_by_producer_phase();
    let scopes = vec![producer, schedule_scope("producer"), auxiliary];
    let envelope = AttemptReservationEnvelope::resolve(&scopes, reservation_policy())
        .expect("auxiliary exact scopes remain valid with a complete schedule");
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(30));
    assert_eq!(envelope.required_phase_floor_bytes(), 30);
    assert!(envelope.ordered_schedule().is_some());
    assert_eq!(envelope.scopes.len(), 3, "equal/auxiliary occurrences are not shape-deduplicated");
}

#[test]
fn envelope_does_not_sum_heterogeneous_isolated_peaks_when_schedule_is_present() {
    let fixture = schedule(
        Some(10),
        vec![
            schedule_occurrence("p", "producer-p", Some(20), Some(5)),
            schedule_occurrence("f", "producer-f", Some(40), Some(8)),
        ],
        Some("consumer"),
        Some(7),
    );
    let producer = fixture.shape;
    let scopes = vec![producer, schedule_scope("producer-p"), schedule_scope("producer-f")];
    let envelope = AttemptReservationEnvelope::resolve(&scopes, reservation_policy()).unwrap();
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(55));
    assert_ne!(envelope.complete_phase_peak_bytes(), Some(80));
}

#[test]
fn ordered_schedule_resolves_the_shared_owner_term_identity() {
    let plan = source_stage_plan(graph_edge_schema());
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("source plan must issue a proof");
    let owner_terms = proof
        .as_requirement()
        .owner_terms
        .iter()
        .map(|term| term.rebind_composed_for_test("source-occurrence"))
        .collect::<Result<Vec<_>>>()
        .expect("source terms can bind to the composed occurrence");
    let source_term = owner_terms.first().expect("source plan must carry one owner term");
    let producer_scope = canonical_execution_scope_fingerprint(&plan);
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![source_term.id().clone()]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    let mut producer = AttemptReservationShape::for_query_plan(&plan);
    producer
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "source-occurrence",
            producer_scope.clone(),
            source_term.id().clone(),
        )])
        .unwrap();
    producer.attach_ordered_schedule(schedule).unwrap();
    let envelope = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[producer, AttemptReservationShape::for_query_plan(&plan)],
        &owner_terms,
        reservation_policy(),
    )
    .expect("shared term identity must resolve before placement");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::OrderedTimeline));
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(source_term.bytes()));
}

#[test]
fn nested_schedule_carries_its_retained_output_across_inner_phases() {
    let plan = source_stage_plan(graph_edge_schema());
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("source plan must issue a proof");
    let term = proof
        .as_requirement()
        .owner_terms
        .first()
        .expect("source plan must carry one owner term")
        .rebind_composed_for_test("outer")
        .expect("retained output can bind to the outer occurrence");
    let term_id = term.id().clone();
    let scope = canonical_execution_scope_fingerprint(&plan);
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_nested_term_groups(
            vec![vec![term_id.clone()], vec![term_id.clone()]],
            vec![term_id.clone()],
            Vec::new(),
        )],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    let mut manifest = AttemptReservationShape::for_query_plan(&plan);
    manifest
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id("outer", scope, term_id)])
        .unwrap();
    manifest.attach_ordered_schedule(schedule).unwrap();

    let envelope = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[manifest, AttemptReservationShape::for_query_plan(&plan)],
        std::slice::from_ref(&term),
        reservation_policy(),
    )
    .expect("one retained output may remain live across sequential nested phases");
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(term.bytes()));
}

#[test]
fn incomplete_ordered_inventory_falls_back_to_the_conservative_owner_sum() {
    let plan = source_stage_plan(graph_edge_schema());
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("source plan must issue a proof");
    let source_stage_proof = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&plan)
        .expect("source-stage proof must issue a capability proof");
    let retained_source_term =
        proof.as_requirement().owner_terms.first().expect("source plan must carry one owner term");
    let source_stage_term = source_stage_proof
        .as_requirement()
        .owner_terms
        .first()
        .expect("graph-input proof must carry one source-stage owner term");
    let retained_output = retained_source_term
        .rebind_composed_for_test("missing-owner-terms")
        .expect("test owner term can bind to one composed occurrence");
    let unclassified = source_stage_term
        .rebind_composed_for_test("missing-owner-terms")
        .expect("test source-stage owner can bind to the manifest occurrence");
    let owner_terms = vec![retained_output.clone(), unclassified];
    let scope = canonical_execution_scope_fingerprint(&plan);
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![retained_output.id().clone()]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    let mut producer = AttemptReservationShape::for_query_plan(&plan);
    producer
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "missing-owner-terms",
            scope.clone(),
            retained_output.id().clone(),
        )])
        .unwrap();
    producer.attach_ordered_schedule(schedule).unwrap();
    let envelope = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[producer, AttemptReservationShape::for_query_plan(&plan)],
        &owner_terms,
        reservation_policy(),
    )
    .expect("a complete owner sum remains available when ordered terms are missing");
    assert_eq!(envelope.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
    assert_eq!(envelope.complete_phase_peak_bytes(), Some(owner_terms.iter().map(AllocationOwnerTerm::bytes).sum()));
}

#[test]
fn ordered_inventory_totality_distinguishes_scheduled_and_auxiliary_scopes() {
    let plan = source_stage_plan(graph_edge_schema());
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("source plan must issue a proof");
    let retained_output = proof
        .as_requirement()
        .owner_terms
        .first()
        .expect("source plan must carry one retained output owner")
        .rebind_composed_for_test("scheduled")
        .expect("retained output can bind to one composed occurrence");
    let source_stage = QueryPlanCapabilityProof::for_cugraph_graph_input_plan(&plan)
        .expect("graph-input proof must issue a source-stage owner")
        .as_requirement()
        .owner_terms
        .first()
        .expect("graph-input proof must carry one source-stage owner")
        .rebind_composed_for_test("scheduled")
        .expect("source-stage owner can bind to one composed occurrence");
    let scheduled_scope = canonical_execution_scope_fingerprint(&plan);
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![retained_output.id().clone()]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    let mut producer_scope = AttemptReservationShape::for_query_plan(&plan);
    producer_scope
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "scheduled",
            scheduled_scope.clone(),
            retained_output.id().clone(),
        )])
        .expect("producer occurrence must bind to the scheduled scope");
    producer_scope.attach_ordered_schedule(schedule).expect("ordered schedule must attach");

    let side_plan = nullable_literal_side_input_plan();
    let side_proof = QueryPlanCapabilityProof::try_for_resident_side_input_plan(&side_plan)
        .expect("side-input proof must carry a finite auxiliary frame");
    let side_term =
        side_proof.as_requirement().owner_terms.first().expect("side-input proof must carry one frame owner").clone();
    let mut side_scope = AttemptReservationShape::for_query_plan(&side_plan);
    side_scope.mark_resident_side_input_scope().expect("side-input scope must be marked once");
    let consumer_scope = AttemptReservationShape::for_query_plan(&plan);

    let incomplete = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[producer_scope.clone(), consumer_scope.clone(), side_scope.clone()],
        &[retained_output.clone(), source_stage, side_term.clone()],
        reservation_policy(),
    )
    .expect("an unreferenced scheduled term must select the complete conservative fallback");
    assert_eq!(incomplete.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));

    let incomplete_side_input = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[producer_scope, consumer_scope, side_scope],
        &[retained_output, side_term],
        reservation_policy(),
    )
    .expect("an unreferenced concurrent side-input term must use the conservative derivation");
    assert_eq!(incomplete_side_input.bound_derivation(), Some(AttemptBoundDerivation::ConservativeOwnerSum));
}

#[test]
fn invalid_retained_output_term_rejects_conservative_fallback() {
    let plan = source_stage_plan(graph_edge_schema());
    let proof = QueryPlanCapabilityProof::for_query_plan(&plan).expect("source plan must issue a proof");
    let owner_terms = proof.as_requirement().owner_terms.clone();
    let scope = canonical_execution_scope_fingerprint(&plan);
    let schedule = AttemptReservationSchedule::from_phase_term_ids(
        Vec::new(),
        vec![AttemptReservationProducerPhase::from_term_groups(vec![vec![
            owner_terms.first().expect("source plan must carry one owner term").id().clone(),
        ]])],
        AttemptReservationConsumerState::Passthrough,
        Vec::new(),
    );
    let mut producer = AttemptReservationShape::for_query_plan(&plan);
    producer
        .set_producer_occurrences([AttemptReservationProducerOccurrence::from_term_id(
            "missing-retained-output",
            scope.clone(),
            OwnerTermId::synthetic_for_test("missing-retained-output", &scope),
        )])
        .unwrap();
    producer.attach_ordered_schedule(schedule).unwrap();
    let error = AttemptReservationEnvelope::resolve_with_owner_terms(
        &[producer, AttemptReservationShape::for_query_plan(&plan)],
        &owner_terms,
        reservation_policy(),
    )
    .expect_err("a producer occurrence without retained output must fail closed");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("reason"), Some("attempt_bound_derivation_unavailable"));
    assert_eq!(error.fact_value("ordered_reason"), Some("missing_owner_term_reference"));
    assert_eq!(error.fact_value("conservative_reason"), Some("attempt_owner_term_missing"));
}

fn resolved_runtime_schedule_envelope(consumer_plan: &QueryPlan) -> AttemptReservationEnvelope {
    let fixture = schedule(
        Some(0),
        vec![
            schedule_occurrence("first", "producer-scope", Some(20), Some(5)),
            schedule_occurrence("second", "producer-scope", Some(30), Some(6)),
        ],
        Some(&canonical_execution_scope_fingerprint(consumer_plan)),
        Some(7),
    );
    let producer_scope = fixture.shape;
    AttemptReservationEnvelope::resolve(
        &[producer_scope, schedule_scope("producer-scope"), schedule_scope("producer-scope")],
        reservation_policy(),
    )
    .expect("complete runtime schedule must resolve")
}

fn resolved_runtime_conservative_envelope(consumer_plan: &QueryPlan) -> AttemptReservationEnvelope {
    let mut envelope = resolved_runtime_schedule_envelope(consumer_plan);
    envelope.ordered_schedule = None;
    envelope.bound_derivation = Some(AttemptBoundDerivation::ConservativeOwnerSum);
    envelope
}

#[test]
fn conservative_manifest_accepts_the_admitted_occurrence_order() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_conservative_envelope(&consumer_plan);
    envelope
        .validate_composed_execution(
            &consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect("conservative derivation still validates the mandatory manifest order");
}

#[test]
fn conservative_manifest_rejects_occurrence_order_mismatch() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_conservative_envelope(&consumer_plan);
    let error = envelope
        .validate_composed_execution(
            &consumer_plan,
            &["second".to_owned(), "first".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("conservative fallback must preserve producer order");
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
}

#[test]
fn conservative_manifest_rejects_occurrence_count_mismatch() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_conservative_envelope(&consumer_plan);
    let error = envelope
        .validate_composed_execution(&consumer_plan, &["first".to_owned()], envelope.required_phase_floor_bytes())
        .expect_err("conservative fallback must preserve producer multiplicity");
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_count_mismatch"));
}

#[test]
fn conservative_manifest_rejects_consumer_scope_mismatch() {
    let admitted_consumer_plan = graph_result_passthrough_plan("graph-result");
    let changed_consumer_plan = graph_result_sort_consumer_plan("graph-result");
    let envelope = resolved_runtime_conservative_envelope(&admitted_consumer_plan);
    let error = envelope
        .validate_composed_execution(
            &changed_consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("conservative fallback must preserve the attested consumer scope");
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_consumer_scope_mismatch"));
}

#[test]
fn conservative_runtime_rejects_a_missing_complete_peak_instead_of_using_the_floor() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let mut envelope = resolved_runtime_conservative_envelope(&consumer_plan);
    envelope.complete_phase_peak_bytes = None;
    let error = envelope
        .validate_composed_execution(
            &consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("conservative runtime must not silently downgrade to the floor");
    assert_eq!(error.fact_value("schedule_reason"), Some("conservative_complete_phase_peak_missing"));
}

#[test]
fn conservative_runtime_rejects_duplicate_manifest_cores() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let mut envelope = resolved_runtime_conservative_envelope(&consumer_plan);
    let duplicate = envelope
        .scopes
        .iter()
        .find(|scope| !scope.producer_occurrences.is_empty())
        .expect("conservative envelope has a manifest")
        .clone();
    envelope.scopes.push(duplicate);
    let error = envelope
        .validate_composed_execution(
            &consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("runtime must reject duplicate mandatory manifest cores");
    assert_eq!(error.fact_value("schedule_reason"), Some("duplicate_admitted_manifest"));
}

#[test]
fn composed_runtime_validation_rejects_producer_occurrence_order_mismatch() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

    let error = envelope
        .validate_composed_execution(
            &consumer_plan,
            &["second".to_owned(), "first".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("runtime execution must preserve the admitted producer order");

    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
    assert_eq!(error.fact_value("scheduled_occurrence_id"), Some("first"));
}

#[test]
fn composed_runtime_validation_rejects_producer_occurrence_count_mismatch() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

    let error = envelope
        .validate_composed_execution(&consumer_plan, &["first".to_owned()], envelope.required_phase_floor_bytes())
        .expect_err("runtime execution must not omit an admitted producer occurrence");

    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_count_mismatch"));
    assert_eq!(error.fact_value("scheduled_occurrence_count"), Some("2"));
    assert_eq!(error.fact_value("runtime_occurrence_count"), Some("1"));
}

#[test]
fn composed_runtime_validation_rejects_consumer_scope_mismatch() {
    let admitted_consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&admitted_consumer_plan);
    let changed_consumer_plan = graph_result_sort_consumer_plan("graph-result");

    let error = envelope
        .validate_composed_execution(
            &changed_consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
        )
        .expect_err("runtime execution must preserve the admitted consumer scope");

    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_consumer_scope_mismatch"));
}

#[test]
fn composed_runtime_validation_rejects_immutable_grant_below_schedule_peak() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

    let error = envelope
        .validate_composed_execution(&consumer_plan, &["first".to_owned(), "second".to_owned()], 34)
        .expect_err("runtime execution must not begin below its schedule peak");

    assert_eq!(envelope.required_phase_floor_bytes(), 35);
    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("immutable_grant_below_schedule_peak"));
    assert_eq!(error.fact_value("immutable_grant_bytes"), Some("34"));
    assert_eq!(error.fact_value("complete_phase_peak_bytes"), Some("35"));
}

#[test]
fn composed_runtime_validation_without_attempt_identity_returns_no_projection() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

    let projection = envelope
        .validate_and_project_composed_execution(
            &consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
            None,
        )
        .expect("ordinary ordered execution must validate without publishing a projection");

    assert!(projection.is_none());
}

#[test]
fn composed_runtime_validation_with_attempt_identity_returns_projection() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);
    let attempt_id = crate::observability::AttemptId::mint();

    let projection = envelope
        .validate_and_project_composed_execution(
            &consumer_plan,
            &["first".to_owned(), "second".to_owned()],
            envelope.required_phase_floor_bytes(),
            Some((attempt_id, 2, 41)),
        )
        .expect("attempt-bound ordered execution must publish a projection")
        .expect("attempt identity requests a projection");

    assert_eq!(projection.attempt_id, attempt_id);
    assert_eq!(projection.device_ordinal, 2);
    assert_eq!(projection.grant_id, 41);
}

#[test]
fn composed_runtime_validation_rejects_malformed_schedule_before_no_projection_shortcut() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);

    let error = envelope
        .validate_and_project_composed_execution(
            &consumer_plan,
            &["second".to_owned(), "first".to_owned()],
            envelope.required_phase_floor_bytes(),
            None,
        )
        .expect_err("malformed ordered execution must reject before the no-projection result");

    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("runtime_occurrence_order_mismatch"));
}

#[test]
fn composed_schedule_projection_is_lossless_and_fingerprint_stable() {
    let consumer_plan = graph_result_passthrough_plan("graph-result");
    let envelope = resolved_runtime_schedule_envelope(&consumer_plan);
    let attempt_id = crate::observability::AttemptId::mint();
    let projection = envelope
        .composed_schedule_projection(attempt_id, 2, 41, envelope.required_phase_floor_bytes())
        .expect("resolved composed schedule must publish a projection");

    assert_eq!(projection.attempt_id, attempt_id);
    assert_eq!(projection.device_ordinal, 2);
    assert_eq!(projection.grant_id, 41);
    assert_eq!(projection.attempt_wide_fixed_bytes, 0);
    assert_eq!(projection.producer_occurrences.len(), 2);
    assert_eq!(projection.producer_occurrences[0].occurrence_id, "first");
    assert_eq!(projection.producer_occurrences[0].stage_peak_bytes, 20);
    assert_eq!(projection.producer_occurrences[1].stage_peak_bytes, 35);
    assert_eq!(projection.consumer_stage_peak_bytes, 18);
    assert_eq!(projection.complete_peak_bytes, 35);
    assert!(projection.fingerprint.starts_with("composed-schedule-fnv1a64:"));
    assert!(projection.fingerprint_matches_contents());

    let mut corrupted = projection.clone();
    corrupted.fingerprint = "composed-schedule-fnv1a64:0000000000000000".to_owned();
    assert!(!corrupted.fingerprint_matches_contents());

    let repeat = envelope
        .composed_schedule_projection(attempt_id, 2, 99, envelope.required_phase_floor_bytes())
        .expect("same schedule must retain its deterministic fingerprint");
    assert_eq!(projection.fingerprint, repeat.fingerprint);
    assert_ne!(projection.grant_id, repeat.grant_id, "placement identity is not part of schedule identity");
}

#[test]
fn composed_schedule_projection_fails_closed_without_an_admitted_schedule() {
    let envelope = AttemptReservationEnvelope::resolve(&[schedule_scope("ordinary")], reservation_policy())
        .expect("ordinary scope resolves without a composed schedule");
    let error = envelope
        .composed_schedule_projection(crate::observability::AttemptId::mint(), 0, 1, 1)
        .expect_err("projection requests must fail closed when schedule evidence is absent");
    assert_eq!(error.fact_value("reason"), Some(GRAPH_ATTEMPT_COMPLETE_PHASE_UNMODELED_REASON));
    assert_eq!(error.fact_value("schedule_reason"), Some("missing_admitted_schedule"));
}

#[test]
fn ordered_schedule_rejects_empty_occurrence_id_and_scope() {
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("", "p", Some(1), Some(1))], Some("c"), Some(1)),
        "missing_producer_occurrence_id",
    );
    assert_schedule_rejected(
        &schedule(Some(0), vec![schedule_occurrence("p", "", Some(1), Some(1))], Some("c"), Some(1)),
        "missing_producer_occurrence_scope",
    );
}

#[test]
fn ordered_schedule_rejects_every_checked_fold_overflow_edge() {
    assert_schedule_rejected(
        &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(1), Some(0))], Some("c"), Some(0)),
        "producer_stage_isolated_peak_overflow",
    );
    assert_schedule_rejected(
        &schedule(
            Some(u64::MAX),
            vec![schedule_occurrence("p0", "p0", Some(0), Some(1)), schedule_occurrence("p1", "p1", Some(0), Some(0))],
            Some("c"),
            Some(0),
        ),
        "producer_stage_fixed_plus_retained_overflow",
    );
    assert_schedule_rejected(
        &schedule(
            Some(0),
            vec![
                schedule_occurrence("p0", "p0", Some(0), Some(u64::MAX)),
                schedule_occurrence("p1", "p1", Some(0), Some(1)),
            ],
            Some("c"),
            Some(0),
        ),
        "retained_output_accumulation_overflow",
    );
    assert_schedule_rejected(
        &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(0), Some(1))], Some("c"), Some(0)),
        "consumer_stage_fixed_plus_retained_overflow",
    );
    assert_schedule_rejected(
        &schedule(Some(u64::MAX), vec![schedule_occurrence("p", "p", Some(0), Some(0))], Some("c"), Some(1)),
        "consumer_stage_additional_peak_overflow",
    );
}

#[test]
fn ordered_schedule_reports_first_and_later_isolated_peak_overflow_occurrences() {
    let first = schedule(
        Some(u64::MAX),
        vec![schedule_occurrence("first", "first", Some(1), Some(0))],
        Some("consumer"),
        Some(0),
    );
    let error = first.checked_fold().expect_err("the first producer isolated term must reject on overflow");
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_stage_isolated_peak_overflow"));
    assert_eq!(error.fact_value("occurrence_id"), Some("first"));

    let later = schedule(
        Some(0),
        vec![
            schedule_occurrence("first", "first", Some(0), Some(u64::MAX)),
            schedule_occurrence("later", "later", Some(1), Some(0)),
        ],
        Some("consumer"),
        Some(0),
    );
    let error = later.checked_fold().expect_err("a later producer isolated term must reject on overflow");
    assert_eq!(error.fact_value("schedule_reason"), Some("producer_stage_isolated_peak_overflow"));
    assert_eq!(error.fact_value("occurrence_id"), Some("later"));
}
