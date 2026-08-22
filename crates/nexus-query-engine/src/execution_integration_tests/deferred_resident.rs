fn shared_roles_fixture() -> (
    crate::plan::QueryPlan,
    crate::admission::QueryPlanCapabilityProof,
    std::sync::Arc<arrow::datatypes::Schema>,
) {
    use crate::{
        expr::{BinaryOp, Expr, LiteralValue},
        source::{LocalParquetFile, LocalParquetSource, SourceNode},
    };

    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("vertexId", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("roleName", arrow::datatypes::DataType::Utf8, false),
    ]));
    let file = LocalParquetFile::new("/tmp/deferred-roles.parquet")
        .with_row_groups(vec![0])
        .with_row_group_rows(vec![100])
        .with_row_group_bytes(vec![4_096])
        .with_projected_row_group_bytes(vec![4_096], vec![4_096])
        .with_variable_width_payload_evidence(std::collections::BTreeMap::from([
            (
                0,
                crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                    0,
                    vec![0],
                    400,
                    crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
            (
                1,
                crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                    1,
                    vec![0],
                    800,
                    crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
        ]));
    let mut builder = crate::plan::QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        std::sync::Arc::clone(&schema),
        vec!["vertexId".to_owned(), "roleName".to_owned()],
    )));
    let filtered = builder.add_filter(
        source,
        Expr::binary(
            Expr::column("vertexId"),
            BinaryOp::NotEq,
            Expr::literal(LiteralValue::Utf8(String::new())),
        ),
        std::sync::Arc::clone(&schema),
    );
    let plan = builder.finish_gpu_resident(filtered);
    let proof = crate::admission::QueryPlanCapabilityProof::for_query_plan(&plan)
        .expect("filtered roles producer must issue a bounded source proof");
    (plan, proof, schema)
}

fn shared_two_join_fixture() -> (
    crate::plan::QueryPlan,
    crate::plan::QueryPlan,
    crate::plan::QueryPlan,
    crate::admission::QueryPlanCapabilityProof,
    std::sync::Arc<arrow::datatypes::Schema>,
    std::sync::Arc<arrow::datatypes::Schema>,
) {
    use crate::{
        admission::ComposedGraphProducerCandidate,
        expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    let (roles_plan, roles_proof, roles_schema) = shared_roles_fixture();
    let similarity_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("first", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("second", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("similarity", arrow::datatypes::DataType::Float64, false),
    ]));
    let similarity_file = LocalParquetFile::new("/tmp/deferred-similarity.parquet")
        .with_row_groups(vec![0])
        .with_row_group_rows(vec![6])
        .with_row_group_bytes(vec![512])
        .with_projected_row_group_bytes(vec![512], vec![512])
        .with_variable_width_payload_evidence(std::collections::BTreeMap::from([
            (
                0,
                crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                    0,
                    vec![0],
                    6,
                    crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
            (
                1,
                crate::parquet_stats::ParquetVariableWidthPayloadEvidence::new(
                    1,
                    vec![0],
                    6,
                    crate::parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
        ]));
    let mut producer_builder = QueryPlanBuilder::new();
    let similarity = producer_builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![similarity_file],
        std::sync::Arc::clone(&similarity_schema),
        vec!["first".to_owned(), "second".to_owned(), "similarity".to_owned()],
    )));
    let similarity = producer_builder.add_filter(
        similarity,
        Expr::binary(
            Expr::column("similarity"),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Float64(0.0)),
        ),
        std::sync::Arc::clone(&similarity_schema),
    );
    let similarity_plan = producer_builder.finish_gpu_resident(similarity);
    let similarity_proof = crate::admission::QueryPlanCapabilityProof::for_query_plan(&similarity_plan)
        .expect("filtered similarity producer must issue a bounded source proof");

    let first_join_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("first", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("second", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("similarity", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("first_role", arrow::datatypes::DataType::Utf8, true),
    ]));
    let output_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("first", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("second", arrow::datatypes::DataType::Utf8, false),
        arrow::datatypes::Field::new("similarity", arrow::datatypes::DataType::Float64, false),
        arrow::datatypes::Field::new("first_role", arrow::datatypes::DataType::Utf8, true),
        arrow::datatypes::Field::new("second_role", arrow::datatypes::DataType::Utf8, true),
    ]));
    let mut consumer_builder = QueryPlanBuilder::new();
    let similarity = consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "similarity",
        std::sync::Arc::clone(&similarity_schema),
    )));
    let roles = consumer_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "roles",
        std::sync::Arc::clone(&roles_schema),
    )));
    let first_join = consumer_builder.add_join(
        similarity,
        roles,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "first", "first"),
                ColumnRef::input(1, "vertexId", "vertexId"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "first", "first"), "first"),
            JoinOutputColumn::left(ColumnRef::input(0, "second", "second"), "second"),
            JoinOutputColumn::left(ColumnRef::input(0, "similarity", "similarity"), "similarity"),
            JoinOutputColumn::right(ColumnRef::input(1, "roleName", "roleName"), "first_role"),
        ]),
        std::sync::Arc::clone(&first_join_schema),
    );
    let second_join = consumer_builder.add_join(
        first_join,
        roles,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                ColumnRef::input(0, "second", "second"),
                ColumnRef::input(1, "vertexId", "vertexId"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "first", "first"), "first"),
            JoinOutputColumn::left(ColumnRef::input(0, "second", "second"), "second"),
            JoinOutputColumn::left(ColumnRef::input(0, "similarity", "similarity"), "similarity"),
            JoinOutputColumn::left(ColumnRef::input(0, "first_role", "first_role"), "first_role"),
            JoinOutputColumn::right(ColumnRef::input(1, "roleName", "roleName"), "second_role"),
        ]),
        std::sync::Arc::clone(&output_schema),
    );
    let consumer_plan = consumer_builder.finish_with_host_sink(second_join);
    let consumer = crate::admission::QueryPlanCapabilityProof::for_composed_consumer_plan(&consumer_plan)
        .expect("two LEFT JOIN consumer must issue a structural proof");
    let proof = crate::admission::QueryPlanCapabilityProof::resolve_composed_graph_attempt(
        &consumer_plan,
        consumer,
        [
            ComposedGraphProducerCandidate::new(
                Some("similarity".to_owned()),
                Some("similarity".to_owned()),
                Some(similarity_proof),
            ),
            ComposedGraphProducerCandidate::new(
                Some("roles".to_owned()),
                Some("roles".to_owned()),
                Some(roles_proof),
            ),
        ],
    )
    .expect("runtime-refinable roles must defer the shared two-join consumer");
    (consumer_plan, similarity_plan, roles_plan, proof, similarity_schema, roles_schema)
}

#[test]
fn shared_resident_two_left_joins_preserve_duplicate_multiplicity() {
    let (consumer_plan, similarity_plan, roles_plan, proof, similarity_schema, roles_schema) =
        shared_two_join_fixture();
    assert_eq!(proof.soft_working_set_hint_bytes(), None);
    assert!(
        proof
            .as_requirement()
            .deferred_resident_phase_for_plan(&consumer_plan)
            .expect("the composed scope must be unambiguous")
            .is_some(),
        "resident string outputs must defer until their exact max lengths are measured"
    );
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed admission service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_proof(proof))
        .expect("deferred attempt admission");
    let immutable_grant = attempt.grant_identity().expect("one immutable grant identity");
    let admitted = attempt.acquire_context().expect("admitted execution context");
    let context = admitted.context();
    let runtime = context.runtime().expect("attempt runtime");

    let similarity_batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&similarity_schema),
        vec![
            std::sync::Arc::new(arrow::array::StringArray::from(vec!["a"; 6])),
            std::sync::Arc::new(arrow::array::StringArray::from(vec!["b"; 6])),
            std::sync::Arc::new(arrow::array::Float64Array::from(vec![0.5; 6])),
        ],
    )
    .expect("valid similarity frame");
    let roles_batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(&roles_schema),
        vec![
            std::sync::Arc::new(arrow::array::StringArray::from(vec!["a", "a", "b", "c"])),
            std::sync::Arc::new(arrow::array::StringArray::from(vec!["owner", "admin", "reader", "writer"])),
        ],
    )
    .expect("valid duplicate-key roles frame");
    let grant = context.memory_grant().unwrap().unwrap();
    let similarity_frame = crate::runtime::resident::GpuDataFrame::new(
        cudf::DataFrame::from_record_batch(&runtime, &similarity_batch).expect("similarity upload"),
    )
    .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(&similarity_plan));
    let similarity_reservation = grant
        .retain_live_reservation(similarity_frame.device_allocated_size_bytes().unwrap())
        .expect("similarity retention fits the immutable grant");
    let similarity_handle = crate::runtime::resident::GpuResultHandle::new_with_live_reservation(
        context,
        similarity_frame,
        std::sync::Arc::clone(&similarity_schema),
        similarity_reservation,
    )
    .expect("similarity resident handle");
    let roles_frame = crate::runtime::resident::GpuDataFrame::new(
        cudf::DataFrame::from_record_batch(&runtime, &roles_batch).expect("roles upload"),
    )
    .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(&roles_plan));
    let roles_reservation = grant
        .retain_live_reservation(roles_frame.device_allocated_size_bytes().unwrap())
        .expect("roles retention fits the immutable grant");
    let roles_handle = crate::runtime::resident::GpuResultHandle::new_with_live_reservation(
        context,
        roles_frame,
        std::sync::Arc::clone(&roles_schema),
        roles_reservation,
    )
    .expect("roles resident handle");
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("similarity".to_owned(), similarity_handle);
    inputs.insert("roles".to_owned(), roles_handle);

    let activated = context
        .with_admitted_gpu_inputs(&consumer_plan, &inputs)
        .expect("resident rows and string payloads activate the deferred consumer");
    assert_eq!(
        activated.memory_grant().unwrap().unwrap().identity(),
        immutable_grant,
        "activation must retain the original immutable grant"
    );
    let result = crate::exec::execute_query_plan_with_gpu_inputs(&activated, &consumer_plan, &inputs)
        .expect("two LEFT JOIN consumer executes after exact resident refinement");
    let batch = result.output().as_host_arrow().expect("consumer host output");
    assert_eq!(batch.num_rows(), 12, "duplicate first-role matches must multiply six input pairs");

}
