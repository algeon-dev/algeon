fn shared_roles_fixture() -> (crate::plan::QueryPlan, std::sync::Arc<arrow::datatypes::Schema>) {
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
        Expr::binary(Expr::column("vertexId"), BinaryOp::NotEq, Expr::literal(LiteralValue::Utf8(String::new()))),
        std::sync::Arc::clone(&schema),
    );
    let plan = builder.finish_gpu_resident(filtered);
    (plan, schema)
}

fn shared_two_join_fixture() -> (
    crate::plan::QueryPlan,
    crate::plan::QueryPlan,
    crate::plan::QueryPlan,
    crate::admission::AttemptCapabilityEvidence,
    std::sync::Arc<arrow::datatypes::Schema>,
    std::sync::Arc<arrow::datatypes::Schema>,
) {
    use crate::{
        expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlanBuilder},
        source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
    };

    let (roles_plan, roles_schema) = shared_roles_fixture();
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
        Expr::binary(Expr::column("similarity"), BinaryOp::Gt, Expr::literal(LiteralValue::Float64(0.0))),
        std::sync::Arc::clone(&similarity_schema),
    );
    let similarity_plan = producer_builder.finish_gpu_resident(similarity);

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
    let roles = consumer_builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("roles", std::sync::Arc::clone(&roles_schema))));
    let first_join = consumer_builder.add_join(
        similarity,
        roles,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(ColumnRef::input(0, "first", "first"), ColumnRef::input(1, "vertexId", "vertexId"))],
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
            vec![JoinKey::new(ColumnRef::input(0, "second", "second"), ColumnRef::input(1, "vertexId", "vertexId"))],
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
    let evidence =
        composed_provenance_evidence(&consumer_plan, &[("similarity", &similarity_plan), ("roles", &roles_plan)]);
    (consumer_plan, similarity_plan, roles_plan, evidence, similarity_schema, roles_schema)
}

fn composed_provenance_evidence(
    consumer_plan: &crate::plan::QueryPlan,
    producers: &[(&str, &crate::plan::QueryPlan)],
) -> crate::admission::AttemptCapabilityEvidence {
    let mut consumer_evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(consumer_plan);
    let mut evidence = Vec::with_capacity(producers.len() + 1);
    for (occurrence_id, producer_plan) in producers {
        let producer_evidence = crate::admission::AttemptCapabilityEvidence::for_query_plan(producer_plan);
        let producer_scope = producer_evidence.execution_scopes().last().expect("producer evidence has a scope");
        consumer_evidence
            .push_producer_occurrence(crate::admission::AttemptProducerOccurrence::new(
                *occurrence_id,
                producer_scope.execution_scope_fingerprint(),
                producer_scope.output_schema_fingerprint(),
            ))
            .expect("consumer evidence has a scope");
        evidence.push(producer_evidence);
    }
    evidence.push(consumer_evidence);
    crate::admission::AttemptCapabilityEvidence::aggregate(evidence)
}

fn same_schema_provenance_fixture()
-> (crate::plan::QueryPlan, crate::plan::QueryPlan, crate::plan::QueryPlan, std::sync::Arc<arrow::datatypes::Schema>) {
    use crate::{
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, LocalParquetSource, SourceNode},
    };

    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "value",
        arrow::datatypes::DataType::Int32,
        false,
    )]));
    let producer = |path: &str| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            path,
            std::sync::Arc::clone(&schema),
            vec!["value".to_string()],
        )));
        builder.finish_gpu_resident(source)
    };
    let producer_a = producer("/tmp/deferred-producer-a.parquet");
    let producer_b = producer("/tmp/deferred-producer-b.parquet");
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", std::sync::Arc::clone(&schema))));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", std::sync::Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], std::sync::Arc::clone(&schema));
    let consumer = builder.finish_with_host_sink(union);
    (consumer, producer_a, producer_b, schema)
}

fn empty_resident_handle(
    context: &crate::runtime::ExecutionContext,
    schema: &std::sync::Arc<arrow::datatypes::Schema>,
    producer_plan: Option<&crate::plan::QueryPlan>,
) -> crate::runtime::resident::GpuResultHandle {
    let runtime = context.runtime().expect("attempt runtime");
    let batch = arrow::record_batch::RecordBatch::try_new(
        std::sync::Arc::clone(schema),
        vec![std::sync::Arc::new(arrow::array::Int32Array::from(Vec::<i32>::new()))],
    )
    .expect("empty input batch");
    let mut frame = crate::runtime::resident::GpuDataFrame::new(
        cudf::DataFrame::from_record_batch(&runtime, &batch).expect("empty resident frame"),
    );
    if let Some(producer_plan) = producer_plan {
        frame =
            frame.with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(producer_plan));
    }
    crate::runtime::resident::GpuResultHandle::new(context, frame, std::sync::Arc::clone(schema))
        .expect("resident handle")
}

#[test]
fn deferred_resident_activation_rejects_same_schema_producer_substitution() {
    let (consumer, producer_a, producer_b, schema) = same_schema_provenance_fixture();
    let evidence = composed_provenance_evidence(&consumer, &[("left", &producer_a), ("right", &producer_b)]);
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
        .expect("deferred attempt admission");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("left".to_string(), empty_resident_handle(context, &schema, Some(&producer_b)));
    inputs.insert("right".to_string(), empty_resident_handle(context, &schema, Some(&producer_a)));

    let error = context
        .with_admitted_gpu_inputs(&consumer, &inputs)
        .expect_err("same-schema producers cannot substitute for one another");
    assert_eq!(error.fact_value("reason"), Some("deferred_resident_producer_binding_mismatch"));
    assert_eq!(error.fact_value("input_name"), Some("left"));
    drop(inputs);
    drop(admitted);
    attempt.finish();
}

#[test]
fn deferred_resident_activation_rejects_a_missing_occurrence() {
    let (consumer, producer_a, producer_b, schema) = same_schema_provenance_fixture();
    let evidence = composed_provenance_evidence(&consumer, &[("left", &producer_a), ("right", &producer_b)]);
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
        .expect("deferred attempt admission");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("left".to_string(), empty_resident_handle(context, &schema, Some(&producer_a)));

    let error = context
        .with_admitted_gpu_inputs(&consumer, &inputs)
        .expect_err("every admitted resident occurrence must be present");
    assert_eq!(error.fact_value("reason"), Some("deferred_resident_input_missing"));
    assert_eq!(error.fact_value("input_name"), Some("right"));
    drop(inputs);
    drop(admitted);
    attempt.finish();
}

#[test]
fn unscoped_upload_does_not_skip_scoped_resident_validation() {
    let (consumer, producer_a, producer_b, schema) = same_schema_provenance_fixture();
    let evidence = composed_provenance_evidence(&consumer, &[("right", &producer_b)]);
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
        .expect("deferred attempt admission");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let mut inputs = crate::runtime::resident::GpuResidentInputMap::new();
    inputs.insert("left".to_string(), empty_resident_handle(context, &schema, None));
    inputs.insert("right".to_string(), empty_resident_handle(context, &schema, Some(&producer_a)));

    let error = context
        .with_admitted_gpu_inputs(&consumer, &inputs)
        .expect_err("an unscoped upload cannot exempt a scoped resident input");
    assert_eq!(error.fact_value("reason"), Some("deferred_resident_producer_binding_mismatch"));
    assert_eq!(error.fact_value("input_name"), Some("right"));
    drop(inputs);
    drop(admitted);
    attempt.finish();
}

#[test]
fn deferred_resident_activation_accepts_repeated_execution_with_fresh_frames() {
    let (consumer, producer_a, producer_b, schema) = same_schema_provenance_fixture();
    let evidence = composed_provenance_evidence(&consumer, &[("left", &producer_a), ("right", &producer_b)]);
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
        .expect("deferred attempt admission");
    let admitted = attempt.acquire_context().expect("admitted context");
    let context = admitted.context();
    let mut first = crate::runtime::resident::GpuResidentInputMap::new();
    first.insert("left".to_string(), empty_resident_handle(context, &schema, Some(&producer_a)));
    first.insert("right".to_string(), empty_resident_handle(context, &schema, Some(&producer_b)));
    let activated = context.with_admitted_gpu_inputs(&consumer, &first).expect("first activation");
    let mut second = crate::runtime::resident::GpuResidentInputMap::new();
    second.insert("left".to_string(), empty_resident_handle(context, &schema, Some(&producer_a)));
    second.insert("right".to_string(), empty_resident_handle(context, &schema, Some(&producer_b)));

    activated
        .with_admitted_gpu_inputs(&consumer, &second)
        .expect("the same admitted plan may execute again with fresh frames");
    drop(second);
    drop(first);
    drop(admitted);
    attempt.finish();
}

#[test]
fn shared_resident_two_left_joins_preserve_duplicate_multiplicity() {
    let (consumer_plan, similarity_plan, roles_plan, evidence, similarity_schema, roles_schema) =
        shared_two_join_fixture();
    let service = try_guaranteed_runtime_test_service().expect("runtime-backed admission service");
    let attempt = service
        .admit_query_with_request(QueryAdmissionRequest::new().with_query_plan_capability_evidence(evidence))
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
        .retain_resident_owner(similarity_frame.device_allocated_size_bytes().unwrap())
        .expect("similarity retention fits the immutable grant");
    let similarity_handle = crate::runtime::resident::GpuResultHandle::new_with_resident_owner(
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
        .retain_resident_owner(roles_frame.device_allocated_size_bytes().unwrap())
        .expect("roles retention fits the immutable grant");
    let roles_handle = crate::runtime::resident::GpuResultHandle::new_with_resident_owner(
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
    let result = execute_plan(&activated, &consumer_plan, Some(&inputs))
        .expect("two LEFT JOIN consumer executes after exact resident refinement");
    let batch = result.output().as_host_arrow().expect("consumer host output");
    assert_eq!(batch.num_rows(), 12, "duplicate first-role matches must multiply six input pairs");
}
