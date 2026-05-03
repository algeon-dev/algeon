use super::*;

#[test]
fn bounded_admission_builds_strict_compatible_contract_report() {
    let plan = root_join_plan();
    let admission = bounded_pipeline_admission(&plan)
        .expect("bounded admission should not fail")
        .expect("root join should produce bounded admission evidence");

    assert!(admission.analysis().admits_bounded_join());
    let report = CapabilityContractReport::from_compiled_pipeline(admission.analysis(), admission.pipeline())
        .expect("bounded admission evidence should build contract report rows");
    assert_eq!(report.execution_mode(), CapabilityEnvelopeStatus::Admitted);
    assert!(report.gate_rows().iter().all(|row| row.execution_mode == CapabilityEnvelopeStatus::Admitted));
}

#[test]
fn bounded_admission_rejects_row_delete_sources() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("left_value", DataType::Int32, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("right_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_value", DataType::Int32, true),
    ]));
    let delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio("/tmp/compiled-row-delete-left-delete.parquet"),
        vec![1],
        vec!["left_id".into()],
    )
    .expect("test equality-delete descriptor should be valid");
    let left_file =
        LocalParquetFile::new("/tmp/compiled-row-delete-left.parquet").with_equality_delete_files(vec![delete_file]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![left_file],
        left_schema,
        vec!["left_id".into(), "left_value".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/compiled-row-delete-right.parquet",
        right_schema,
        vec!["right_id".into(), "right_value".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "left_id", "left_id"), ColumnRef::input(1, "right_id", "right_id"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_value", "right_value"), "right_value"),
        ])
        .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);

    assert!(bounded_pipeline_admission(&plan).expect("bounded admission should not fail").is_none());
    let projection = bounded_pipeline_rejection_projection(&plan)
        .expect("bounded rejection projection should not fail")
        .expect("row-delete source should produce a bounded rejection projection");
    assert_eq!(projection.reason, "strict_row_delete_application_unpriced");
    assert_eq!(projection.memory_boundary, "row_delete_application");
    assert_eq!(projection.facts.get("required_contract").map(String::as_str), Some("row_delete_application_contract"));
    assert_eq!(projection.facts.get("equality_delete_files").map(String::as_str), Some("1"));
}
