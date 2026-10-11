use std::sync::Arc;

use super::*;

#[test]
fn native_query_rejects_duplicate_projection_output_names_generically_before_runtime_materialization() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let duplicate_projection_schema = Arc::new(Schema::new(vec![
        Field::new("dup", DataType::Int32, false),
        Field::new("dup", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing_input", Arc::clone(&input_schema))));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![Expr::column("key").alias("dup"), Expr::column("payload").alias("dup")],
                duplicate_projection_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => {
            panic!("native validation should reject duplicate projection output names before runtime materialization")
        },
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.plan.validate_output_names");
    assert_eq!(error.fact_value("duplicate_output_name"), Some("dup"));
    assert_eq!(error.fact_value("first_output_index"), Some("0"));
    assert_eq!(error.fact_value("duplicate_output_index"), Some("1"));
    assert_eq!(error.fact_value("output_names"), Some("dup,dup"));
    assert_eq!(error.fact_value("node_kind"), Some("projection"));
    assert!(error.fact_value("input_name").is_none());
}

#[test]
fn native_projection_schema_coercion_rejects_unsupported_output_type_mismatch() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_key", DataType::Utf8, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("int_input", Arc::clone(&input_schema)).with_row_count_upper_bound(3),
    ));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(vec![Expr::column("key").alias("bad_key")], Arc::clone(&output_schema)),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(projection).unwrap();
    let error = match context.admit_plan(&plan) {
        Ok(_) => panic!("projection schema coercion should reject unsupported Int32 to Utf8 mismatch before grant"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.projection.schema_coerce");
    assert_eq!(error.fact_value("reason"), Some("projection_output_schema_mismatch"));
    assert_eq!(error.fact_value("reason_detail"), Some("implicit_cast_not_supported"));
    assert_eq!(error.fact_value("field_name"), Some("bad_key"));
    assert_eq!(error.fact_value("field_index"), Some("0"));
    assert_eq!(error.fact_value("actual_type"), Some("Int32"));
    assert_eq!(error.fact_value("expected_type"), Some("Utf8"));
}
