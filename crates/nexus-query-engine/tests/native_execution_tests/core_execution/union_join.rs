#[test]
fn native_query_executes_union_over_gpu_resident_inputs() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(Int64Array::from(vec![30]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 20, 30]);
    admitted.finish();
}

#[test]
fn bounded_native_query_executes_union_over_gpu_resident_inputs() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(Int64Array::from(vec![30]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 20, 30]);
    admitted.finish();
}

#[test]
fn bounded_native_query_executes_join_over_gpu_resident_inputs() {
    let context = backend_context();
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("lval", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("rval", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("lval", DataType::Int64, false),
        Field::new("rval", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int64Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(Int64Array::from(vec![200, 300, 400]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&left_schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "lval", "lval"), "lval"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "rval", "rval"), "rval"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("id"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&left_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize joined rows");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[20, 30]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[200, 300]);
    admitted.finish();
}

#[test]
fn native_union_and_scalar_key_hash_join_carry_basic_list_payloads_from_gpu_inputs() {
    let context = backend_context();
    let left_batch = basic_list_payload_batch(&[3, 1]);
    let right_batch = basic_list_payload_batch(&[4, 2]);
    let payload_schema = left_batch.schema();
    let dimension_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("label", DataType::Int64, false),
    ]));
    let dimension_batch = RecordBatch::try_new(
        Arc::clone(&dimension_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(Int64Array::from(vec![1000, 2000, 3000, 4000]))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        payload_schema.field(2).clone(),
        payload_schema.field(3).clone(),
        payload_schema.field(4).clone(),
        payload_schema.field(5).clone(),
        Field::new("label", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("list_left", Arc::clone(&payload_schema))));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("list_right", Arc::clone(&payload_schema))));
    let dimension = builder
        .add_source(SourceNode::GpuResident(GpuResidentSource::new("list_dimension", Arc::clone(&dimension_schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&payload_schema));
    let join = builder.add_join(
        union,
        dimension,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "key", "payload_key"),
                nexus_query_engine::expr::ColumnRef::input(1, "key", "dimension_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "key", "key"), "key"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "ints", "ints"), "ints"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "flags", "flags"), "flags"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "tags", "tags"), "tags"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "scores", "scores"), "scores"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "label", "label"), "label"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    for (name, batch) in [("list_left", &left_batch), ("list_right", &right_batch)] {
        gpu_inputs.insert(
            name.to_string(),
            GpuResultHandle::new(
                &admitted,
                GpuDataFrame::new(DataFrame::from_record_batch(&runtime, batch).unwrap()),
                Arc::clone(&payload_schema),
            )
            .unwrap(),
        );
    }
    gpu_inputs.insert(
        "list_dimension".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &dimension_batch).unwrap()),
            Arc::clone(&dimension_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize joined list payloads");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_basic_list_payload_rows(batch, 0, 1, &[1, 2, 3, 4]);
    assert_eq!(int64_array(batch, 5).values().as_ref(), &[1000, 2000, 3000, 4000]);
    admitted.finish();
}
