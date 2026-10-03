fn single_source_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source).unwrap()
}

fn between_i32(column: &str, min: i32, max: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column(column), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(min))),
        BinaryOp::And,
        Expr::binary(Expr::column(column), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(max))),
    )
}

fn estimated_local_source(name: &str, key_name: &str, bytes: u64, rows: u64) -> SourceNode {
    estimated_local_source_with_fields(name, &[key_name], bytes, rows)
}

fn estimated_local_source_with_fields(name: &str, field_names: &[&str], bytes: u64, rows: u64) -> SourceNode {
    estimated_local_source_with_stable_id(name, name, field_names, bytes, rows)
}

fn estimated_local_source_with_row_group_count(
    name: &str,
    field_names: &[&str],
    row_group_count: usize,
    bytes_per_row_group: u64,
    rows_per_row_group: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let row_group_bytes = vec![bytes_per_row_group; row_group_count];
    let row_group_rows = vec![rows_per_row_group; row_group_count];
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(name)
        .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
        .with_row_group_rows(row_group_rows);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], source_schema, read_columns))
}

fn estimated_local_source_with_stable_id(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let file_indices = (0..field_names.len()).collect::<Vec<_>>();
    estimated_local_source_with_stable_id_and_file_indices(name, stable_id, field_names, &file_indices, bytes, rows)
}

fn estimated_local_source_with_stable_id_and_file_indices(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    file_indices: &[usize],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet"))
        .with_stable_id(stable_id)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(vec![file], source_schema, read_columns).with_column_file_indices(
        file_indices.to_vec(),
        file_indices.to_vec(),
        Vec::new(),
    );
    SourceNode::LocalParquet(source)
}

fn estimated_source_with_resolved_contract(
    resolved_source: ResolvedParquetSource,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let source_schema = schema(field_names);
    let file = LocalParquetFile::from_resolved_parquet_source(resolved_source)
        .unwrap()
        .with_stable_id(stable_id)
        .with_projected_row_group_bytes(vec![bytes], vec![bytes])
        .with_row_group_rows(vec![rows]);
    let read_columns = field_names.iter().map(|name| (*name).to_string()).collect::<Vec<_>>();
    SourceNode::LocalParquet(LocalParquetSource::from_files(vec![file], source_schema, read_columns))
}

fn estimated_local_source_with_stable_id_and_predicate(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
    predicate: Expr,
) -> SourceNode {
    let SourceNode::LocalParquet(source) =
        estimated_local_source_with_stable_id(name, stable_id, field_names, bytes, rows)
    else {
        unreachable!("estimated source helper always builds local Parquet")
    };
    SourceNode::LocalParquet(source.with_reader_predicate(predicate))
}

fn schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

fn plan_has_output_name_prefix(plan: &QueryPlan, prefix: &str) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => aggregate.spec.aggr_exprs.iter().any(|expr| expr.alias.starts_with(prefix)),
        PlanNode::Projection(projection) => {
            projection.outputs.iter().any(|output| output.field.name().starts_with(prefix))
        },
        _ => false,
    })
}

fn left_projection(fields: &[&str]) -> Vec<JoinOutputColumn> {
    fields.iter().map(|name| JoinOutputColumn::left(ColumnRef::input(0, *name, *name), *name)).collect()
}

fn assert_all_node_inputs_exist(plan: &QueryPlan) {
    for node in plan.nodes() {
        for input in node.input_ids() {
            assert!(
                plan.node(*input).is_some(),
                "node {} has dangling input reference {} after optimizer rewrite",
                node.id,
                input
            );
        }
    }
}

fn source_input_name(
    plan: &QueryPlan,
    node_id: NodeId,
) -> Option<String> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else { return None };
    match source.as_ref() {
        SourceNode::GpuResident(source) => Some(source.input_name.clone()),
        _ => None,
    }
}
