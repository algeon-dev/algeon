use super::*;

pub(super) fn single_source_plan() -> QueryPlan {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
    builder.finish_gpu_resident(source).unwrap()
}

pub(super) fn between_i32(column: &str, min: i32, max: i32) -> Expr {
    Expr::binary(
        Expr::binary(Expr::column(column), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(min))),
        BinaryOp::And,
        Expr::binary(Expr::column(column), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(max))),
    )
}

pub(super) fn estimated_local_source(name: &str, key_name: &str, bytes: u64, rows: u64) -> SourceNode {
    estimated_local_source_with_fields(name, &[key_name], bytes, rows)
}

pub(super) fn estimated_local_source_with_fields(
    name: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    estimated_local_source_with_stable_id(name, name, field_names, bytes, rows)
}

pub(super) fn estimated_local_source_with_row_group_count(
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

pub(super) fn estimated_local_source_with_stable_id(
    name: &str,
    stable_id: &str,
    field_names: &[&str],
    bytes: u64,
    rows: u64,
) -> SourceNode {
    let file_indices = (0..field_names.len()).collect::<Vec<_>>();
    estimated_local_source_with_stable_id_and_file_indices(name, stable_id, field_names, &file_indices, bytes, rows)
}

pub(super) fn estimated_local_source_with_stable_id_and_file_indices(
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

pub(super) fn estimated_source_with_resolved_contract(
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

pub(super) fn estimated_local_source_with_stable_id_and_predicate(
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

pub(super) fn schema(fields: &[&str]) -> Arc<Schema> {
    Arc::new(Schema::new(fields.iter().map(|name| Field::new(*name, DataType::Int32, false)).collect::<Vec<_>>()))
}

pub(super) fn plan_has_output_name_prefix(plan: &QueryPlan, prefix: &str) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        PlanNode::Aggregate(aggregate) => aggregate.spec.aggr_exprs.iter().any(|expr| expr.alias.starts_with(prefix)),
        PlanNode::Projection(projection) => {
            projection.outputs.iter().any(|output| output.field.name().starts_with(prefix))
        },
        _ => false,
    })
}

pub(super) fn left_projection(fields: &[&str]) -> Vec<JoinOutputColumn> {
    fields.iter().map(|name| JoinOutputColumn::left(ColumnRef::input(0, *name, *name), *name)).collect()
}

pub(super) fn assert_all_node_inputs_exist(plan: &QueryPlan) {
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

pub(super) fn source_input_name(plan: &QueryPlan, node_id: NodeId) -> Option<String> {
    let node = plan.node(node_id)?;
    let PlanNode::Source(source) = &node.kind else { return None };
    match source.as_ref() {
        SourceNode::GpuResident(source) => Some(source.input_name.clone()),
        _ => None,
    }
}

pub(super) fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder
        .add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name]))
        .unwrap()
}

pub(super) fn expr_contains_int32_equality(expr: &Expr, column: &str, value: i32) -> bool {
    match expr {
        Expr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            (expr_is_column(left, column) && expr_is_int32_literal(right, value))
                || (expr_is_column(right, column) && expr_is_int32_literal(left, value))
                || expr_contains_int32_equality(left, column, value)
                || expr_contains_int32_equality(right, column, value)
        },
        Expr::Binary { left, right, .. } => {
            expr_contains_int32_equality(left, column, value) || expr_contains_int32_equality(right, column, value)
        },
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            expr_contains_int32_equality(inner, column, value)
        },
        Expr::Cast { expr, .. } | Expr::TryCast { expr, .. } => expr_contains_int32_equality(expr, column, value),
        Expr::Alias { expr, .. } => expr_contains_int32_equality(expr, column, value),
        _ => false,
    }
}

fn expr_is_column(expr: &Expr, column: &str) -> bool {
    matches!(expr, Expr::Column(expr_column) if expr_column.name() == column)
}

fn expr_is_int32_literal(expr: &Expr, value: i32) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::Int32(candidate)) if *candidate == value)
}

pub(super) fn max_direct_join_probe_consumer_count(plan: &QueryPlan) -> usize {
    let mut counts = BTreeMap::<NodeId, usize>::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let probe_input = match join.spec.build_side {
            JoinSide::Left => join.inputs[1],
            JoinSide::Right => join.inputs[0],
        };
        *counts.entry(probe_input).or_default() += 1;
    }
    counts.into_values().max().unwrap_or(0)
}

pub(super) fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", left_bytes, left_rows));
    let right = builder.add_source(estimated_local_source("right", "right_key", right_bytes, right_rows));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "left_key", "left_key"),
                    ColumnRef::input(1, "right_key", "right_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
            ])
            .with_build_side(build_side),
            output_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(join).unwrap()
}

pub(super) fn estimated_inner_residual_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_fields(
        "left_residual",
        &["left_key", "left_payload"],
        left_bytes,
        left_rows,
    ));
    let right = builder.add_source(estimated_local_source_with_fields(
        "right_residual",
        &["right_key", "right_payload"],
        right_bytes,
        right_rows,
    ));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "left_key", "left_key"),
                    ColumnRef::input(1, "right_key", "right_key"),
                )],
            )
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "left_payload", "left_payload"),
                BinaryOp::Gt,
                Expr::input_column(1, "right_payload", "right_payload"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
            ])
            .with_build_side(build_side),
            output_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(join).unwrap()
}
