use std::{path::Path, sync::Arc};

use query_engine::{dimension_pruning::local_dimension_pruning_resolver, rapids_backend::rapids_analysis_context};
use query_planning::{
    expr::ColumnRef,
    optimizer::{NativeOptimizerChange, optimize_query_plan},
};

use super::*;

#[test]
fn dimension_derived_row_group_pruning_uses_filtered_dimension_key_range() {
    let dir = unique_temp_dir("query-engine-dimension-pruning");
    let dim_path = dir.join("dim.parquet");
    let fact_path = dir.join("fact.parquet");

    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("day", DataType::Date32, false),
    ]));
    let dim_batch = RecordBatch::try_new(
        Arc::clone(&dim_schema),
        vec![Arc::new(Int32Array::from(vec![10, 30, 50])), Arc::new(Date32Array::from(vec![10, 30, 30]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&dim_path, Arc::clone(&dim_schema), dim_batch, 1);

    let fact_schema = Arc::new(Schema::new(vec![
        Field::new("fact_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));
    let fact_batch = RecordBatch::try_new(
        Arc::clone(&fact_schema),
        vec![Arc::new(Int32Array::from(vec![10, 20, 30, 40])), Arc::new(Int32Array::from(vec![100, 200, 300, 400]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&fact_path, Arc::clone(&fact_schema), fact_batch, 1);

    let plan = dimension_pruning_plan(&dim_path, &dim_schema, &fact_path, &fact_schema);
    let optimizer_result = optimize_query_plan(
        plan,
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(optimizer_result.change, NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.dimension_derived_row_group_pruning.applied_count, 1);
    let candidate = optimizer_result
        .diagnostics
        .dimension_derived_row_group_pruning
        .candidates
        .iter()
        .find(|candidate| candidate.applied)
        .expect("dimension-derived pruning should apply");
    assert_eq!(candidate.derived_min_key, Some(30));
    assert_eq!(candidate.derived_max_key, Some(50));
    assert_eq!(candidate.selected_row_groups_before, Some(4));
    // The qualifying dimension key set is {30, 50}; its bounds are [30, 50].
    // Set-aware pruning keeps only the fact row group whose key (30) is an actual
    // qualifying key and prunes fact_key=40; which falls inside the [30,50]
    // bounding range but matches no dimension key, so it joins to nothing. A
    // single-range filter would have wrongly retained it.
    assert_eq!(candidate.selected_row_groups_after, Some(1));

    let fact_source = optimizer_result
        .plan
        .nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) if source.files.first().is_some_and(|file| file.path == fact_path) => {
                    Some(source)
                },
                _ => None,
            },
            _ => None,
        })
        .expect("optimized plan should retain fact source");
    assert_eq!(fact_source.files[0].row_groups.as_deref(), Some(&[2][..]));

    fs::remove_dir_all(dir).unwrap();
}

fn dimension_pruning_plan(
    dim_path: &Path,
    dim_schema: &Arc<Schema>,
    fact_path: &Path,
    fact_schema: &Arc<Schema>,
) -> QueryPlan {
    let mut dim_source =
        LocalParquetSource::from_path(dim_path, Arc::clone(dim_schema), vec!["dim_key".to_string(), "day".to_string()]);
    dim_source = dim_source.with_reader_predicate(Expr::binary(
        Expr::binary(Expr::column("day"), BinaryOp::GtEq, Expr::literal(LiteralValue::Date32(15))),
        BinaryOp::And,
        Expr::binary(Expr::column("day"), BinaryOp::LtEq, Expr::literal(LiteralValue::Date32(35))),
    ));
    let fact_source = LocalParquetSource::from_path(
        fact_path,
        Arc::clone(fact_schema),
        vec!["fact_key".to_string(), "payload".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(SourceNode::LocalParquet(dim_source));
    let dim = builder
        .add_projection(
            dim,
            vec![ProjectionOutput::new(Expr::column("dim_key"), Field::new("dim_key", DataType::Int32, false))],
        )
        .unwrap();
    let fact = builder.add_source(SourceNode::LocalParquet(fact_source));
    let fact = builder
        .add_projection(
            fact,
            vec![
                ProjectionOutput::new(Expr::column("fact_key"), Field::new("fact_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ],
        )
        .unwrap();
    let join = builder
        .add_join(
            dim,
            fact,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "dim_key", "dim_key"),
                    ColumnRef::input(1, "fact_key", "fact_key"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "dim_key", "dim_key"), "dim_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "payload", "payload"), "payload"),
            ]),
            Arc::clone(&output_schema),
        )
        .unwrap();
    builder.finish_with_host_sink(join).unwrap()
}
