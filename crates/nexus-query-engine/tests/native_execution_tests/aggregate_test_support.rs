use std::{path::Path, sync::Arc};

use super::*;

pub(super) fn nullable_global_aggregate_plan(
    path: &Path,
    input_schema: Arc<Schema>,
    filter: Option<Expr>,
) -> nexus_query_engine::plan::QueryPlan {
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("avg_m00", DataType::Float64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m00", DataType::Int64, true),
        Field::new("count_m00", DataType::Int64, false),
        Field::new("count_rows", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m00"), "max_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count_rows"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate_input =
        if let Some(filter) = filter { builder.add_filter(source, filter, input_schema) } else { source };
    let aggregate = builder.add_aggregate(aggregate_input, aggregate, aggregated_schema);
    builder.finish_with_host_sink(aggregate)
}
