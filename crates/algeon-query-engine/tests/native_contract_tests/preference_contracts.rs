use std::sync::Arc;

use query_engine::plan::{
    NativePlanCostDecision, NativePlanCostOptions, NativePlanCostReason, QueryPlan, native_plan_cost_decision,
};

use super::*;

#[test]
fn small_selective_topk_join_is_not_selected_by_cost() {
    let plan = topk_join_plan(4, 10_000, 1024 * 1024);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    let NativePlanCostDecision::NotSelectedByCost { reason, facts } = decision else {
        panic!("small top-k join should not be selected by cost");
    };
    assert_eq!(reason, NativePlanCostReason::SmallSelectiveTopK);
    assert_eq!(facts.selected_row_groups, Some(8));
    assert_eq!(facts.estimated_source_bytes, Some(2 * 1024 * 1024));
    assert!(facts.selected_source_bytes_known);
    assert_eq!(facts.join_count, 1);
    assert_eq!(facts.min_topk_fetch, Some(100));
    assert_eq!(facts.diagnostic_facts().get("selected_row_groups").map(String::as_str), Some("8"));
}

#[test]
fn tiny_residual_join_is_not_selected_by_cost() {
    let plan = residual_topk_join_plan(4);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    let NativePlanCostDecision::NotSelectedByCost { reason, facts } = decision else {
        panic!("tiny residual join should not be selected by cost");
    };
    assert_eq!(reason, NativePlanCostReason::SmallSelectiveResidualJoin);
    assert_eq!(facts.selected_row_groups, Some(8));
    assert_eq!(facts.residual_join_count, 1);
}

#[test]
fn residual_join_over_tight_threshold_continues_native() {
    let plan = residual_topk_join_plan(31);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    assert!(!decision.is_not_selected_by_cost(), "residual joins use a tighter small-shape gate than ordinary top-k");
    assert_eq!(decision.facts().selected_row_groups, Some(62));
    assert_eq!(decision.facts().residual_join_count, 1);
    assert_eq!(decision.facts().min_topk_fetch, Some(100));
}

#[test]
fn residual_join_can_use_profile_specific_row_group_threshold() {
    let plan = residual_topk_join_plan(12);
    let options =
        NativePlanCostOptions { max_small_selective_residual_join_row_groups: 24, ..NativePlanCostOptions::default() };

    let decision = native_plan_cost_decision(&plan, &options, true);

    let NativePlanCostDecision::NotSelectedByCost { reason, facts } = decision else {
        panic!("profile-specific residual threshold should decline a small low-byte residual join by cost");
    };
    assert_eq!(reason, NativePlanCostReason::SmallSelectiveResidualJoin);
    assert_eq!(facts.selected_row_groups, Some(24));
    assert_eq!(facts.estimated_source_bytes, Some(2 * 1024 * 1024));
}

#[test]
fn larger_topk_join_continues_native() {
    let plan = topk_join_plan(254, 5_000_000, 512 * 1024 * 1024);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    assert!(!decision.is_not_selected_by_cost(), "larger top-k join must not be declined by the small-shape cost gate");
    assert_eq!(decision.facts().selected_row_groups, Some(508));
}

#[test]
fn few_large_row_groups_continue_native() {
    let plan = topk_join_plan(4, 10_000, 2 << 30);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    assert!(!decision.is_not_selected_by_cost(), "selected source bytes must gate small/selective cost selection");
    assert_eq!(decision.facts().selected_row_groups, Some(8));
    assert_eq!(decision.facts().estimated_source_bytes, Some(4 << 30));
}

#[test]
fn missing_row_group_evidence_continues_native() {
    let plan = topk_join_without_row_group_metadata();

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    assert!(!decision.is_not_selected_by_cost(), "missing scan evidence should continue native selection");
    assert!(!decision.facts().selected_row_groups_known);
    assert_eq!(decision.facts().selected_row_groups, None);
}

#[test]
fn small_grouped_aggregate_without_topk_or_residual_continues_native() {
    let plan = grouped_aggregate_plan(4);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    assert!(!decision.is_not_selected_by_cost(), "small final output alone is not enough to decline by cost");
    assert_eq!(decision.facts().grouped_aggregate_count, 1);
    assert_eq!(decision.facts().min_topk_fetch, None);
}

#[test]
fn small_selective_full_sort_join_aggregate_is_not_selected_by_cost() {
    let plan = full_sort_join_aggregate_plan(4, 10_000, 1024 * 1024);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::default(), true);

    let NativePlanCostDecision::NotSelectedByCost { reason, facts } = decision else {
        panic!("small full-sort join/aggregate should not be selected by cost");
    };
    assert_eq!(reason, NativePlanCostReason::SmallSelectiveFullSort);
    assert_eq!(facts.selected_row_groups, Some(8));
    assert_eq!(facts.sort_count, 1);
    assert_eq!(facts.join_count, 1);
    assert_eq!(facts.grouped_aggregate_count, 1);
    assert_eq!(facts.min_topk_fetch, None);
}

#[test]
fn disabled_cost_options_continue_native() {
    let plan = topk_join_plan(4, 10_000, 1024 * 1024);

    let decision = native_plan_cost_decision(&plan, &NativePlanCostOptions::disabled(), true);

    assert!(!decision.is_not_selected_by_cost());
}

fn topk_join_plan(row_groups_per_source: usize, rows_per_source: u64, bytes_per_source: u64) -> QueryPlan {
    topk_join_plan_with_residual(row_groups_per_source, rows_per_source, bytes_per_source, false)
}

fn residual_topk_join_plan(row_groups_per_source: usize) -> QueryPlan {
    topk_join_plan_with_residual(row_groups_per_source, 10_000, 1024 * 1024, true)
}

fn topk_join_plan_with_residual(
    row_groups_per_source: usize,
    rows_per_source: u64,
    bytes_per_source: u64,
    residual: bool,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left_schema = key_value_schema();
    let right_schema = key_label_schema();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let left = builder.add_source(SourceNode::LocalParquet(parquet_source(
        "/tmp/cost-left.parquet",
        Arc::clone(&left_schema),
        vec!["k", "v"],
        row_groups_per_source,
        rows_per_source,
        bytes_per_source,
    )));
    let right = builder.add_source(SourceNode::LocalParquet(parquet_source(
        "/tmp/cost-right.parquet",
        Arc::clone(&right_schema),
        vec!["k", "label"],
        row_groups_per_source,
        rows_per_source,
        bytes_per_source,
    )));
    let mut join_spec = JoinSpec::new(
        JoinType::Inner,
        vec![JoinKey::new(ColumnRef::input(0, "k", "left_k"), ColumnRef::input(1, "k", "right_k"))],
    )
    .with_output_projection(vec![
        JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
        JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        JoinOutputColumn::right(ColumnRef::input(1, "label", "label"), "label"),
    ])
    .with_build_side(JoinSide::Right);
    if residual {
        join_spec = join_spec.with_residual_predicate(Expr::literal(LiteralValue::Boolean(true)));
    }
    let join = builder.add_join(left, right, join_spec, Arc::clone(&output_schema)).unwrap();
    let sort = builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn topk_join_without_row_group_metadata() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left_schema = key_value_schema();
    let right_schema = key_label_schema();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/cost-left-unknown.parquet")],
        Arc::clone(&left_schema),
        vec!["k".to_string(), "v".to_string()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new("/tmp/cost-right-unknown.parquet")],
        Arc::clone(&right_schema),
        vec!["k".to_string(), "label".to_string()],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "k", "left_k"), ColumnRef::input(1, "k", "right_k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                JoinOutputColumn::right(ColumnRef::input(1, "label", "label"), "label"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort = builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("v"), true, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

fn grouped_aggregate_plan(row_groups: usize) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let source_schema = key_value_schema();
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("sum_v", DataType::Int64, true),
    ]));
    let source = builder.add_source(SourceNode::LocalParquet(parquet_source(
        "/tmp/cost-aggregate.parquet",
        Arc::clone(&source_schema),
        vec!["k", "v"],
        row_groups,
        10_000,
        1024 * 1024,
    )));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec {
                group_exprs: vec![Expr::column("k")],
                aggr_exprs: vec![AggregateExpr {
                    function: AggregateFunction::Sum,
                    expr: Expr::column("v"),
                    alias: "sum_v".to_string(),
                    distinct: false,
                    filter: None,
                }],
                grouping_sets: None,
            },
            aggregate_schema,
        )
        .unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

fn full_sort_join_aggregate_plan(
    row_groups_per_source: usize,
    rows_per_source: u64,
    bytes_per_source: u64,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left_schema = key_value_schema();
    let right_schema = key_label_schema();
    let join_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("v", DataType::Int64, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("sum_v", DataType::Int64, true),
    ]));
    let left = builder.add_source(SourceNode::LocalParquet(parquet_source(
        "/tmp/cost-full-sort-left.parquet",
        Arc::clone(&left_schema),
        vec!["k", "v"],
        row_groups_per_source,
        rows_per_source,
        bytes_per_source,
    )));
    let right = builder.add_source(SourceNode::LocalParquet(parquet_source(
        "/tmp/cost-full-sort-right.parquet",
        Arc::clone(&right_schema),
        vec!["k", "label"],
        row_groups_per_source,
        rows_per_source,
        bytes_per_source,
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "k", "left_k"), ColumnRef::input(1, "k", "right_k"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            ])
            .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("k")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "sum_v")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort =
        builder.add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("sum_v"), true, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

fn parquet_source(
    path: &str,
    schema: Arc<Schema>,
    read_columns: Vec<&str>,
    row_groups: usize,
    rows: u64,
    bytes: u64,
) -> LocalParquetSource {
    let row_group_rows = split_evenly(rows, row_groups);
    let row_group_bytes = split_evenly(bytes, row_groups);
    LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(path)
                .with_row_groups((0..row_groups).collect())
                .with_row_group_rows(row_group_rows)
                .with_row_group_bytes(row_group_bytes),
        ],
        schema,
        read_columns.into_iter().map(str::to_string).collect(),
    )
}

fn split_evenly(total: u64, count: usize) -> Vec<u64> {
    assert!(count > 0);
    let base = total / count as u64;
    let remainder = total % count as u64;
    (0..count).map(|index| base + u64::from(index < remainder as usize)).collect()
}

fn key_value_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("v", DataType::Int64, false)]))
}

fn key_label_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), Field::new("label", DataType::Utf8, true)]))
}
