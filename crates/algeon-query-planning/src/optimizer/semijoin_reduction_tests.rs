use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

use super::*;
use crate::{
    expr::{AggregateSpec, BinaryOp, LimitSpec, LiteralValue, SortExpr, SortSpec},
    plan::{JoinOutputColumn, JoinSpec, ProjectionOutput, QueryPlanBuilder},
    source::{GpuResidentSource, LocalParquetFile, LocalParquetSource, SourceNode},
};

#[test]
fn lineage_accepts_source_passthrough_key() {
    let (plan, source) = single_source_plan(vec![field("k"), field("payload")]);
    let analysis = analyze_keys(&plan, source, &[left_key("k")]).expect("source key should map");

    assert_eq!(analysis.target_subtree_id, source);
    assert!(analysis.steps.is_empty());
    assert_eq!(analysis.key_mappings[0].after_column.as_deref(), Some("k"));
}

#[test]
fn lineage_accepts_filter_and_simple_alias_projection() {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("orders", vec![field("o_orderkey"), field("payload")]));
    let filter = builder.add_filter(source, Expr::column("payload")).unwrap();
    let projection = builder
        .add_projection(
            filter,
            vec![ProjectionOutput::new(
                Expr::column("o_orderkey").alias("orderkey"),
                Field::new("orderkey", DataType::Int32, false),
            )],
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(projection).unwrap();

    let analysis = analyze_keys(&plan, projection, &[left_key("orderkey")]).expect("alias should map to source key");

    assert_eq!(analysis.target_subtree_id, source);
    assert_eq!(analysis.key_mappings[0].before_column, "orderkey");
    assert_eq!(analysis.key_mappings[0].after_column.as_deref(), Some("o_orderkey"));
}

#[test]
fn lineage_records_join_output_side_for_left_child_key() {
    let (plan, join, left, _) = two_source_join_plan(
        JoinType::Inner,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "join_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_payload", "r_payload"), "r_payload"),
        ],
    );

    let analysis = analyze_keys(&plan, join, &[left_key("join_key")]).expect("join key should map to left child");

    assert_eq!(analysis.target_subtree_id, left);
    assert_eq!(analysis.steps.len(), 1);
    assert_eq!(analysis.steps[0].crossed_join_node_id, join);
    assert_eq!(analysis.steps[0].target_side, JoinSide::Left);
    assert_eq!(analysis.steps[0].target_child_id, left);
}

#[test]
fn lineage_records_join_output_side_for_right_child_key() {
    let (plan, join, _, right) = two_source_join_plan(
        JoinType::Inner,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_payload", "l_payload"), "l_payload"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "join_key"),
        ],
    );

    let analysis = analyze_keys(&plan, join, &[left_key("join_key")]).expect("join key should map to right child");

    assert_eq!(analysis.target_subtree_id, right);
    assert_eq!(analysis.steps[0].target_side, JoinSide::Right);
}

#[test]
fn lineage_reaches_deepest_supported_inner_join_subtree() {
    let mut builder = QueryPlanBuilder::new();
    let a = builder.add_source(source("a", vec![field("a_key")]));
    let b = builder.add_source(source("b", vec![field("b_key")]));
    let c = builder.add_source(source("c", vec![field("c_key")]));
    let ab = builder
        .add_join(
            a,
            b,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "a_key", "a_key"), ColumnRef::input(1, "b_key", "b_key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "a_key", "a_key"), "a_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "b_key", "b_key"), "b_key"),
            ]),
            schema(vec![field("a_key"), field("b_key")]),
        )
        .unwrap();
    let abc = builder
        .add_join(
            ab,
            c,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "b_key", "b_key"), ColumnRef::input(1, "c_key", "c_key"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "b_key", "b_key"), "join_key")]),
            schema(vec![Field::new("join_key", DataType::Int32, false)]),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(abc).unwrap();

    let analysis = analyze_keys(&plan, abc, &[left_key("join_key")]).expect("nested key should map to b");

    assert_eq!(analysis.target_subtree_id, b);
    assert_eq!(analysis.steps.len(), 2);
    assert_eq!(analysis.steps[0].crossed_join_node_id, abc);
    assert_eq!(analysis.steps[0].target_side, JoinSide::Left);
    assert_eq!(analysis.steps[1].crossed_join_node_id, ab);
    assert_eq!(analysis.steps[1].target_side, JoinSide::Right);
}

#[test]
fn lineage_rejects_expression_derived_projection_key() {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("orders", vec![field("k")]));
    let projection = builder
        .add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::binary(Expr::column("k"), BinaryOp::Add, Expr::column("k")),
                Field::new("k2", DataType::Int32, false),
            )],
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(projection).unwrap();

    assert_eq!(
        analyze_keys(&plan, projection, &[left_key("k2")]).unwrap_err(),
        SemijoinReductionRejectedReason::ExpressionDerivedKey
    );
}

#[test]
fn lineage_rejects_ambiguous_duplicate_output_names() {
    let (plan, source) = single_source_plan(vec![field("k"), field("k")]);

    assert_eq!(
        analyze_keys(&plan, source, &[left_key("k")]).unwrap_err(),
        SemijoinReductionRejectedReason::AmbiguousKeyLineage
    );
}

#[test]
fn lineage_rejects_dropped_join_projection_key() {
    let (plan, join, _, _) = two_source_join_plan(
        JoinType::Inner,
        vec![JoinOutputColumn::right(ColumnRef::input(1, "r_payload", "r_payload"), "r_payload")],
    );

    assert_eq!(
        analyze_keys(&plan, join, &[left_key("l_key")]).unwrap_err(),
        SemijoinReductionRejectedReason::DroppedKeyLineage
    );
}

#[test]
fn lineage_rejects_required_keys_that_map_to_both_join_sides() {
    let (plan, join, _, _) = two_source_join_plan(
        JoinType::Inner,
        vec![
            JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
        ],
    );

    assert_eq!(
        analyze_keys(&plan, join, &[left_key("l_key"), left_key("r_key")]).unwrap_err(),
        SemijoinReductionRejectedReason::AmbiguousKeyLineage
    );
}

#[test]
fn lineage_rejects_non_inner_crossed_join() {
    let (plan, join, _, _) = two_source_join_plan(
        JoinType::Left,
        vec![JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key")],
    );

    assert_eq!(
        analyze_keys(&plan, join, &[left_key("l_key")]).unwrap_err(),
        SemijoinReductionRejectedReason::UnsupportedCrossedJoinType
    );
}

#[test]
fn lineage_rejects_aggregate_sort_and_limit_boundaries() {
    for plan_case in [unsupported_aggregate_plan(), unsupported_sort_plan(), unsupported_limit_plan()] {
        assert_eq!(
            analyze_keys(&plan_case.0, plan_case.1, &[left_key("k")]).unwrap_err(),
            SemijoinReductionRejectedReason::UnsupportedLeftShape
        );
    }
}

#[test]
fn cost_gate_accepts_selective_row_estimate() {
    let mut builder = QueryPlanBuilder::new();
    let target = builder.add_source(local_source("target", 1_000, Some(8_000)));
    let semi_right = builder.add_source(local_source("semi_right", 100, Some(800)));
    let plan = gate_plan(builder, target, semi_right);

    let analysis = lineage_analysis_for_target(target);
    let gate = evaluate_semijoin_cost_gate(&plan, semi_right, &analysis, 8);

    assert!(gate.accepted);
    assert_eq!(gate.threshold_multiplier, 8);
    assert_eq!(gate.signal, SemijoinReductionCostSignal::RowEstimate { semi_right_rows: 100, target_rows: 1_000 });
}

#[test]
fn cost_gate_rejects_unselective_row_estimate() {
    let mut builder = QueryPlanBuilder::new();
    let target = builder.add_source(local_source("target", 700, Some(7_000)));
    let semi_right = builder.add_source(local_source("semi_right", 100, Some(800)));
    let plan = gate_plan(builder, target, semi_right);

    let analysis = lineage_analysis_for_target(target);
    let gate = evaluate_semijoin_cost_gate(&plan, semi_right, &analysis, 8);

    assert!(!gate.accepted);
    assert_eq!(gate.signal, SemijoinReductionCostSignal::RowEstimate { semi_right_rows: 100, target_rows: 700 });
}

#[test]
fn cost_gate_prefers_filtered_keyset_evidence_over_stale_row_estimate() {
    let mut builder = QueryPlanBuilder::new();
    let target = builder.add_source(local_source("target", 700, Some(7_000)));
    let semi_right = builder.add_source(local_filtered_source("semi_right", 100, Some(800)));
    let plan = gate_plan(builder, target, semi_right);

    let analysis = lineage_analysis_crossing_to_target(target);
    let gate = evaluate_semijoin_cost_gate(&plan, semi_right, &analysis, 8);

    assert!(gate.accepted);
    assert_eq!(
        gate.signal,
        SemijoinReductionCostSignal::FilteredSourceKeysetShape {
            filtered_source_count: 1,
            semi_right_source_bytes: Some(800),
            target_rows: Some(700),
            target_bytes: Some(7_000),
            crossed_join_count: 1,
        }
    );
}

#[test]
fn cost_gate_falls_back_to_byte_estimate_when_rows_are_unavailable() {
    let mut builder = QueryPlanBuilder::new();
    let target = builder.add_source(local_source_with_bytes_only("target", 8_000));
    let semi_right = builder.add_source(local_source_with_bytes_only("semi_right", 1_100));
    let plan = gate_plan(builder, target, semi_right);

    let analysis = lineage_analysis_for_target(target);
    let gate = evaluate_semijoin_cost_gate(&plan, semi_right, &analysis, 8);

    assert!(!gate.accepted);
    assert_eq!(gate.signal, SemijoinReductionCostSignal::ByteEstimate { semi_right_bytes: 1_100, target_bytes: 8_000 });
}

fn analyze_keys(
    plan: &QueryPlan,
    left_input: NodeId,
    keys: &[JoinKey],
) -> std::result::Result<SemijoinLineageAnalysis, SemijoinReductionRejectedReason> {
    SemijoinLineageAnalyzer::new(plan).analyze_left_semi_key_lineage(left_input, keys)
}

fn lineage_analysis_for_target(target_subtree_id: NodeId) -> SemijoinLineageAnalysis {
    SemijoinLineageAnalysis {
        target_subtree_id,
        target_key_columns: vec![ColumnRef::unqualified("k")],
        key_mappings: vec![SemijoinReductionKeyMappingDiagnostics {
            before_column: "k".to_string(),
            after_column: Some("k".to_string()),
            target_subtree_id: Some(target_subtree_id),
        }],
        steps: Vec::new(),
    }
}

fn lineage_analysis_crossing_to_target(target_subtree_id: NodeId) -> SemijoinLineageAnalysis {
    let mut analysis = lineage_analysis_for_target(target_subtree_id);
    analysis.steps.push(SemijoinLineageStep {
        crossed_join_node_id: target_subtree_id,
        target_side: JoinSide::Left,
        target_child_id: target_subtree_id,
        key_mappings: Vec::new(),
    });
    analysis
}

fn single_source_plan(fields: Vec<Field>) -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("input", fields));
    let plan = builder.finish_gpu_resident(source).unwrap();
    (plan, source)
}

fn two_source_join_plan(
    join_type: JoinType,
    output_projection: Vec<JoinOutputColumn>,
) -> (QueryPlan, NodeId, NodeId, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(source("left", vec![field("l_key"), field("l_payload")]));
    let right = builder.add_source(source("right", vec![field("r_key"), field("r_payload")]));
    let output_schema = Arc::new(Schema::new(
        output_projection
            .iter()
            .map(|output| Field::new(output.output_name.as_str(), DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                join_type,
                vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
            )
            .with_output_projection(output_projection),
            output_schema,
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(join).unwrap();
    (plan, join, left, right)
}

fn unsupported_aggregate_plan() -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("input", vec![field("k")]));
    let aggregate = builder
        .add_aggregate(source, AggregateSpec::new(vec![Expr::column("k")], vec![]), schema(vec![field("k")]))
        .unwrap();
    let plan = builder.finish_gpu_resident(aggregate).unwrap();
    (plan, aggregate)
}

fn unsupported_sort_plan() -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("input", vec![field("k")]));
    let sort = builder.add_sort(source, SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)])).unwrap();
    let plan = builder.finish_gpu_resident(sort).unwrap();
    (plan, sort)
}

fn unsupported_limit_plan() -> (QueryPlan, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(source("input", vec![field("k")]));
    let limit = builder.add_limit(source, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_gpu_resident(limit).unwrap();
    (plan, limit)
}

fn source(name: &str, fields: Vec<Field>) -> SourceNode {
    SourceNode::GpuResident(GpuResidentSource::new(name, schema(fields)))
}

fn local_source(name: &str, rows: u64, bytes: Option<u64>) -> SourceNode {
    let file = LocalParquetFile::new(format!("/tmp/{name}.parquet")).with_row_group_rows(vec![rows]);
    let file =
        if let Some(bytes) = bytes { file.with_projected_row_group_bytes(vec![bytes], vec![bytes]) } else { file };
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![file],
        schema(vec![field("k")]),
        vec!["k".to_string()],
    ))
}

fn local_filtered_source(name: &str, rows: u64, bytes: Option<u64>) -> SourceNode {
    let SourceNode::LocalParquet(source) = local_source(name, rows, bytes) else {
        unreachable!("local_source always builds local Parquet")
    };
    SourceNode::LocalParquet(source.with_reader_predicate(Expr::binary(
        Expr::column("k"),
        BinaryOp::Eq,
        Expr::literal(LiteralValue::Int32(1)),
    )))
}

fn local_source_with_bytes_only(name: &str, bytes: u64) -> SourceNode {
    SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(format!("/tmp/{name}.parquet"))
                .with_projected_row_group_bytes(vec![bytes], vec![bytes]),
        ],
        schema(vec![field("k")]),
        vec!["k".to_string()],
    ))
}

/// Both gate inputs, reachable through one union root.
fn gate_plan(mut builder: QueryPlanBuilder, target: NodeId, semi_right: NodeId) -> QueryPlan {
    let root = builder.add_union(vec![target, semi_right], schema(vec![field("k")])).unwrap();
    builder.finish_gpu_resident(root).unwrap()
}

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn field(name: &str) -> Field {
    Field::new(name, DataType::Int32, false)
}

fn left_key(name: &str) -> JoinKey {
    JoinKey::new(ColumnRef::input(0, name, name), ColumnRef::input(1, name, name))
}
