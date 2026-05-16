use std::{fs::File, path::Path};

use arrow::{
    array::{ArrayRef, Int32Array},
    record_batch::RecordBatch,
};
use datafusion::parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

use super::*;

const NON_TPCH_SEMIJOIN_SQL: &str = "
SELECT c.c_key, o.o_key, i.i_value
FROM customers c
JOIN orders o ON c.c_key = o.o_custkey
JOIN items i ON o.o_key = i.i_orderkey
WHERE o.o_key IN (
    SELECT q.q_orderkey
    FROM qualifying_orders q
    GROUP BY q.q_orderkey
    HAVING COUNT(*) > 0
)
ORDER BY c.c_key ASC, o.o_key ASC, i.i_value ASC
LIMIT 10
";

#[tokio::test]
async fn non_tpch_sql_in_subquery_semijoin_reduction_is_generic() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-generic-semijoin-reduction");
    write_non_tpch_semijoin_fixture(&dir);

    let plan = non_tpch_semijoin_sql_plan(&dir).await;
    let lowering = lower_physical_plan_to_native(plan);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());

    let direct_plan = &lowering.plans[0];
    let direct_left_semi = native_join_ids(direct_plan, NativeJoinType::LeftSemi);
    assert_eq!(direct_left_semi.len(), 1, "direct SQL lowering should expose one left-semi candidate");
    assert!(
        source_tables_below(direct_plan, direct_left_semi[0].1.inputs[0]).len() > 1,
        "pre-optimizer left-semi should still sit above an inner-join tree"
    );

    let optimized = cudf_query_engine::optimize_query_plan(direct_plan.clone(), &Default::default());
    assert_eq!(optimized.change, cudf_query_engine::NativeOptimizerChange::Changed);
    assert_eq!(optimized.diagnostics.semijoin_reduction.considered_count, 1);
    assert_eq!(optimized.diagnostics.semijoin_reduction.applied_count, 1);
    assert!(
        optimized
            .diagnostics
            .semijoin_reduction
            .candidates
            .iter()
            .all(|candidate| candidate.rule_name == cudf_query_engine::SEMIJOIN_REDUCTION_RULE_NAME)
    );

    let optimized_left_semi = native_join_ids(&optimized.plan, NativeJoinType::LeftSemi);
    assert_eq!(optimized_left_semi.len(), 1, "optimized plan should still have one pushed left-semi");
    let (_, semi_join) = optimized_left_semi[0];
    assert_eq!(
        source_tables_below(&optimized.plan, semi_join.inputs[0]),
        BTreeSet::from(["orders".to_string()]),
        "generic SQL semijoin should push to the orders subtree that owns o_key"
    );
    assert_eq!(
        source_tables_below(&optimized.plan, semi_join.inputs[1]),
        BTreeSet::from(["qualifying_orders".to_string()]),
        "semijoin keyset branch should remain the grouped qualifying_orders input"
    );

    fs::remove_dir_all(dir).unwrap();
}

async fn non_tpch_semijoin_sql_plan(root: &Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    for table in ["customers", "orders", "items", "qualifying_orders"] {
        let path = root.join(format!("{table}.parquet"));
        ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
    let df = ctx.sql(NON_TPCH_SEMIJOIN_SQL).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

fn write_non_tpch_semijoin_fixture(root: &Path) {
    fs::create_dir_all(root).unwrap();
    write_i32_table(&root.join("customers.parquet"), vec![field("c_key")], vec![vec![1, 2, 3]]);
    write_i32_table(
        &root.join("orders.parquet"),
        vec![field("o_key"), field("o_custkey")],
        vec![vec![1, 2, 3, 4], vec![1, 2, 3, 2]],
    );
    write_i32_table(
        &root.join("items.parquet"),
        vec![field("i_orderkey"), field("i_value")],
        vec![vec![1, 2, 3, 4], vec![10, 20, 30, 40]],
    );
    write_i32_table(&root.join("qualifying_orders.parquet"), vec![field("q_orderkey")], vec![vec![2, 2, 4]]);
}

fn write_i32_table(path: &Path, fields: Vec<Field>, columns: Vec<Vec<i32>>) {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        columns.into_iter().map(|values| Arc::new(Int32Array::from(values)) as ArrayRef).collect(),
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn field(name: &str) -> Field {
    Field::new(name, DataType::Int32, false)
}

fn native_join_ids(plan: &QueryPlan, join_type: NativeJoinType) -> Vec<(NodeId, &cudf_query_engine::JoinNode)> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Join(join) if join.spec.join_type == join_type => Some((node.id, join)),
            _ => None,
        })
        .collect()
}

fn source_tables_below(plan: &QueryPlan, root: NodeId) -> BTreeSet<String> {
    let mut tables = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        let node = plan.node(node_id).expect("native subtree node should exist");
        if let PlanNode::Source(source) = &node.kind
            && let SourceNode::LocalParquet(source) = source.as_ref()
            && let Some(table) = local_parquet_source_table(source)
        {
            tables.insert(table.to_string());
        }
        stack.extend(node.input_ids().iter().copied());
    }
    tables
}
