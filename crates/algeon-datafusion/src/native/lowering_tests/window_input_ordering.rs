use std::{fs::File, path::Path};

use arrow::{
    array::{ArrayRef, Int64Array},
    record_batch::RecordBatch,
};
use datafusion::{parquet::arrow::ArrowWriter, physical_plan::displayable};
use query_planning::plan::{WindowFrame, WindowFunction, WindowNode};

use super::*;
use crate::native::{ParquetReadStrategyConfig, reestablish_declared_output_ordering_if_needed};

const FACT_ROWS: i64 = 5_000;
const DIMENSION_ROWS: i64 = 997;

/// A fact table `t` and a smaller dimension `u`. DataFusion builds the hash
/// table on the dimension and probes with the fact table, so a sort on the
/// fact columns is pushed below the join.
fn write_window_join_tables(dir: &Path) {
    let write = |name: &str, columns: Vec<(&str, Vec<i64>)>| {
        let schema = Arc::new(Schema::new(
            columns.iter().map(|(column, _)| Field::new(*column, DataType::Int64, false)).collect::<Vec<_>>(),
        ));
        let arrays = columns.into_iter().map(|(_, values)| Arc::new(Int64Array::from(values)) as ArrayRef).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), arrays).unwrap();
        let mut writer = ArrowWriter::try_new(File::create(dir.join(name)).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    };
    write(
        "t.parquet",
        vec![
            ("id", (0..FACT_ROWS).collect()),
            ("k", (0..FACT_ROWS).map(|row| (row * 31) % DIMENSION_ROWS).collect()),
            ("a", (0..FACT_ROWS).map(|row| (row * 7919) % 53).collect()),
            ("b", (0..FACT_ROWS).map(|row| (row * 131) % 211).collect()),
            ("v", (0..FACT_ROWS).map(|row| row % 1000).collect()),
        ],
    );
    write(
        "u.parquet",
        vec![("k", (0..DIMENSION_ROWS).collect()), ("w", (0..DIMENSION_ROWS).map(|row| row * 3).collect())],
    );
}

async fn window_join_sql_plan(dir: &Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(direct_friendly_sql_session_config());
    for table in ["t", "u"] {
        let path = dir.join(format!("{table}.parquet"));
        ctx.register_parquet(table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
    ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
}

fn sort_nodes(plan: &QueryPlan) -> Vec<NodeId> {
    plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Sort(_))).map(|node| node.id).collect()
}

fn window_node(plan: &QueryPlan) -> &WindowNode {
    plan.nodes()
        .iter()
        .find_map(|node| match &node.kind {
            PlanNode::Window(window) => Some(window),
            _ => None,
        })
        .unwrap_or_else(|| panic!("the native plan has a window:\n{plan}"))
}

fn assert_hash_join_probe_is_sorted_below_the_join(plan: &Arc<dyn ExecutionPlan>) {
    let tree = displayable(plan.as_ref()).indent(false).to_string();
    let join = tree.find("HashJoinExec").unwrap_or_else(|| panic!("DataFusion plans a hash join:\n{tree}"));
    let sort = tree.find("SortExec").unwrap_or_else(|| panic!("DataFusion plans a sort:\n{tree}"));
    assert!(join < sort && tree.matches("SortExec").count() == 1, "DataFusion sorts only below the join:\n{tree}");
}

/// DataFusion pushes the window's sort through the hash join to its probe
/// input and relies on the join to keep the probe order. The native join does
/// not, so the sort moves to the window's input, and no sort remains below the
/// join.
#[tokio::test]
async fn window_over_a_hash_join_reads_one_sort_of_its_keys_above_the_join() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-over-join");
    write_window_join_tables(&dir);

    for (sql, expected_keys) in [
        (
            "SELECT t.a, t.b, t.id, LEAD(t.v) OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS next \
             FROM t JOIN u ON t.k = u.k",
            ["a", "b", "id"].as_slice(),
        ),
        (
            "SELECT t.a, t.b, t.id, ROW_NUMBER() OVER (PARTITION BY t.a ORDER BY t.b, t.id) AS rn \
             FROM t JOIN u ON t.k = u.k",
            ["a", "b", "id"].as_slice(),
        ),
        ("SELECT t.a, t.id, ROW_NUMBER() OVER (PARTITION BY t.a) AS rn FROM t JOIN u ON t.k = u.k", ["a"].as_slice()),
    ] {
        let plan = window_join_sql_plan(&dir, sql).await;
        assert_hash_join_probe_is_sorted_below_the_join(&plan);

        let (report, native_plan) = optimize_with_native_rule(plan);

        assert!(
            matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]),
            "{sql}\noutcomes={:?}",
            report.outcomes()
        );
        let native_plan = native_plan.expect("the rule keeps the native plan");
        let window = window_node(&native_plan);
        let Some(PlanNode::Sort(sort)) = native_plan.node(window.input).map(|node| &node.kind) else {
            panic!("the window reads a sort of its keys:\n{native_plan}");
        };
        assert_eq!(sort_nodes(&native_plan).len(), 1, "{sql}\n{native_plan}");
        assert!(
            matches!(native_plan.node(sort.input).map(|node| &node.kind), Some(PlanNode::Join(_))),
            "{native_plan}"
        );
        let keys = sort
            .spec
            .expressions
            .iter()
            .map(|key| match &key.expr {
                Expr::Column(column) => column.name().to_string(),
                other => panic!("sort key is a column: {other}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(keys, expected_keys, "{sql}");
    }

    fs::remove_dir_all(dir).unwrap();
}

/// A declared root ordering that only the probe-side sort provided is
/// reproduced by the one sort the adapter appends at the root.
#[tokio::test]
async fn ordered_join_output_sorts_once_at_the_root() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-ordered-join");
    write_window_join_tables(&dir);
    let plan =
        window_join_sql_plan(&dir, "SELECT t.a, t.b, t.id FROM t JOIN u ON t.k = u.k ORDER BY t.a, t.b, t.id").await;
    assert_hash_join_probe_is_sorted_below_the_join(&plan);

    let (report, native_plan) = optimize_with_native_rule(plan);

    assert!(matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]), "outcomes={:?}", report.outcomes());
    let native_plan = native_plan.expect("the rule keeps the native plan");
    assert_eq!(sort_nodes(&native_plan), vec![native_plan.root_id()], "{native_plan}");

    fs::remove_dir_all(dir).unwrap();
}

/// A sort the plan already carries below the window stays the window's only
/// sort.
#[tokio::test]
async fn window_over_a_sorted_scan_adds_no_second_sort() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-over-scan");
    write_window_join_tables(&dir);
    let plan =
        window_join_sql_plan(&dir, "SELECT a, id, LEAD(v) OVER (PARTITION BY a ORDER BY b, id) AS next FROM t").await;

    let (report, native_plan) = optimize_with_native_rule(plan);

    assert!(matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]), "outcomes={:?}", report.outcomes());
    assert_eq!(sort_nodes(&native_plan.expect("the rule keeps the native plan")).len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

/// The audit that runs after the native optimizer accepts what lowering
/// builds, whichever operator the window reads.
#[tokio::test]
async fn lowered_windows_satisfy_the_input_ordering_audit_over_order_breaking_inputs() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-audit");
    write_window_join_tables(&dir);

    for sql in [
        "SELECT a, id, LAG(v) OVER (PARTITION BY a ORDER BY id) AS previous \
         FROM (SELECT a, MIN(id) AS id, SUM(v) AS v FROM t GROUP BY a)",
        "SELECT a, id, LEAD(v) OVER (PARTITION BY a ORDER BY id) AS next \
         FROM (SELECT a, id, v FROM t UNION ALL SELECT a, id, v FROM t)",
        "SELECT t.a, t.id, LEAD(t.v) OVER (PARTITION BY t.a ORDER BY t.id) AS next FROM t JOIN u ON t.k = u.k",
    ] {
        let plan = window_join_sql_plan(&dir, sql).await;
        let lowering = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_eq!(lowering.selected_count(), 1, "{sql}\noutcomes={:?}", lowering.report.outcomes());
        reestablish_declared_output_ordering_if_needed(lowering.selected_plan().clone(), None)
            .unwrap_or_else(|_| panic!("the audit rejects the lowered window:\n{sql}\n{}", lowering.selected_plan()));
    }

    fs::remove_dir_all(dir).unwrap();
}

/// The native window kernel reads columns, so an aggregate or LAG / LEAD
/// argument that is not a column, such as the `1` of `COUNT(*)`, is computed
/// below the window.
#[tokio::test]
async fn window_functions_over_computed_arguments_plan_natively() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-computed-argument");
    write_window_join_tables(&dir);

    for sql in [
        "SELECT a, COUNT(*) OVER (PARTITION BY a) AS n FROM t",
        "SELECT a, SUM(v + 1) OVER (PARTITION BY a) AS s FROM t",
        "SELECT a, id, LAG(v + 1) OVER (PARTITION BY a ORDER BY id) AS previous FROM t",
    ] {
        let plan = window_join_sql_plan(&dir, sql).await;

        let (report, native_plan) = optimize_with_native_rule(plan);

        assert!(
            matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]),
            "{sql}\noutcomes={:?}",
            report.outcomes()
        );
        let native_plan = native_plan.expect("the rule keeps the native plan");
        let window = window_node(&native_plan);
        let argument = window
            .spec
            .functions
            .iter()
            .find_map(|function| match function {
                WindowFunction::Aggregate { expr, .. } | WindowFunction::LagLead { expr, .. } => Some(expr),
                _ => None,
            })
            .expect("the window carries an aggregate or LAG argument");
        assert!(matches!(argument, Expr::Column(_)), "{sql}\nargument={argument}");
    }

    fs::remove_dir_all(dir).unwrap();
}

/// An empty OVER clause makes the whole input one partition, so a window
/// aggregate lowers without partition keys, beside a ranking window and under
/// a comparison against the row count.
#[tokio::test]
async fn whole_input_aggregate_windows_plan_natively() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-whole-input");
    write_window_join_tables(&dir);

    for sql in [
        "SELECT id, COUNT(*) OVER () AS n FROM t",
        "SELECT id, SUM(v) OVER () AS s FROM t",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY a, id) AS rk, COUNT(*) OVER () AS n FROM t",
        "SELECT id FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY a, id) AS rk, COUNT(*) OVER () AS n FROM t) \
         WHERE rk <= n / 100",
    ] {
        let plan = window_join_sql_plan(&dir, sql).await;

        let (report, native_plan) = optimize_with_native_rule(plan);

        assert!(
            matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]),
            "{sql}\noutcomes={:?}",
            report.outcomes()
        );
        let native_plan = native_plan.expect("the rule keeps the native plan");
        let whole_input = native_plan.nodes().iter().any(|node| match &node.kind {
            PlanNode::Window(window) => {
                window.spec.partition_by.is_empty()
                    && window.spec.functions.iter().any(|function| {
                        matches!(function, WindowFunction::Aggregate { frame: WindowFrame::WholePartition, .. })
                    })
            },
            _ => false,
        });
        assert!(whole_input, "{sql}\n{native_plan}");
    }

    fs::remove_dir_all(dir).unwrap();
}

/// A cumulative aggregate other than SUM, MIN and MAX has no scan kernel. The
/// refusal names the window function and its frame, with or without a UNION
/// ALL around it, and never the union.
#[tokio::test]
async fn unsupported_cumulative_window_names_its_function_inside_a_union_branch() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-window-union-branch");
    write_window_join_tables(&dir);
    let cumulative = "COUNT(v) OVER (ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)";

    for sql in [
        format!("SELECT id, {cumulative} AS n FROM t"),
        format!(
            "SELECT 'a' AS k, COUNT(*) AS c FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY a DESC, id) AS rk FROM t) x \
             WHERE rk <= 10 \
             UNION ALL SELECT 'b', COUNT(*) FROM (SELECT id, {cumulative} AS n FROM t) y WHERE n > 5"
        ),
    ] {
        let plan = window_join_sql_plan(&dir, &sql).await;

        let (report, native_plan) = optimize_with_native_rule(plan);

        assert!(native_plan.is_none(), "{sql}");
        let blocking = report
            .outcomes()
            .iter()
            .find_map(|outcome| match outcome {
                CandidateOutcome::NotSupported { detail, .. } => detail.blocking(),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{sql}\noutcomes={:?}", report.outcomes()));
        assert_eq!(
            (blocking.kind(), blocking.text()),
            ("window_function", "window function count over frame ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"),
            "{sql}"
        );
    }

    fs::remove_dir_all(dir).unwrap();
}
