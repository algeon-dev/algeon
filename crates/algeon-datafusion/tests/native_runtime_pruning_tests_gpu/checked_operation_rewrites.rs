//! Native optimizer rewrites that move an operation that can fail onto rows
//! the query does not evaluate it on. Each table holds a failing value only on
//! rows DataFusion never evaluates the operation on, so the native result must
//! match DataFusion's instead of failing.

use std::{fs::File, path::Path};

use arrow::array::{Float64Array, Int64Array};
use datafusion::parquet::arrow::ArrowWriter;
use tokio::runtime::Builder as TestRuntimeBuilder;

use super::*;

const TABLES: &[&str] = &["t", "t_ok", "u", "u_all", "u_big", "fact", "dim_big"];

fn int64(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(values))
}

fn write_table(dir: &Path, name: &str, columns: Vec<(&str, ArrayRef)>) {
    let schema = Arc::new(Schema::new(
        columns.iter().map(|(column, array)| Field::new(*column, array.data_type().clone(), false)).collect::<Vec<_>>(),
    ));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), columns.into_iter().map(|(_, array)| array).collect()).unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(dir.join(format!("{name}.parquet"))).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// The tables of the 2026-11-14 review audit reproductions. Row `k = 2` of `t`
/// holds an `f` no INT or BIGINT represents, and `u_big`, larger than `t` so
/// that `t` builds and no key filter reaches it first, has no `k = 2`. Row
/// `dk = 99` of `fact` has `qty = 0` and no `dim_big` row, and `dim_big` is
/// larger than `fact` for the same reason.
fn write_tables(dir: &Path) {
    write_table(
        dir,
        "t",
        vec![("k", int64([1, 2, 3, 4])), ("f", Arc::new(Float64Array::from(vec![1.5, 1e30, 2.5, 3.5])))],
    );
    write_table(
        dir,
        "t_ok",
        vec![("k", int64([1, 2, 3, 4])), ("f", Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5, 4.5])))],
    );
    write_table(dir, "u", vec![("k", int64([1, 3, 4])), ("b", int64([1, 1, 100]))]);
    write_table(dir, "u_all", vec![("k", int64([1, 2, 3, 4])), ("b", int64([100; 4]))]);
    write_table(
        dir,
        "u_big",
        vec![("k", int64([1, 3, 4].into_iter().chain(1000..3000))), ("b", int64(std::iter::repeat_n(100, 2003)))],
    );
    write_table(
        dir,
        "fact",
        vec![
            ("dk", int64([1, 1, 2, 2, 99])),
            ("grp", int64([1, 2, 1, 2, 1])),
            ("amount", int64([10, 20, 30, 40, 50])),
            ("qty", int64([2, 4, 5, 8, 0])),
        ],
    );
    let dim_keys = (1..=5000).filter(|key| *key != 99).collect::<Vec<i64>>();
    write_table(
        dir,
        "dim_big",
        vec![("dk", int64(dim_keys.iter().copied())), ("region", int64(dim_keys.iter().map(|key| key % 7)))],
    );
}

async fn tables_sql_plan(dir: &Path, sql: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(SessionConfig::new());
    for table in TABLES {
        let path = dir.join(format!("{table}.parquet"));
        ctx.register_parquet(*table, path.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
    ctx.sql(sql).await.unwrap().create_physical_plan().await.unwrap()
}

/// Runs `sql` on DataFusion and, without a CPU fallback, natively, and
/// compares the results.
async fn assert_native_matches_cpu(name: &str, sql: &str) {
    let dir = unique_temp_dir(&format!("algeon-datafusion-native-runtime-checked-rewrites-{name}"));
    write_tables(&dir);
    let cpu = collect(tables_sql_plan(&dir, sql).await, Arc::new(TaskContext::default())).await.unwrap();
    assert!(cpu.iter().map(RecordBatch::num_rows).sum::<usize>() > 0, "{name} must exercise output rows");
    let optimization = optimize_with_pruning_template_requiring_native(tables_sql_plan(&dir, sql).await);
    assert_algeon_native_exec(&optimization.plan, name);
    let native = collect(optimization.plan, Arc::new(TaskContext::default()))
        .await
        .unwrap_or_else(|error| panic!("{name}: native execution failed where DataFusion succeeds: {error:?}"));
    assert_batches_equal(&cpu, &native);
    fs::remove_dir_all(dir).unwrap();
}

/// E1a: the residual cast runs on matched pairs only, so the failing row,
/// which matches no key, never fails it.
async fn residual_cast_skips_a_row_that_matches_no_key() {
    assert_native_matches_cpu(
        "residual-cast-unmatched-row",
        "SELECT t.k FROM t JOIN u_big u ON t.k = u.k AND CAST(t.f AS INT) < u.b ORDER BY t.k",
    )
    .await;
}

/// E1b: the failing row matches a key, and `WHERE` removes it before the join.
async fn residual_cast_skips_a_row_the_where_clause_removes() {
    assert_native_matches_cpu(
        "residual-cast-filtered-row",
        "SELECT t.k FROM t JOIN u_all u ON t.k = u.k AND CAST(t.f AS INT) < u.b WHERE t.f < 1000 ORDER BY t.k",
    )
    .await;
}

/// E2: the measure divides by zero only on the fact row that matches no
/// dimension key.
async fn aggregate_measure_skips_a_fact_row_that_matches_no_key() {
    assert_native_matches_cpu(
        "measure-inner-join",
        "SELECT d.region, SUM(f.amount / f.qty) AS s FROM fact f JOIN dim_big d ON f.dk = d.dk \
         GROUP BY d.region ORDER BY 1",
    )
    .await;
}

/// E2 through the semi join an `IN` subquery plans.
async fn aggregate_measure_skips_a_fact_row_the_semi_join_removes() {
    assert_native_matches_cpu(
        "measure-semi-join",
        "SELECT f.grp, SUM(f.amount / f.qty) AS s FROM fact f WHERE f.dk IN (SELECT dk FROM dim_big) \
         GROUP BY f.grp ORDER BY 1",
    )
    .await;
}

/// C1: the cuDF AST casts Float64 to BIGINT without the range check, so the
/// residual runs as the checked filter over matched pairs like `CAST(f AS
/// INT)` does.
async fn residual_cast_to_bigint_runs_natively() {
    assert_native_matches_cpu(
        "residual-cast-bigint",
        "SELECT t.k FROM t_ok t JOIN u ON t.k = u.k AND CAST(t.f AS BIGINT) < u.b ORDER BY t.k",
    )
    .await;
}

fn block_on(case: impl std::future::Future<Output = ()>) {
    TestRuntimeBuilder::new_current_thread().enable_all().build().expect("case runtime").block_on(case)
}

#[test]
fn checked_operation_rewrite_cases() {
    testing::run_named_cases(&[
        ("residual_cast_skips_a_row_that_matches_no_key", || block_on(residual_cast_skips_a_row_that_matches_no_key())),
        ("residual_cast_skips_a_row_the_where_clause_removes", || {
            block_on(residual_cast_skips_a_row_the_where_clause_removes())
        }),
        ("aggregate_measure_skips_a_fact_row_that_matches_no_key", || {
            block_on(aggregate_measure_skips_a_fact_row_that_matches_no_key())
        }),
        ("aggregate_measure_skips_a_fact_row_the_semi_join_removes", || {
            block_on(aggregate_measure_skips_a_fact_row_the_semi_join_removes())
        }),
        ("residual_cast_to_bigint_runs_natively", || block_on(residual_cast_to_bigint_runs_natively())),
    ]);
}
