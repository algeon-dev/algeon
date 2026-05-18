//! GroupBy aggregation, merge/join, and concatenation.
//!
//! This example focuses on multi-table operations. For basics (loading,
//! filtering, sorting), see `getting_started.rs`.
//!
//! Run (requires GPU):
//! ```sh
//! cargo run --example groupby_and_joins
//! ```

use std::path::Path;

use cudf::{AggSpec, DataFrame, DropNaOptions, GroupByAgg, JoinType, MergeOptions, ParquetReadOptions, Runtime};

fn section(title: &str) {
    println!("\n{}", "=".repeat(72));
    println!("  {title}");
    println!("{}\n", "=".repeat(72));
}

fn main() -> cudf::Result<()> {
    let runtime = Runtime::builder().build()?;
    // Load a subset to work with.
    let parquet_path = Path::new(concat!(
        "~/code/iceberg-rust/crates/examples/output-data/",
        "3f1c2b8e-6d4a-4e0b-9a7c-5b2d8e1f4a60/VertexInfo/node.parquet"
    ));
    let opts = ParquetReadOptions::builder()
        .filepath(parquet_path)
        .columns(&["entityName", "entityType", "risk_score", "influence", "importance"])
        .num_rows(100_000)
        .with_runtime(&runtime)
        .build();
    let df = DataFrame::read_parquet_with_options(opts)?;

    // Drop nulls so aggregation examples are cleaner.
    let has_risk = df.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;
    println!("Working set: {} rows x {} cols", has_risk.num_rows(), has_risk.num_columns());

    // ── 1. Simple convenience aggregations ──────────────────────────────
    section("1. GroupBy convenience: mean, count");

    // Python: df.groupby('entityType')['risk_score'].mean()
    let mean_by_type = has_risk.groupby(["entityType"]).mean(["risk_score"])?;
    println!("groupby('entityType').mean('risk_score'):");
    println!("  columns: {:?}", mean_by_type.column_names());

    // Batch-materialize to host, then iterate — never slice-per-row.
    let types: Vec<Option<String>> = mean_by_type.series("entityType")?.to_optional_strings()?;
    let means: Vec<f64> = mean_by_type.series("risk_score_mean")?.to_vec_strict()?;
    for (tp, avg) in types.iter().zip(&means) {
        println!("  {:?} -> mean risk_score = {avg:.2}", tp.as_deref().unwrap_or("<null>"));
    }

    let count_by_type = df.groupby(["entityType"]).count(["entityName"])?;
    println!("\ngroupby('entityType').count('entityName'):");
    let types: Vec<Option<String>> = count_by_type.series("entityType")?.to_optional_strings()?;
    let counts: Vec<i32> = count_by_type.series("entityName_count")?.to_vec_strict()?;
    for (tp, cnt) in types.iter().zip(&counts) {
        println!("  {:?} -> count = {cnt}", tp.as_deref().unwrap_or("<null>"));
    }

    // ── 2. Custom multi-agg spec ────────────────────────────────────────
    section("2. Custom multi-agg with explicit aliases");

    // Python: df.groupby('entityType').agg({'risk_score': ['min','max','mean'], 'influence': 'sum'})
    let agg_result = has_risk.groupby(["entityType"]).agg(&[
        AggSpec::new("risk_score", GroupByAgg::Min).with_alias("risk_min"),
        AggSpec::new("risk_score", GroupByAgg::Max).with_alias("risk_max"),
        AggSpec::new("risk_score", GroupByAgg::Mean).with_alias("risk_mean"),
        AggSpec::new("influence", GroupByAgg::Sum).with_alias("influence_total"),
        AggSpec::new("entityName", GroupByAgg::Count { include_nulls: false }).with_alias("count"),
    ])?;

    // Batch-materialize all result columns, then iterate on host.
    let agg_types: Vec<Option<String>> = agg_result.series("entityType")?.to_optional_strings()?;
    let risk_mins: Vec<f64> = agg_result.series("risk_min")?.to_vec_strict()?;
    let risk_maxes: Vec<f64> = agg_result.series("risk_max")?.to_vec_strict()?;
    let risk_means: Vec<f64> = agg_result.series("risk_mean")?.to_vec_strict()?;

    println!("Custom multi-agg:");
    for (i, tp) in agg_types.iter().enumerate() {
        println!(
            "  {:?} -> min={:.1}, max={:.1}, mean={:.2}",
            tp.as_deref().unwrap_or("<null>"),
            risk_mins[i],
            risk_maxes[i],
            risk_means[i],
        );
    }

    // ── 3. Merge / Join ────────────────────────────────────────────────
    section("3. Merge / Join");

    let type_lookup = cudf::dataframe! {
        "entityType" => runtime.series("entityType", &["cloudasset", "user", "device", "identity"])?,
        "priority" => runtime.series::<i32>("priority", &[1, 2, 3, 4])?
    }?;

    // Inner join (Python: df.merge(lookup, on='entityType', how='inner'))
    let inner = df.merge(&type_lookup, MergeOptions::new().on(["entityType"]).how(JoinType::Inner))?;
    println!("Inner join: {} rows, columns={:?}", inner.num_rows(), inner.column_names());

    // Left join (Python: df.merge(lookup, on='entityType', how='left'))
    let left = df.merge(&type_lookup, MergeOptions::new().on(["entityType"]).how(JoinType::Left))?;
    println!("Left join:  {} rows", left.num_rows());

    // Asymmetric keys + suffixes.
    let lookup2 = cudf::dataframe! {
        "type_code" => runtime.series("type_code", &["cloudasset", "user"])?,
        "tier" => runtime.series::<i32>("tier", &[1, 2])?
    }?;
    let asym_join = df.merge(
        &lookup2,
        MergeOptions::new()
            .left_on(["entityType"])
            .right_on(["type_code"])
            .how(JoinType::Inner)
            .suffixes("_left", "_right"),
    )?;
    println!("Asymmetric-key inner join: {} rows, columns={:?}", asym_join.num_rows(), asym_join.column_names());

    // ── 4. Concatenation ────────────────────────────────────────────────
    section("4. Concatenation");

    // Python: cudf.concat([df1, df2])
    let top_half = df.head(50_000)?;
    let bottom_half = df.tail(50_000)?;
    let concatenated = top_half.concat(&bottom_half)?;
    println!("concat(head(50k), tail(50k)): {} rows", concatenated.num_rows());

    println!("\nDone.");
    Ok(())
}
