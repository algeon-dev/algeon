//! Getting started with cudf-rs — the essentials.
//!
//! This example walks through the most common DataFrame workflows a cuDF Python
//! user would reach for:
//!
//! 1. Loading data from Parquet
//! 2. Inspecting metadata (`info`)
//! 3. Accessing and materializing Series
//! 4. Constructing DataFrames from host data
//! 5. Column selection and schema reshaping
//! 6. Row slicing
//! 7. Filtering (fused predicates, boolean masks, isin)
//! 8. Series arithmetic
//! 9. Adding / replacing columns
//! 10. Sorting
//! 11. Null handling
//! 12. Deduplication
//! 13. Series unique / nunique / factorize
//! 14. Type casting
//! 15. Writing results
//! 16. Error handling
//!
//! # Key difference from cuDF Python
//!
//! In cudf-rs, **DataFrame is an immutable value type**. Every operation
//! (`filter`, `sort_values`, `with_column`, `drop_duplicates`, ...) returns a
//! **new** DataFrame instead of mutating in place. Columns are shared via
//! `Arc<Column>`, so creating a new frame is cheap when most columns are
//! unchanged — only the modified columns allocate new GPU memory.
//!
//! # `Display` vs data preview
//!
//! `println!("{df}")` prints **metadata only** (column names, dtypes, row
//! count) — it never materializes GPU data to host. To preview actual values,
//! use `Series::to_vec_strict`, `Series::to_optional_vec`, or
//! `Series::to_optional_strings` after slicing to a small number of rows.
//!
//! Run (requires GPU):
//! ```sh
//! cargo run --example getting_started
//! ```

use std::path::Path;

use cudf::{
    DataFrame, DropDuplicatesOptions, DropNaOptions, DuplicateKeepOption, InfoDetail, ParquetReadOptions, Runtime,
    Series, SortOptions, ops::expr::col,
};

/// Pretty-print a section header to stdout.
fn section(title: &str) {
    println!("\n{}", "=".repeat(72));
    println!("  {title}");
    println!("{}\n", "=".repeat(72));
}

fn main() -> cudf::Result<()> {
    let runtime = Runtime::builder().build()?;
    let parquet_path = Path::new(concat!(
        "~/code/iceberg-rust/crates/examples/output-data/",
        "3f1c2b8e-6d4a-4e0b-9a7c-5b2d8e1f4a60/VertexInfo/node.parquet"
    ));

    // ── 1. Loading data ─────────────────────────────────────────────────
    section("1. Read Parquet into DataFrame");

    let df = DataFrame::read_parquet(&runtime, parquet_path)?;
    // Display prints metadata (column names, dtypes, row count), not data.
    println!("Loaded full file (metadata): {df}");

    // With options: select subset of columns and limit rows.
    let opts = ParquetReadOptions::builder()
        .filepath(parquet_path)
        .columns(&["entityName", "entityType", "risk_score", "influence", "importance"])
        .num_rows(100_000)
        .with_runtime(&runtime)
        .build();
    let df_subset = DataFrame::read_parquet_with_options(opts)?;
    println!("Subset (5 cols, 100k rows): {df_subset}");

    // ── 2. Inspecting metadata ──────────────────────────────────────────
    section("2. DataFrame metadata & info()");

    println!("num_rows:    {}", df_subset.num_rows());
    println!("num_columns: {}", df_subset.num_columns());
    println!("columns:     {:?}", df_subset.column_names());

    let info = df_subset.info_with(InfoDetail::ALL)?;
    println!("\nDataFrameInfo:");
    println!("  row_count:    {}", info.row_count);
    println!("  column_count: {}", info.column_count);
    for dtc in &info.dtypes {
        println!("  dtype {:?}: {} columns", dtc.data_type.id(), dtc.column_count);
    }
    if let Some(cols) = &info.columns {
        for c in cols {
            println!(
                "  [{:>2}] {:<20} type={:?}  nullable={}  nulls={:?}  non_null={:?}",
                c.position,
                c.name,
                c.data_type.id(),
                c.nullable,
                c.null_count,
                c.non_null_count,
            );
        }
    }
    if let Some(mem) = &info.memory_accounting {
        println!(
            "  memory: payload={:?} bytes, device_allocated={:?} bytes",
            mem.payload_bytes, mem.device_allocated_bytes
        );
    }

    // ── 3. Series basics ────────────────────────────────────────────────
    section("3. Series: access, metadata, host materialization");

    let entity_type: Series = df_subset.series("entityType")?;
    println!(
        "entityType: len={}, null_count={}, dtype={:?}",
        entity_type.len(),
        entity_type.null_count(),
        entity_type.dtype()?
    );

    let risk: Series = df_subset.series("risk_score")?;
    println!("risk_score: len={}, null_count={}", risk.len(), risk.null_count());

    // Materialize first 10 string values to host memory.
    let small_slice = df_subset.head(10)?;
    let names_series = small_slice.series("entityName")?;
    let names_host: Vec<Option<String>> = names_series.to_optional_strings()?;
    println!("\nFirst 10 entityName values:");
    for (i, v) in names_host.iter().enumerate() {
        match v {
            Some(s) => println!("  [{i}] {s}"),
            None => println!("  [{i}] <null>"),
        }
    }

    // Materialize numeric values directly from Series.
    let influence_series = small_slice.series("influence")?;
    let influence_vals: Vec<f64> = influence_series.to_vec_strict::<f64>()?;
    println!("\nFirst 10 influence values: {influence_vals:?}");

    // ── 4. Constructing Series / DataFrames from scratch ────────────────
    section("4. Construct Series and DataFrames from host data");

    // Prefer an explicit Runtime for new high-level construction paths.
    let rt = Runtime::builder().build()?;
    let ids = rt.series("id", &[1i64, 2, 3, 4, 5])?;
    let scores = rt.series("score", &[0.9f64, 0.75, 0.6, 0.85, 0.95])?;
    let labels = rt.series("label", &["high", "medium", "low", "high", "high"])?;
    let nullable_vals =
        rt.nullable_series("rating", &[4.5f64, 0.0, 3.2, 0.0, 5.0], &[true, false, true, false, true])?;

    // dataframe! macro — convenient for small frames.
    // Series is moved into the frame (no GPU copy — only Arc ownership transfer).
    let built = cudf::dataframe! {
        "id" => ids,
        "score" => scores,
        "label" => labels,
        "rating" => nullable_vals
    }?;
    println!("Built DataFrame: {built}");

    // Or use from_named_pairs for explicit schema override.
    //
    // NOTE: The pair's key ("x", "y") becomes the DataFrame column name,
    // overriding the Series' own name.
    let built2 = DataFrame::from_named_pairs([
        ("x", rt.series("ignored_name", &[10i32, 20, 30])?),
        ("y", rt.series("also_ignored", &[1.1f32, 2.2, 3.3])?),
    ])?;
    assert_eq!(built2.column_names(), &["x", "y"]);
    println!("from_named_pairs: {built2}");

    // ── 5. Column selection & schema reshaping ──────────────────────────
    section("5. select / without_columns / rename");

    // Select specific columns (Python: df[["entityType", "risk_score"]])
    let selected = df_subset.select(["entityType", "risk_score"])?;
    println!("select(['entityType', 'risk_score']): {selected}");

    // Drop columns (Python: df.drop(columns=["influence", "importance"]))
    let dropped = df_subset.without_columns(["influence", "importance"])?;
    println!("without_columns(['influence', 'importance']): {dropped}");

    // Rename (Python: df.rename(columns={...}))
    let renamed = df_subset.rename_column("entityType", "asset_type")?;
    println!("rename 'entityType' -> 'asset_type': {renamed}");

    let multi_renamed = df_subset.rename_columns([("entityType", "type"), ("risk_score", "risk")])?;
    println!("rename multiple: {multi_renamed}");

    // ── 6. Row slicing ──────────────────────────────────────────────────
    section("6. head / tail / slice");

    let first_5 = df_subset.head(5)?;
    println!("head(5): {} rows", first_5.num_rows());

    let last_5 = df_subset.tail(5)?;
    println!("tail(5): {} rows", last_5.num_rows());

    // Slice: offset + length (not start:stop). Equivalent to df.iloc[1000:1010].
    let sliced = df_subset.slice(1000, 10)?;
    println!("slice(1000, 10): {} rows", sliced.num_rows());

    // ── 7. Filtering ────────────────────────────────────────────────────
    section("7. Filtering: fused predicates, boolean masks, isin");

    // Preferred multi-condition path: fuse predicates into one row filter.
    let high_risk = df_subset.filter_expr(col("risk_score").is_not_null() & col("risk_score").gt(50.0_f64))?;
    println!("Rows with non-null risk_score > 50.0: {}", high_risk.num_rows());

    // filter_all(...) is a convenience wrapper for common AND chains.
    let cleaned = df_subset.filter_all([
        col("entityName").is_not_null(),
        col("entityType").is_not_null(),
        col("entityName").ne(""),
        col("entityType").ne(""),
    ])?;
    println!("Rows with non-empty entityName/entityType: {}", cleaned.num_rows());

    // If you already have a computed boolean mask, filter(&Series) is still
    // the right low-level primitive.
    let not_null_mask = risk.is_not_null()?;
    let has_risk = df_subset.filter(&not_null_mask)?;
    println!("Rows with non-null risk_score (precomputed mask): {}", has_risk.num_rows());

    // Another precomputed-mask example: string equality.
    let cloud_mask = df_subset.series("entityType")?.eq_scalar("cloudasset")?;
    let cloud_assets = df_subset.filter(&cloud_mask)?;
    println!("Rows where entityType == 'cloudasset': {}", cloud_assets.num_rows());

    // isin filter (Python: df[df['entityType'].isin(['cloudasset', 'user'])])
    let wanted_types = runtime.series("wanted", &["cloudasset", "user"])?;
    let isin_result = df_subset.filter_isin("entityType", &wanted_types)?;
    println!("Rows where entityType in ['cloudasset', 'user']: {}", isin_result.num_rows());

    // ── 8. Series arithmetic — operator overloads ───────────────────────
    section("8. Series arithmetic");

    // All arithmetic operators (+, -, *, /) return Result<Series>, never panic.
    let influence = has_risk.series("influence")?;
    let importance = has_risk.series("importance")?;
    let risk = has_risk.series("risk_score")?;

    // Series + Series (Python: df['influence'] + df['importance'])
    let combined = (&influence + &importance)?;
    println!("influence + importance: len={}", combined.len());

    // Series * scalar (Python: df['risk_score'] * 100)
    let risk_pct = (&risk * 100.0_f64)?;
    println!("risk_score * 100: len={}, name={:?}", risk_pct.len(), risk_pct.name());

    // scalar / Series (Python: 1.0 / df['risk_score'])
    let inverted = (1.0_f64 / &risk)?;
    println!("1.0 / risk_score (reciprocal): len={}", inverted.len());

    // Named methods are still available for explicit style.
    let explicit_add = risk.add(&influence)?;
    println!("Named .add(): len={}", explicit_add.len());

    // ── 9. Adding / replacing columns ───────────────────────────────────
    //
    // Three methods, each for a different use case:
    //
    //   with_column(name, series) — add or replace ONE column.
    //     The `name` parameter controls the DataFrame column name.
    //     The Series' own name (from Series::rename) is IGNORED.
    //
    //   assign([(name, series), ...]) — batch add/replace MULTIPLE columns.
    //     Same name authority rule. Last-wins on duplicate names.
    //
    //   with_columns([expr, ...]) — declarative computed columns via Expr DSL.
    //     ALL expressions are evaluated against the ORIGINAL DataFrame state
    //     (unlike Python's df.assign() where later assignments can reference
    //     earlier ones).
    //     Expr reductions such as col("risk_score").agg().sum() are
    //     broadcast aggregates: they reduce once, then broadcast back to the
    //     row count. They also require an explicit output name when used as
    //     standalone columns in with_columns().
    //     Use Series::sum() / sum_value() when you want a scalar result.
    section("9. with_column / assign / with_columns");

    // with_column: single column.
    let risk_pct_series = (&has_risk.series("risk_score")? * 100.0_f64)?.rename("whatever");
    // NOTE: "risk_pct" (the parameter) becomes the column name, not "whatever".
    let df_with_pct = has_risk.with_column("risk_pct", risk_pct_series)?;
    println!("with_column('risk_pct'): columns={:?}", df_with_pct.column_names());

    // assign: batch add multiple columns.
    let df_multi = has_risk.assign([
        ("is_important", has_risk.series("importance")?.gt_scalar(0.5_f64)?),
        ("combined", (&has_risk.series("influence")? + &has_risk.series("importance")?)?),
    ])?;
    println!("assign 2 columns: columns={:?}", df_multi.column_names());

    // with_columns: expression-based.
    let df_expr = has_risk.with_columns([
        ("risk_x10", col("risk_score") * 10.0_f64),
        ("score_diff", col("influence") - col("importance")),
        ("risk_share", col("risk_score") / col("risk_score").agg().sum()),
    ])?;
    let risk_share_values: Vec<f64> = df_expr.series("risk_share")?.to_vec_strict()?;
    println!("with_columns (Expr DSL): columns={:?}", df_expr.column_names());
    println!("with_columns broadcast aggregate example: risk_share[0]={:?}", risk_share_values.first());
    // NOTE: with_columns([col("risk_score").agg().sum()]) now errors
    // because broadcast reductions must be named explicitly.

    // ── 10. Sorting ─────────────────────────────────────────────────────
    section("10. Sorting");

    let sorted = has_risk
        .sort_values(["risk_score"], SortOptions::new().with_descending(vec![true]).with_nulls_first(vec![false]))?;
    let top3_vals: Vec<f64> = sorted.head(3)?.series("risk_score")?.to_vec_strict()?;
    println!("Top 3 risk_score (sorted desc): {top3_vals:?}");

    // nlargest / nsmallest (Python: df.nlargest(5, 'risk_score'))
    let top5_vals: Vec<f64> = has_risk.nlargest(5, "risk_score")?.series("risk_score")?.to_vec_strict()?;
    println!("nlargest(5, 'risk_score'): {top5_vals:?}");

    let bottom5_vals: Vec<f64> = has_risk.nsmallest(5, "risk_score")?.series("risk_score")?.to_vec_strict()?;
    println!("nsmallest(5, 'risk_score'): {bottom5_vals:?}");

    // ── 11. Null handling ───────────────────────────────────────────────
    section("11. Null handling");

    let null_mask = df_subset.series("risk_score")?.is_null()?;
    let null_rows = df_subset.filter(&null_mask)?;
    println!("Rows where risk_score IS NULL: {}", null_rows.num_rows());

    // fillna with scalar (Python: df['risk_score'].fillna(0.0))
    let filled_series = df_subset.series("risk_score")?.fillna(0.0_f64)?;
    println!("After fillna(0.0): null_count={}", filled_series.null_count());

    // fillna at DataFrame level
    let df_filled = df_subset.fillna("risk_score", -1.0_f64)?;
    println!("DataFrame fillna('risk_score', -1.0): null_count={}", df_filled.series("risk_score")?.null_count());

    // dropna (Python: df.dropna(subset=['risk_score', 'influence']))
    let df_dropna = df_subset.dropna(DropNaOptions::builder().subset(&["risk_score", "influence"]).build())?;
    println!("dropna(how='any'): {} rows", df_dropna.num_rows());

    let df_drop2 = df_subset.dropna(DropNaOptions::builder().subset(&["risk_score"]).build())?;
    println!("dropna(subset=['risk_score']): {} rows", df_drop2.num_rows());

    // ── 12. Deduplication ───────────────────────────────────────────────
    section("12. Deduplication");

    let deduped = df_subset.drop_duplicates(
        DropDuplicatesOptions::builder().subset(&["entityType"]).keep(DuplicateKeepOption::KeepFirst).build(),
    )?;
    println!("Unique entityType (keep first): {} rows", deduped.num_rows());

    // KeepAny is faster (no ordering guarantee).
    let deduped_any = df_subset.drop_duplicates(
        DropDuplicatesOptions::builder().subset(&["entityType"]).keep(DuplicateKeepOption::KeepAny).build(),
    )?;
    println!("Unique entityType (keep any): {} rows", deduped_any.num_rows());

    // ── 13. Series unique / nunique / factorize ─────────────────────────
    section("13. Series: unique, nunique, factorize");

    let entity_type_col = df_subset.series("entityType")?;

    let unique_types = entity_type_col.unique()?;
    let unique_strs = unique_types.to_optional_strings()?;
    println!("entityType unique values: {unique_strs:?}");

    let n_unique = entity_type_col.nunique()?;
    println!("entityType nunique: {n_unique}");

    let factorized = entity_type_col.factorize()?;
    println!(
        "factorize -> row_codes.len()={}, codebook.len()={}",
        factorized.row_codes().len(),
        factorized.codebook().len()
    );
    let first_10_codes: Vec<i32> = factorized.row_codes().to_vec_strict()?;
    println!("  first 10 codes: {:?}", &first_10_codes[..10.min(first_10_codes.len())]);

    // The returned codebook is reusable for later encode()/decode() calls.
    let codebook = factorized.codebook();
    let query = unique_types.slice(0, 3.min(unique_types.len()))?;
    let encoded_query = codebook.encode(&query)?;
    let decoded_query = codebook.decode(&encoded_query)?;
    println!("safe codebook query: {:?}", query.to_optional_strings()?);
    println!("safe codebook decoded: {:?}", decoded_query.to_optional_strings()?);

    // ── 14. Type casting ────────────────────────────────────────────────
    section("14. Type casting");

    // Series-level cast (Python: s.astype('int64'))
    let risk_as_i64 = has_risk.series("risk_score")?.astype(cudf::DataType::new(cudf::TypeId::Int64))?;
    println!("risk_score as Int64: dtype={:?}", risk_as_i64.dtype()?);

    // DataFrame-level cast_column.
    let df_casted = has_risk.cast_column("risk_score", cudf::DataType::new(cudf::TypeId::Float32))?;
    println!("cast_column -> Float32: dtype={:?}", df_casted.series("risk_score")?.dtype()?);

    // ── 15. Writing results ─────────────────────────────────────────────
    section("15. Write results to Parquet");

    let output_path = "/tmp/cudf_example_output.parquet";
    high_risk.write_parquet(output_path)?;
    println!("Written {} rows to {output_path}", high_risk.num_rows());

    let readback = DataFrame::read_parquet(&runtime, output_path)?;
    println!("Read back (metadata): {readback}");

    // ── 16. Error handling ──────────────────────────────────────────────
    section("16. Error handling");

    // Accessing a non-existent column.
    match df_subset.series("nonexistent_column") {
        Ok(_) => println!("Unexpectedly succeeded"),
        Err(e) => {
            println!("Expected error: {e}");
            println!("  kind:      {:?}", e.kind());
            println!("  operation: {:?}", e.operation());
            println!("  status:    {:?}", e.status());
        },
    }

    // Type mismatch when materializing.
    match df_subset.series("risk_score")?.to_vec_strict::<i32>() {
        Ok(_) => println!("Unexpectedly succeeded"),
        Err(e) => {
            println!("Type mismatch error: {e}");
            println!("  kind:    {:?}", e.kind());
            println!("  context: {:?}", e.context());
        },
    }

    // ── Summary ─────────────────────────────────────────────────────────
    section("Summary");
    println!("All operations completed successfully.");
    println!("cudf-rs gives cuDF Python users a familiar DataFrame workflow");
    println!("with Rust ownership safety, zero-copy column sharing, and");
    println!("Result-based error handling -- all on the GPU.");

    Ok(())
}
