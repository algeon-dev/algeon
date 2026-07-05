//! Minimal repro probe: native aggregate semantics over EMPTY input.
//!
//! TPC-DS tiny-dataset parity showed 9 queries with `cpu=1 native=0` row
//! mismatches (all ROLLUP/global-aggregate shapes) plus 2 `compiled final
//! merge produced no non-empty partitions` failures. This probe checks the
//! minimal shapes on a single local parquet source with a never-true filter,
//! with no joins involved (so the join-leaf empty-chunk fix is out of scope).
//!
//! Run: `flock /tmp/cudf-gpu.lock cargo run -p datafusion-nexus-bench --bin empty_input_agg_probe`

use std::{error::Error, fs, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use arrow::{
    array::{Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
    util::pretty::pretty_format_batches,
};
use datafusion::{
    execution::SessionStateBuilder,
    parquet::{arrow::ArrowWriter, file::properties::WriterProperties},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, CudfSessionStateBuilderExt, GpuFallbackPolicy,
    NativeCudfExecutionConfig, NativeDataFusionPlanningReport,
};

const PROBES: &[(&str, &str)] = &[
    ("global_agg_over_empty", "SELECT SUM(payload) AS s, COUNT(*) AS c FROM bench WHERE key > 100000"),
    ("global_avg_over_empty", "SELECT AVG(payload) AS a FROM bench WHERE key > 100000"),
    ("group_by_over_empty", "SELECT key, SUM(payload) AS s FROM bench WHERE key > 100000 GROUP BY key"),
    (
        "rollup_over_empty",
        "SELECT key, SUM(payload) AS s FROM bench WHERE key > 100000 GROUP BY ROLLUP(key) ORDER BY key",
    ),
    (
        "rollup_over_nonempty",
        "SELECT key, SUM(payload) AS s FROM bench WHERE key >= 0 GROUP BY ROLLUP(key) ORDER BY key",
    ),
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = unique_temp_dir("empty-input-agg-probe");
    let path = dir.join("bench.parquet");
    write_parquet(&path);

    for (id, sql) in PROBES {
        // Native session.
        let optimizer_config = CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
            NativeCudfExecutionConfig::bounded(256 * 1024 * 1024, 64 * 1024 * 1024, 1, 3, 1),
        );
        let rule = Arc::new(CudfNativePhysicalOptimizerRule::new(optimizer_config));
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(probe_config())
            .with_cudf_native_rule(Arc::clone(&rule))
            .build();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await?;
        let df = ctx.sql(sql).await?;
        let plan = df.clone().create_physical_plan().await?;
        let report = NativeDataFusionPlanningReport::from_retained_rule(*id, plan, rule.as_ref());
        let row = report.rows.first().unwrap();
        let native = match df.collect().await {
            Ok(batches) => pretty_format_batches(&batches)?.to_string(),
            Err(e) => format!("NATIVE ERROR: {e}"),
        };

        // Plain CPU session.
        let cpu_ctx = SessionContext::new_with_state(
            SessionStateBuilder::new_with_default_features().with_config(probe_config()).build(),
        );
        cpu_ctx.register_parquet("bench", path.to_str().unwrap(), ParquetReadOptions::default()).await?;
        let cpu = pretty_format_batches(&cpu_ctx.sql(sql).await?.collect().await?)?.to_string();

        let verdict = if native == cpu { "MATCH" } else { "MISMATCH" };
        println!(
            "== {id}: {verdict} (path={:?}, outcome={:?}, reason={:?})",
            row.post_native_execution_path, row.outcome, row.reason_code
        );
        if native != cpu {
            println!("-- cpu:\n{cpu}\n-- native:\n{native}");
        }
    }

    fs::remove_dir_all(&dir).ok();
    Ok(())
}

fn probe_config() -> SessionConfig {
    let mut config = SessionConfig::new()
        .with_repartition_file_scans(false)
        .with_repartition_aggregations(false)
        .with_repartition_sorts(false)
        .with_target_partitions(1);
    config.options_mut().optimizer.enable_topk_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_aggregate_dynamic_filter_pushdown = false;
    config.options_mut().optimizer.enable_dynamic_filter_pushdown = false;
    config
}

fn write_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from((0..32).map(|i| i % 4).collect::<Vec<_>>())),
            Arc::new(Int64Array::from((0..32).map(|i| i as i64).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    let file = fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("{prefix}-{id}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}
