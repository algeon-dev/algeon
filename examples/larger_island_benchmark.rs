use datafusion_cudf::{LargerIslandBenchmarkConfig, run_larger_island_benchmark};

fn main() -> datafusion::common::Result<()> {
    let config = LargerIslandBenchmarkConfig::default()
        .with_input_rows(env_usize("CUDF_BENCH_ROWS", 1_000_000)?)
        .with_batch_size(env_usize("CUDF_BENCH_BATCH_SIZE", 65_536)?)
        .with_group_count(env_usize("CUDF_BENCH_GROUPS", 4_096)?)
        .with_limit(env_usize("CUDF_BENCH_LIMIT", 100)?)
        .with_warmup_iterations(env_usize("CUDF_BENCH_WARMUPS", 1)?)
        .with_iterations(env_usize("CUDF_BENCH_ITERATIONS", 3)?);

    let report = run_larger_island_benchmark(config)?;
    print!("{}", report.to_text());
    Ok(())
}

fn env_usize(name: &'static str, default: usize) -> datafusion::common::Result<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| datafusion::common::DataFusionError::Plan(format!("invalid {name}: {error}")))
}
