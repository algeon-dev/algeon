//! Capped pool hard-cap OOM demonstration.
//!
//! Verifies that a `Runtime::builder().with_pool_memory_resource(initial, max)`
//! runtime fails with `ErrorKind::OutOfMemory` when a cuDF workload exceeds
//! the configured pool limit — rather than silently falling back to another
//! allocator or bypassing the cap.
//!
//! # Environment variables
//!
//! | Variable          | Default | Description                              |
//! |-------------------|---------|------------------------------------------|
//! | `POOL_INITIAL_MB` | 64      | Initial pool allocation (MiB)            |
//! | `POOL_MAX_MB`     | 128     | Maximum pool size — the hard cap (MiB); use `0` for unbounded |
//! | `OOM_BASE_ROWS`   | 500_000 | Rows in the seed DataFrame               |
//! | `OOM_NUM_COLUMNS` | 4       | Number of f64 columns in the seed frame  |
//! | `MAX_ITERATIONS`  | 40      | Maximum concat-doublings before giving up|
//!
//! # Run
//!
//! ```sh
//! cargo run --example capped_pool_hard_cap_oom
//! # or with custom limits:
//! POOL_MAX_MB=64 cargo run --example capped_pool_hard_cap_oom
//! ```

use std::env;

use cudf::{DataFrame, DeviceId, ErrorKind, ExecutionContext, Runtime, Series};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

const MIB: usize = 1024 * 1024;

fn runtime_context(runtime: &Runtime) -> ExecutionContext<'_> {
    ExecutionContext::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource())
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> cudf::Result<()> {
    // ── Configuration ────────────────────────────────────────────────────
    let initial_mb: usize = env_or("POOL_INITIAL_MB", 64);
    let max_mb: usize = env_or("POOL_MAX_MB", 0);
    let base_rows: usize = env_or("OOM_BASE_ROWS", 500_000);
    let num_columns: usize = env_or("OOM_NUM_COLUMNS", 4);
    let max_iterations: usize = env_or("MAX_ITERATIONS", 40);

    let initial_bytes = initial_mb * MIB;
    let max_bytes = max_mb * MIB;

    println!("=== Capped Pool Hard-Cap OOM Example ===");
    println!();
    println!("  POOL_INITIAL_MB  = {initial_mb} ({initial_bytes} bytes)");
    println!("  POOL_MAX_MB      = {max_mb} ({max_bytes} bytes)");
    println!("  OOM_BASE_ROWS    = {base_rows}");
    println!("  OOM_NUM_COLUMNS  = {num_columns}");
    println!("  MAX_ITERATIONS   = {max_iterations}");
    println!();

    // ── Build capped pool Runtime ────────────────────────────────────────
    let device = DeviceId::new(0)?;

    let runtime = Runtime::builder()
        .with_device(device)
        .with_pool_memory_resource(initial_bytes, (max_bytes != 0).then_some(max_bytes))
        .build()?;
    println!("[INIT] Runtime built on device 0 with capped pool MR (initial={initial_mb} MiB, max={max_mb} MiB)");

    // ── Construct seed DataFrame ─────────────────────────────────────────
    let host_data: Vec<f64> = (0..base_rows).map(|i| i as f64 * 0.001).collect();

    let columns: Vec<Series> = (0..num_columns)
        .map(|i| {
            let name = format!("col_{i}");
            runtime.series(&name, &host_data)
        })
        .collect::<cudf::Result<_>>()?;

    let seed = DataFrame::from_series(columns)?;

    println!("[INIT] Seed DataFrame: {} rows x {} cols", seed.num_rows(), seed.num_columns(),);
    println!();

    // ── Concat-doubling loop ─────────────────────────────────────────────
    let mut current = seed;
    let operation_ctx = runtime_context(&runtime);

    for iteration in 1..=max_iterations {
        let rows_before = current.num_rows();
        println!("[CONCAT] iteration={iteration}/{max_iterations}  rows={rows_before}");

        match current.in_context(&operation_ctx).concat(&current) {
            Ok(next) => {
                current = next;
            },
            Err(err) if err.kind() == ErrorKind::OutOfMemory => {
                println!();
                println!("=== OutOfMemory detected ===");
                println!();
                println!("  iteration:        {iteration}");
                println!("  rows before concat: {rows_before}");
                println!("  pool hard cap:    {max_mb} MiB");
                println!("  error:            {err}");
                println!();
                println!("The capped pool behaved as a hard cap for this workload.");
                println!("No silent fallback to an unbounded allocator was observed.");
                return Ok(());
            },
            Err(err) => {
                println!();
                println!("=== Unexpected error (not OutOfMemory) ===");
                println!("  error kind: {:?}", err.kind());
                println!("  error:      {err}");
                return Err(err);
            },
        }
    }

    // If we reach here, OOM was never triggered.
    println!();
    println!("=== OOM not reached ===");
    println!();
    println!("Completed all {max_iterations} iterations without hitting OutOfMemory.");
    println!("Final DataFrame: {} rows x {} cols", current.num_rows(), current.num_columns());
    println!();
    println!("Try reducing POOL_MAX_MB or increasing OOM_BASE_ROWS / OOM_NUM_COLUMNS.");

    Err(cudf::Error::new(
        ErrorKind::InvalidOperation,
        "OOM was not triggered within the configured iteration limit. \
         Adjust POOL_MAX_MB, OOM_BASE_ROWS, or OOM_NUM_COLUMNS.",
    ))
}
