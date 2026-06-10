//! Dual-pipeline concurrency benchmark (P0-6 quantification).
//!
//! Measures wall-clock overlap of N independent pipelines on one device.
//! Each pipeline is one `Runtime` (its own CUDA stream) plus a stream
//! reservation, per the device-domain concurrency model. The workload per
//! round is host-to-device upload, sort, binary add, and reduction — a mix of
//! enqueue-heavy and temporary-allocating operations.
//!
//! Reported metric: `overlap ratio = T_concurrent / T_single_pipeline`.
//! A ratio near 1.0 means pipelines overlap fully (host-side enqueue no
//! longer serializes); a ratio near N means pipelines serialize.
//!
//! Usage: `cargo run --release --example dual_pipeline_bench [pipelines] [rounds] [rows]`

use std::time::Instant;

use cudf::{DeviceId, DeviceMemoryManager, Runtime, SortOptions};

const DEFAULT_PIPELINES: usize = 2;
const DEFAULT_ROUNDS: usize = 30;
const DEFAULT_ROWS: usize = 1 << 20;
const RESERVATION_BYTES: usize = 1 << 30;

fn pipeline_round(rt: &Runtime, data: &[i32]) {
    let series = rt.series(data).expect("series upload");
    let sorted = series.sort_values(SortOptions::default()).expect("sort");
    let shifted = sorted.add_scalar(1i32).expect("add");
    // Host materialization: the per-op cost includes draining this
    // pipeline's stream, which is where host-side serialization (if any)
    // becomes visible across pipelines.
    let total: i64 = shifted.sum_value().expect("reduce");
    assert!(total > 0);
    // Device-to-host materialization: under the retired global lock this
    // copy-and-synchronize ran while holding the process-wide mutex.
    let host = shifted.to_vec_strict::<i32>().expect("to host");
    assert_eq!(host.len(), data.len());
}

fn run_pipeline(rounds: usize, rows: usize, pipeline_id: u64) {
    let rt = Runtime::builder().with_device(DeviceId::new(0).expect("device 0")).build().expect("runtime");
    let _lease = rt.stream_reservation(pipeline_id, RESERVATION_BYTES).expect("reservation");
    let data: Vec<i32> = (0..rows as i32).rev().collect();
    for _ in 0..rounds {
        pipeline_round(&rt, &data);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let pipelines: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_PIPELINES);
    let rounds: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_ROUNDS);
    let rows: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_ROWS);

    let device = DeviceId::new(0).expect("device 0");
    DeviceMemoryManager::install_for_device(device).expect("install memory domain");

    // Warm up: pool growth, module loading, stream pool creation.
    run_pipeline(3, rows, 999);

    // Baseline: one pipeline, `rounds` rounds.
    let start = Instant::now();
    run_pipeline(rounds, rows, 1000);
    let single = start.elapsed();

    // Concurrent: `pipelines` pipelines, `rounds` rounds each.
    let start = Instant::now();
    let workers: Vec<_> =
        (0..pipelines).map(|p| std::thread::spawn(move || run_pipeline(rounds, rows, 2000 + p as u64))).collect();
    for worker in workers {
        worker.join().expect("pipeline worker");
    }
    let concurrent = start.elapsed();

    let ratio = concurrent.as_secs_f64() / single.as_secs_f64();
    println!("pipelines={pipelines} rounds={rounds} rows={rows}");
    println!("single-pipeline wall-clock:    {single:?}");
    println!("{pipelines}-pipeline wall-clock:        {concurrent:?}");
    println!("overlap ratio (lower is better, 1.0 = full overlap, {pipelines}.0 = serialized): {ratio:.2}");
}
