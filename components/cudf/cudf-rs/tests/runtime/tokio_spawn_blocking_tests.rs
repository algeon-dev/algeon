//! Tokio-focused concurrency tests for safe cudf-rs APIs.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use cudf::{
    TypeId,
    core::{Column, Table},
};

fn assert_send_sync<T: Send + Sync>() {}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn test_cudf_api_inside_spawn_blocking_is_thread_safe_and_runtime_responsive() {
    // Compile-time guarantees for key shared types.
    assert_send_sync::<Column>();
    assert_send_sync::<Table>();

    let data: Vec<i32> = (0..4096).collect();
    let shared = Arc::new(Column::from_slice(&data).unwrap());

    // Heartbeat task: if Tokio's async worker gets blocked, tick count will stall.
    let ticker = tokio::spawn(async {
        let start = Instant::now();
        let mut ticks = 0usize;

        while start.elapsed() < Duration::from_millis(300) {
            tokio::time::sleep(Duration::from_millis(5)).await;
            ticks += 1;
        }

        ticks
    });

    // Stress the API concurrently via spawn_blocking workers.
    let workers = 8;
    let iterations = 32;
    let mut handles = Vec::with_capacity(workers);

    for _ in 0..workers {
        let shared = Arc::clone(&shared);
        handles.push(tokio::task::spawn_blocking(move || -> cudf::Result<()> {
            for _ in 0..iterations {
                // Shared read access from multiple threads.
                let view = shared.view()?;
                assert_eq!(view.len()?, 4096);
                assert_eq!(view.data_type()?.id(), TypeId::Int32);

                // Per-thread deep copy and host transfer.
                let copied = shared.copy()?;
                let host = copied.view()?.to_vec::<i32>()?;
                assert_eq!(host.first().copied(), Some(0));
                assert_eq!(host.last().copied(), Some(4095));
            }
            Ok(())
        }));
    }

    for handle in handles {
        let result = tokio::time::timeout(Duration::from_secs(20), handle)
            .await
            .expect("spawn_blocking task timed out")
            .expect("spawn_blocking task panicked");

        result.expect("cudf operation failed inside spawn_blocking");
    }

    let ticks = ticker.await.expect("ticker task panicked");
    assert!(ticks >= 10, "Tokio runtime heartbeat stalled while spawn_blocking tasks were running (ticks={ticks})");
}
