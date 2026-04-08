//! Runtime integration contract tests.
//!
//! These tests validate the current runtime/threading contract described by
//! `docs/PROPOSAL.md` and the retained-domain public API:
//!
//! 1. Send+Sync static assertions for owned types.
//! 2. Negative assertions for view types (!Send).
//! 3. DataFrame spawn_blocking round-trip.
//! 4. Series spawn_blocking round-trip.
//! 5. I/O from spawn_blocking.
//! 6. Host materialization from spawn_blocking.
//! 7. Multiple chained operations in spawn_blocking.
//! 8. Filter operations in spawn_blocking.
//! 9. Concurrent compute on shared data (Sync soundness).
//!
//! Categories 1–2 are compile-time checks (no GPU required).
//! Categories 3–9 require a GPU to run.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use cudf::{
    Agg, AggSpec, DataFrame, Series, SortOptions,
    core::{Column, Scalar, Table, memory::DeviceMemoryResource},
    runtime::{CudaEvent, CudaStream},
};
use static_assertions::assert_not_impl_any;

use crate::common::sample_df;

// ===========================================================================
// Category 1: Send+Sync static assertions for owned types
// ===========================================================================

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn dataframe_is_send_sync() {
    assert_send_sync::<DataFrame>();
}

#[test]
fn series_is_send_sync() {
    assert_send_sync::<Series>();
}

#[test]
fn column_is_send_sync() {
    assert_send_sync::<Column>();
}

#[test]
fn table_is_send_sync() {
    assert_send_sync::<Table>();
}

#[test]
fn scalar_is_send_sync() {
    assert_send_sync::<Scalar>();
}

#[test]
fn cuda_stream_is_send_sync() {
    assert_send_sync::<CudaStream>();
}

#[test]
fn cuda_event_is_send_sync() {
    assert_send_sync::<CudaEvent>();
}

#[test]
fn device_memory_resource_is_send_sync() {
    assert_send_sync::<DeviceMemoryResource>();
}

#[test]
fn cuda_event_can_record_wait_and_synchronize() {
    let stream = CudaStream::new().unwrap();
    let event = CudaEvent::new().unwrap();

    event.record(&stream).unwrap();
    stream.wait_event(&event).unwrap();
    event.synchronize().unwrap();
    assert!(event.is_ready().unwrap());
}

// ===========================================================================
// Category 2: Negative assertions for view types
//
// ColumnView and TableView contain raw NonNull<c_void> pointers
// without `unsafe impl Send/Sync`. The compiler auto-derives !Send !Sync.
//
// We verify this with `static_assertions`-style negative tests. These use
// trait-bound checks that fail at compile time if the type unexpectedly
// gains Send or Sync in the future.
//
// Approach: use `std::marker::PhantomData` + a const assertion trick.
// If the type IS Send, the negative test would fail to compile — which is
// exactly what we want to guard against.
// ===========================================================================

assert_not_impl_any!(cudf::core::ColumnView<'static>: Send, Sync);
assert_not_impl_any!(cudf::core::TableView<'static>: Send, Sync);

// NOTE: The authoritative negative test is that the following code DOES NOT
// compile (verified manually and kept here as the authoritative example):
//
//   let col = Column::from_slice(&[1i32]).unwrap();
//   let view = col.view().unwrap();
//   tokio::task::spawn_blocking(move || {
//       view.len()  // ERROR: ColumnView is !Send
//   });
//
// Similarly for TableView.

// ===========================================================================
// Category 3: DataFrame spawn_blocking round-trip
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_sort_survives_spawn_blocking() {
    let df = sample_df();
    let result = tokio::task::spawn_blocking(move || df.sort_values(["id"], SortOptions::new()))
        .await
        .expect("blocking task panicked")
        .expect("sort_values failed");

    assert_eq!(result.num_rows(), 5);
    let ids: Vec<i32> = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(ids, vec![1, 1, 3, 4, 5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_head_survives_spawn_blocking() {
    let df = sample_df();
    let result =
        tokio::task::spawn_blocking(move || df.head(3)).await.expect("blocking task panicked").expect("head failed");

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.num_columns(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arc_dataframe_shared_across_workers() {
    let df = Arc::new(sample_df());

    let mut handles = Vec::new();
    for _ in 0..4 {
        let df = Arc::clone(&df);
        handles.push(tokio::task::spawn_blocking(move || {
            // Read-only: each worker creates its own views internally.
            let series = df.series("id").unwrap();
            series.to_vec_strict::<i32>()
        }));
    }

    for handle in handles {
        let ids = handle.await.expect("blocking task panicked").expect("to_vec failed");
        assert_eq!(ids, vec![3, 1, 4, 1, 5]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_drop_inside_spawn_blocking_is_safe() {
    // Move DataFrame into spawn_blocking, do work, let it drop inside
    // the closure (not returned). Verify no leak or crash.
    let df = sample_df();
    tokio::task::spawn_blocking(move || {
        let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
        assert_eq!(sorted.num_rows(), 5);
        // df and sorted both drop here, freeing GPU memory.
    })
    .await
    .expect("blocking task panicked");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_returned_from_spawn_blocking_drops_on_async_thread() {
    // Move DataFrame into spawn_blocking, return it, drop on the async task.
    let df = sample_df();
    let result = tokio::task::spawn_blocking(move || df.sort_values(["id"], SortOptions::new()))
        .await
        .expect("blocking task panicked")
        .expect("sort_values failed");

    // result (DataFrame) is now owned by the async task.
    assert_eq!(result.num_rows(), 5);
    // Dropping result here (on the async thread) must not leak or crash.
    drop(result);
}

// ===========================================================================
// Category 4: Series spawn_blocking round-trip
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_sort_survives_spawn_blocking() {
    let series = Series::new::<i64>("vals", &[50, 10, 30, 20, 40]).unwrap();
    let sorted = tokio::task::spawn_blocking(move || series.sort_values(SortOptions::new()))
        .await
        .expect("blocking task panicked")
        .expect("sort_values failed");

    let vals: Vec<i64> = sorted.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![10, 20, 30, 40, 50]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_arithmetic_in_spawn_blocking() {
    let a = Series::new::<f64>("a", &[1.0, 2.0, 3.0]).unwrap();
    let b = Series::new::<f64>("b", &[10.0, 20.0, 30.0]).unwrap();

    let result =
        tokio::task::spawn_blocking(move || a.add(&b)).await.expect("blocking task panicked").expect("add failed");

    let vals: Vec<f64> = result.to_vec_strict::<f64>().unwrap();
    assert_eq!(vals, vec![11.0, 22.0, 33.0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_unique_in_spawn_blocking() {
    let series = Series::new::<i32>("x", &[1, 2, 2, 3, 3, 3]).unwrap();
    let unique = tokio::task::spawn_blocking(move || series.unique())
        .await
        .expect("blocking task panicked")
        .expect("unique failed");

    assert_eq!(unique.nunique().unwrap(), 3);
}

// ===========================================================================
// Category 5: I/O from spawn_blocking
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parquet_round_trip_in_spawn_blocking() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("runtime_test.parquet");
    let path_clone = path.clone();

    // Write from spawn_blocking.
    let df = sample_df();
    tokio::task::spawn_blocking(move || df.write_parquet(&path))
        .await
        .expect("blocking task panicked")
        .expect("write_parquet failed");

    // Read from a separate spawn_blocking.
    let loaded = tokio::task::spawn_blocking(move || DataFrame::read_parquet(&path_clone))
        .await
        .expect("blocking task panicked")
        .expect("read_parquet failed");

    assert_eq!(loaded.num_columns(), 2);
    assert_eq!(loaded.num_rows(), 5);
    assert_eq!(loaded.column_names(), vec!["id", "score"]);
}

// ===========================================================================
// Category 6: Host materialization from spawn_blocking
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_materialization_to_vec_in_spawn_blocking() {
    let df = sample_df();
    let ids = tokio::task::spawn_blocking(move || {
        let col = df.series("id").unwrap();
        col.to_vec_strict::<i32>()
    })
    .await
    .expect("blocking task panicked")
    .expect("to_vec failed");

    assert_eq!(ids, vec![3, 1, 4, 1, 5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn to_record_batch_in_spawn_blocking() {
    let df = sample_df();
    let batch = tokio::task::spawn_blocking(move || df.to_record_batch())
        .await
        .expect("blocking task panicked")
        .expect("to_record_batch failed");

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(batch.schema().field(0).name(), "id");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_round_trip_across_spawn_blocking() {
    use cudf::io::arrow::dataframe_from_record_batch;

    let df = sample_df();

    // GPU → host (RecordBatch) in spawn_blocking.
    let batch = tokio::task::spawn_blocking(move || df.to_record_batch())
        .await
        .expect("blocking task panicked")
        .expect("to_record_batch failed");

    // host → GPU (DataFrame) in another spawn_blocking.
    let df2 = tokio::task::spawn_blocking(move || dataframe_from_record_batch(&batch))
        .await
        .expect("blocking task panicked")
        .expect("from_record_batch failed");

    assert_eq!(df2.num_rows(), 5);
    let ids: Vec<i32> = df2.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(ids, vec![3, 1, 4, 1, 5]);
}

// ===========================================================================
// Category 7: Multiple operations chained in one spawn_blocking
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_pipeline_in_spawn_blocking() {
    // Full pipeline: create → sort → head → groupby → materialize.
    let result = tokio::task::spawn_blocking(|| -> cudf::Result<Vec<(i32, i64)>> {
        let keys = Column::from_slice(&[2i32, 1, 2, 1, 3, 3]).unwrap();
        let vals = Column::from_slice(&[10i64, 20, 30, 40, 50, 60]).unwrap();
        let df = crate::common::dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals])?;

        // Sort by key.
        let sorted = df.sort_values(["key"], SortOptions::new())?;

        // Take first 4 rows.
        let top = sorted.head(4)?;
        assert_eq!(top.num_rows(), 4);

        // GroupBy sum.
        let agg_result = top.groupby(["key"]).agg(&[AggSpec::new("value", Agg::Sum)])?;

        // Materialize to host.
        let key_data: Vec<i32> = agg_result.series("key").unwrap().to_vec_strict::<i32>()?;
        let sum_data: Vec<i64> = agg_result.series("value_sum").unwrap().to_vec_strict::<i64>()?;

        let mut pairs: Vec<(i32, i64)> = key_data.into_iter().zip(sum_data).collect();
        pairs.sort_by_key(|&(k, _)| k);
        Ok(pairs)
    })
    .await
    .expect("blocking task panicked")
    .expect("pipeline failed");

    // After sorting [2,1,2,1,3,3] → [1,1,2,2,3,3], head(4) → [1,1,2,2].
    // GroupBy sum: key=1 → 20+40=60, key=2 → 10+30=40.
    assert_eq!(result, vec![(1, 60), (2, 40)]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_inside_spawn_blocking() {
    let result = tokio::task::spawn_blocking(|| -> cudf::Result<usize> {
        let left = crate::common::dataframe_from_columns(
            vec!["id".into(), "name".into()],
            vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_strings(&["Alice", "Bob", "Carol"]).unwrap()],
        )?;

        let right = crate::common::dataframe_from_columns(
            vec!["id".into(), "score".into()],
            vec![Column::from_slice(&[2i32, 3, 4]).unwrap(), Column::from_slice(&[85.0f64, 92.0, 78.0]).unwrap()],
        )?;

        let opts = cudf::MergeOptions::new().on(["id"]).how_inner();
        let joined = left.merge(&right, opts)?;
        Ok(joined.num_rows())
    })
    .await
    .expect("blocking task panicked")
    .expect("join failed");

    // Inner join on id: matches 2 and 3.
    assert_eq!(result, 2);
}

// ===========================================================================
// Category 8: Filter operations in spawn_blocking
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_filter_in_spawn_blocking() {
    let df = sample_df();
    let result = tokio::task::spawn_blocking(move || -> cudf::Result<Vec<i32>> {
        // Build a boolean mask: score > 20.0
        let score_series = df.series("score")?;
        let mask_series = score_series.gt_scalar(20.0f64)?;
        let filtered = df.filter(&mask_series)?;

        // Materialize filtered ids to host.
        let ids = filtered.series("id").unwrap().to_vec_strict::<i32>()?;
        Ok(ids)
    })
    .await
    .expect("blocking task panicked")
    .expect("filter pipeline failed");

    // Original data: id=[3,1,4,1,5], score=[30,10,40,15,50].
    // score > 20 → rows 0,2,4 → id=[3,4,5].
    assert_eq!(result, vec![3, 4, 5]);
}

// ===========================================================================
// Category 9: Concurrent compute on shared data (Sync soundness)
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_sort_on_shared_arc_dataframe() {
    // Two spawn_blocking workers both sort the same Arc<DataFrame>.
    // Each sort produces an independent result — proving Sync is sound
    // for concurrent compute on shared data.
    let df = Arc::new(sample_df());

    let df1 = Arc::clone(&df);
    let df2 = Arc::clone(&df);

    let (sorted_asc, sorted_desc) = tokio::join!(
        tokio::task::spawn_blocking(move || { df1.sort_values(["id"], SortOptions::new()) }),
        tokio::task::spawn_blocking(move || {
            df2.sort_values(["id"], SortOptions::new().with_descending(vec![true]))
        }),
    );

    let asc = sorted_asc.expect("asc task panicked").expect("asc sort failed");
    let desc = sorted_desc.expect("desc task panicked").expect("desc sort failed");

    let asc_ids: Vec<i32> = asc.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    let desc_ids: Vec<i32> = desc.series("id").unwrap().to_vec_strict::<i32>().unwrap();

    assert_eq!(asc_ids, vec![1, 1, 3, 4, 5]);
    assert_eq!(desc_ids, vec![5, 4, 3, 1, 1]);
}

// ===========================================================================
// Runtime responsiveness: async heartbeat during GPU work
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn async_runtime_stays_responsive_during_dataframe_gpu_ops() {
    // Heartbeat task: counts async ticks while GPU work runs.
    let ticker = tokio::spawn(async {
        let start = Instant::now();
        let mut ticks = 0usize;
        while start.elapsed() < Duration::from_millis(300) {
            tokio::time::sleep(Duration::from_millis(5)).await;
            ticks += 1;
        }
        ticks
    });

    // Stress the GPU via DataFrame operations from spawn_blocking.
    let workers = 4;
    let mut handles = Vec::with_capacity(workers);

    for _ in 0..workers {
        handles.push(tokio::task::spawn_blocking(|| {
            for _ in 0..16 {
                let df = sample_df();
                let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
                let _ids: Vec<i32> = sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap();
            }
        }));
    }

    for handle in handles {
        tokio::time::timeout(Duration::from_secs(20), handle)
            .await
            .expect("spawn_blocking task timed out")
            .expect("spawn_blocking task panicked");
    }

    let ticks = ticker.await.expect("ticker task panicked");
    assert!(ticks >= 10, "Tokio runtime heartbeat stalled during DataFrame GPU ops (ticks={ticks})");
}

// ===========================================================================
// Timeout behavior: wrapping spawn_blocking in timeout is safe
// ===========================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_on_spawn_blocking_is_safe() {
    // Even if the timeout fires, the blocking thread completes its GPU work
    // cleanly. No crash, no corruption.
    let handle = tokio::task::spawn_blocking(|| {
        let df = sample_df();
        let sorted = df.sort_values(["id"], SortOptions::new()).unwrap();
        assert_eq!(sorted.num_rows(), 5);
        sorted
    });

    // Use a generous timeout — we are testing safety, not speed.
    let result = tokio::time::timeout(Duration::from_secs(30), handle).await;

    match result {
        Ok(Ok(df)) => {
            // Normal case: completed within timeout.
            assert_eq!(df.num_rows(), 5);
        },
        Ok(Err(e)) => panic!("spawn_blocking panicked: {e}"),
        Err(_elapsed) => {
            // Timeout fired. The blocking thread is still running (or
            // finished). This is safe — no GPU corruption, the DataFrame
            // will be dropped when the blocking thread exits.
            //
            // In CI this should not happen (30s is very generous), but
            // this branch documents the contract: timeout doesn't crash.
        },
    }
}
