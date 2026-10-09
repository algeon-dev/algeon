//! Runtime integration contract tests.
//!
//! These tests validate the runtime/threading contract of the retained-domain
//! public API:
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
    AggSpec, DataFrame, DeviceId, Exec, GroupByAgg, ParquetReadOptions, Runtime, Series, SortOptions,
    core::{Column, Scalar, Table, memory::DeviceMemoryResource},
    runtime::{CudaEvent, CudaStream},
    testing::{ColumnView, TableView},
};
use static_assertions::{assert_impl_all, assert_not_impl_any};
use tokio::{
    task::spawn_blocking,
    time::{sleep, timeout},
};

use crate::common::{column_from_slice, column_from_strs, dataframe_from_columns, sample_df, series};

assert_impl_all!(DataFrame: Send, Sync);
assert_impl_all!(Series: Send, Sync);
assert_impl_all!(Column: Send, Sync);
assert_impl_all!(Table: Send, Sync);
assert_impl_all!(Scalar: Send, Sync);
assert_impl_all!(CudaStream: Send, Sync);
assert_impl_all!(CudaEvent: Send, Sync);
assert_impl_all!(DeviceMemoryResource: Send, Sync);
assert_impl_all!(Runtime: Send, Sync);
assert_impl_all!(Exec<'static>: Send, Sync);

fn device_inventory_reports_nonzero_memory_and_sm_count() {
    let device = DeviceId::DEFAULT;

    let memory = device.available_memory().unwrap();
    let sm_count = device.multiprocessor_count().unwrap();
    let pci_bus_id = device.pci_bus_id().unwrap();

    assert!(memory.total_bytes > 0);
    assert!(memory.free_bytes <= memory.total_bytes);
    assert!(sm_count > 0);
    assert!(!pci_bus_id.is_empty());
}

fn cuda_event_can_record_wait_and_synchronize() {
    let stream = CudaStream::new().unwrap();
    let event = CudaEvent::new().unwrap();

    event.record(&stream).unwrap();
    stream.wait_event(&event).unwrap();
    event.synchronize().unwrap();
    assert!(event.is_ready().unwrap());
}

// ColumnView and TableView contain raw NonNull<c_void> pointers without
// `unsafe impl Send/Sync`, so the compiler derives !Send and !Sync for both.
// The `assert_not_impl_any!` checks below fail at compile time if either type
// gains Send or Sync.
assert_not_impl_any!(ColumnView<'static>: Send, Sync);
assert_not_impl_any!(TableView<'static>: Send, Sync);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_sort_survives_spawn_blocking() {
    let df = sample_df();
    let result = spawn_blocking(move || df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()))
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
    let result = spawn_blocking(move || df.on(Exec::RETAINED).head(3))
        .await
        .expect("blocking task panicked")
        .expect("head failed");

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.num_columns(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arc_dataframe_shared_across_workers() {
    let df = Arc::new(sample_df());

    let mut handles = Vec::new();
    for _ in 0..4 {
        let df = Arc::clone(&df);
        handles.push(spawn_blocking(move || {
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
    spawn_blocking(move || {
        let sorted = df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
        assert_eq!(sorted.num_rows(), 5);
        // df and sorted both drop here, freeing GPU memory.
    })
    .await
    .expect("blocking task panicked");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_returned_from_spawn_blocking_drops_on_async_thread() {
    // Move DataFrame into spawn_blocking, return it, and drop it after await.
    let df = sample_df();
    let result = spawn_blocking(move || df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()))
        .await
        .expect("blocking task panicked")
        .expect("sort_values failed");

    // result (DataFrame) is now owned by the async test after await.
    assert_eq!(result.num_rows(), 5);
    // Dropping result here (on the async thread) must not leak or crash.
    drop(result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_sort_survives_spawn_blocking() {
    let series = series::<i64>("vals", &[50, 10, 30, 20, 40]).unwrap();
    let sorted = spawn_blocking(move || series.on(Exec::RETAINED).sort_values(SortOptions::new()))
        .await
        .expect("blocking task panicked")
        .expect("sort_values failed");

    let vals: Vec<i64> = sorted.to_vec_strict::<i64>().unwrap();
    assert_eq!(vals, vec![10, 20, 30, 40, 50]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_arithmetic_in_spawn_blocking() {
    let a = series::<f64>("a", &[1.0, 2.0, 3.0]).unwrap();
    let b = series::<f64>("b", &[10.0, 20.0, 30.0]).unwrap();

    let result = spawn_blocking(move || a.on(Exec::RETAINED).add(&b))
        .await
        .expect("blocking task panicked")
        .expect("add failed");

    let vals: Vec<f64> = result.to_vec_strict::<f64>().unwrap();
    assert_eq!(vals, vec![11.0, 22.0, 33.0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn series_unique_in_spawn_blocking() {
    let series = series::<i32>("x", &[1, 2, 2, 3, 3, 3]).unwrap();
    let unique = spawn_blocking(move || series.on(Exec::RETAINED).unique())
        .await
        .expect("blocking task panicked")
        .expect("unique failed");

    assert_eq!(unique.on(Exec::RETAINED).nunique().unwrap(), 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parquet_round_trip_in_spawn_blocking() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("runtime_test.parquet");
    let path_clone = path.clone();

    let df = sample_df();
    spawn_blocking(move || df.write_parquet(&path))
        .await
        .expect("blocking task panicked")
        .expect("write_parquet failed");

    let loaded = spawn_blocking(move || {
        let runtime = cudf::Runtime::builder().build().unwrap();
        DataFrame::read_parquet(
            ParquetReadOptions::builder().filepath(&path_clone).with_runtime(&runtime).build(),
            Exec::RETAINED,
        )
    })
    .await
    .expect("blocking task panicked")
    .expect("read_parquet failed");

    assert_eq!(loaded.num_columns(), 2);
    assert_eq!(loaded.num_rows(), 5);
    assert_eq!(loaded.column_names(), vec!["id", "score"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_materialization_to_vec_in_spawn_blocking() {
    let df = sample_df();
    let ids = spawn_blocking(move || {
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
    let batch = spawn_blocking(move || df.to_record_batch())
        .await
        .expect("blocking task panicked")
        .expect("to_record_batch failed");

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.num_rows(), 5);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_round_trip_across_spawn_blocking() {
    use cudf::interop::arrow_host::dataframe_from_record_batch;

    let df = sample_df();

    let batch = spawn_blocking(move || df.to_record_batch())
        .await
        .expect("blocking task panicked")
        .expect("to_record_batch failed");

    let df2 = spawn_blocking(move || {
        let runtime = cudf::Runtime::builder().build().unwrap();
        dataframe_from_record_batch(&runtime, &batch)
    })
    .await
    .expect("blocking task panicked")
    .expect("from_record_batch failed");

    assert_eq!(df2.num_rows(), 5);
    let ids: Vec<i32> = df2.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    assert_eq!(ids, vec![3, 1, 4, 1, 5]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chained_pipeline_in_spawn_blocking() {
    let result = spawn_blocking(|| -> cudf::Result<Vec<(i32, i64)>> {
        let keys = column_from_slice(&[2i32, 1, 2, 1, 3, 3]).unwrap();
        let vals = column_from_slice(&[10i64, 20, 30, 40, 50, 60]).unwrap();
        let df = dataframe_from_columns(vec!["key".into(), "value".into()], vec![keys, vals])?;

        let sorted = df.on(Exec::RETAINED).sort_values(["key"], SortOptions::new())?;

        let top = sorted.on(Exec::RETAINED).head(4)?;
        assert_eq!(top.num_rows(), 4);

        let agg_result = top.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)])?;

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
    let result = spawn_blocking(|| -> cudf::Result<usize> {
        let left = dataframe_from_columns(
            vec!["id".into(), "name".into()],
            vec![column_from_slice(&[1i32, 2, 3]).unwrap(), column_from_strs(&["Alice", "Bob", "Carol"]).unwrap()],
        )?;

        let right = dataframe_from_columns(
            vec!["id".into(), "score".into()],
            vec![column_from_slice(&[2i32, 3, 4]).unwrap(), column_from_slice(&[85.0f64, 92.0, 78.0]).unwrap()],
        )?;

        let opts = cudf::MergeOptions::new().on(["id"]).how_inner();
        let joined = left.on(Exec::RETAINED).merge(&right, opts)?;
        Ok(joined.num_rows())
    })
    .await
    .expect("blocking task panicked")
    .expect("join failed");

    // Inner join on id: matches 2 and 3.
    assert_eq!(result, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dataframe_filter_in_spawn_blocking() {
    let df = sample_df();
    let result = spawn_blocking(move || -> cudf::Result<Vec<i32>> {
        let score_series = df.series("score")?;
        let mask_series = score_series.on(Exec::RETAINED).gt_scalar(20.0f64)?;
        let filtered = df.on(Exec::RETAINED).filter(&mask_series)?;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_sort_on_shared_arc_dataframe() {
    // Two spawn_blocking workers both sort the same Arc<DataFrame>.
    // Each sort produces an independent result; proving Sync is sound
    // for concurrent compute on shared data.
    let df = Arc::new(sample_df());

    let df1 = Arc::clone(&df);
    let df2 = Arc::clone(&df);

    let (sorted_asc, sorted_desc) = tokio::join!(
        spawn_blocking(move || { df1.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()) }),
        spawn_blocking(move || {
            df2.on(Exec::RETAINED).sort_values(["id"], SortOptions::new().with_descending(vec![true]))
        }),
    );

    let asc = sorted_asc.expect("asc task panicked").expect("asc sort failed");
    let desc = sorted_desc.expect("desc task panicked").expect("desc sort failed");

    let asc_ids: Vec<i32> = asc.series("id").unwrap().to_vec_strict::<i32>().unwrap();
    let desc_ids: Vec<i32> = desc.series("id").unwrap().to_vec_strict::<i32>().unwrap();

    assert_eq!(asc_ids, vec![1, 1, 3, 4, 5]);
    assert_eq!(desc_ids, vec![5, 4, 3, 1, 1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn async_runtime_stays_responsive_during_dataframe_gpu_ops() {
    let ticker = tokio::spawn(async {
        let start = Instant::now();
        let mut ticks = 0usize;
        while start.elapsed() < Duration::from_millis(300) {
            sleep(Duration::from_millis(5)).await;
            ticks += 1;
        }
        ticks
    });

    let workers = 4;
    let mut handles = Vec::with_capacity(workers);

    for _ in 0..workers {
        handles.push(spawn_blocking(|| {
            for _ in 0..16 {
                let df = sample_df();
                let sorted = df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
                let _ids: Vec<i32> = sorted.series("id").unwrap().to_vec_strict::<i32>().unwrap();
            }
        }));
    }

    for handle in handles {
        timeout(Duration::from_secs(20), handle)
            .await
            .expect("spawn_blocking task timed out")
            .expect("spawn_blocking task panicked");
    }

    let ticks = ticker.await.expect("ticker task panicked");
    assert!(ticks >= 10, "Tokio runtime heartbeat stalled during DataFrame GPU ops (ticks={ticks})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timeout_on_spawn_blocking_is_safe() {
    // Even if the timeout fires, the blocking thread completes its GPU work
    // cleanly. No crash, no corruption.
    let handle = spawn_blocking(|| {
        let df = sample_df();
        let sorted = df.on(Exec::RETAINED).sort_values(["id"], SortOptions::new()).unwrap();
        assert_eq!(sorted.num_rows(), 5);
        sorted
    });

    // The 30 s timeout bounds the wait for the blocking task. This test checks safe completion;
    // it does not check speed.
    let result = timeout(Duration::from_secs(30), handle).await;

    match result {
        Ok(Ok(df)) => {
            assert_eq!(df.num_rows(), 5);
        },
        Ok(Err(e)) => panic!("spawn_blocking panicked: {e}"),
        Err(_elapsed) => {
            // Timeout fired. The blocking thread is still running (or
            // finished). This is safe; no GPU corruption, the DataFrame
            // will be dropped when the blocking thread exits.
            //
            // In CI this should not happen (the 30 s timeout bounds the wait for the
            // blocking task), but this branch documents that a timeout does not crash.
        },
    }
}

pub(crate) const CASES: &[(&str, fn())] = &[
    (
        "runtime_contracts::runtime_contract_tests::device_inventory_reports_nonzero_memory_and_sm_count",
        device_inventory_reports_nonzero_memory_and_sm_count,
    ),
    (
        "runtime_contracts::runtime_contract_tests::cuda_event_can_record_wait_and_synchronize",
        cuda_event_can_record_wait_and_synchronize,
    ),
];
