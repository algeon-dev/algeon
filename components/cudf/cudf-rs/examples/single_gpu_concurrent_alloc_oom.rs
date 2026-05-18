mod single_gpu;

/// Single-GPU OOM isolation stress test: **concurrent heavy allocators**.
///
/// All 4 `spawn_blocking` threads build isolated `Runtime`s on device 0.
/// Unlike the other single-GPU tests, healthy threads here continuously
/// allocate GPU memory (re-read parquet, sort, concat, groupby) throughout
/// their lifetime.  A barrier synchronizes all threads so the allocation-heavy
/// phases overlap in time.
///
/// This is the most aggressive test: when the OOM thread fills VRAM, healthy
/// threads are mid-allocation.  If OOM cascades across threads, it will show
/// up here.
#[tokio::main(flavor = "multi_thread")]
async fn main() -> cudf::Result<()> {
    single_gpu::run_concurrent_alloc_oom().await
}
