mod single_gpu;

/// Single-GPU OOM isolation experiment: **shared Runtime**.
///
/// All 4 `spawn_blocking` threads clone the same `Runtime` — same CUDA stream,
/// same device memory resource, same device (GPU 0).  One thread grows data
/// until OutOfMemory.  The question: does OOM in one thread cascade to the
/// other 3?
#[tokio::main(flavor = "multi_thread")]
async fn main() -> cudf::Result<()> {
    single_gpu::run_shared_runtime_oom().await
}
