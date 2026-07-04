mod single_gpu;

/// Single-GPU OOM isolation experiment: **isolated Runtimes**.
///
/// Each of 4 `spawn_blocking` threads builds its own `Runtime` — separate CUDA
/// stream, separate device memory resource, but all on the same GPU (device 0).
/// One thread grows data until OutOfMemory.  The question: does per-thread
/// stream/MR isolation prevent OOM from cascading to other threads?
#[tokio::main(flavor = "multi_thread")]
async fn main() -> cudf::Result<()> {
    single_gpu::run_isolated_runtime_oom().await
}
