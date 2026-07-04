mod multi_gpu;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> cudf::Result<()> {
    multi_gpu::run_runtime_init_oom_example().await
}
