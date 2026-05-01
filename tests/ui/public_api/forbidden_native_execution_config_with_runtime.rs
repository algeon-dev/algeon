use datafusion_nexus::native_exec::NexusNativeExecutionConfig;

fn main() {
    let runtime = cudf::Runtime::new().unwrap();
    let _config = NexusNativeExecutionConfig::bounded_default().with_runtime(runtime);
}
