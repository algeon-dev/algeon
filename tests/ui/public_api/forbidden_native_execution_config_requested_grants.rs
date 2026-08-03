use datafusion_nexus::native_exec::NexusNativeExecutionConfig;

fn main() {
    let _ = NexusNativeExecutionConfig::bounded_default().with_requested_grant_min_bytes(Some(1024));
    let _ = NexusNativeExecutionConfig::bounded_default().with_requested_grant_bytes(Some(2048));
}
