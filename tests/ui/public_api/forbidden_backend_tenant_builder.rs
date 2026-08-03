use datafusion_nexus::backend::NexusGpuBackend;

fn main() {
    let _ = NexusGpuBackend::builder().tenant_id("tenant");
}
