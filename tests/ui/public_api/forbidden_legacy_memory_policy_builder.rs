use datafusion_nexus::backend::NexusGpuBackend;

fn main() {
    let _ = NexusGpuBackend::builder().memory_policy(());
}
