use datafusion_nexus::backend::{NexusGpuBackend, NexusGpuResourcePolicy};

fn main() {
    let _ = NexusGpuBackend::builder().admission(NexusGpuResourcePolicy::default());
}
