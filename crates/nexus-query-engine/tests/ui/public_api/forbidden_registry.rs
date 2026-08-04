use nexus_query_engine::runtime::{DeviceRegistry, DeviceRegistryConfig, NexusGpuInventory};

fn main() {
    let inventory = NexusGpuInventory;
    let config = DeviceRegistryConfig::new(vec![0], 1);
    let _ = DeviceRegistry::install_resolved;
}
