use cudf::DeviceMemoryManagerBase;
use nexus_query_engine::runtime::QueryExecutionConfig;

fn main() {
    let _ = QueryExecutionConfig {
        device_memory_manager_base: DeviceMemoryManagerBase::Pool {
            initial_pool_size: 64 * 1024 * 1024,
            maximum_pool_size: Some(64 * 1024 * 1024),
        },
        ..QueryExecutionConfig::default()
    };
}
