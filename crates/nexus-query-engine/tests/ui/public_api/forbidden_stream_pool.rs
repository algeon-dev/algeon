use nexus_query_engine::runtime::ServiceStreamPool;

fn main() {
    let device = cudf::DeviceId::DEFAULT;
    let _ = ServiceStreamPool::new(device, 1);
}
