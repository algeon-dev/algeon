use nexus_query_engine::runtime::gpu::owner::{DeviceOwnerToken, LiveBackendLease};

fn main() {
    let _token = std::mem::size_of::<DeviceOwnerToken<'static>>();
    let _lease = std::mem::size_of::<LiveBackendLease>();
}
