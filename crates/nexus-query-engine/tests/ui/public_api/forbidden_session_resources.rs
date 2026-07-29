use nexus_query_engine::runtime::GpuBackendSession;

fn expose_resources(session: &GpuBackendSession) {
    let _service = session.query_service();
    let _registry = session.device_registry();
    let _stream_pool = session.service_stream_pool();
}

fn main() {
    let _ = expose_resources;
}
