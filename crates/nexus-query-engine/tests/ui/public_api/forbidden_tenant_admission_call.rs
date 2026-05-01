use nexus_query_engine::runtime::{QueryServiceAdmissionSnapshot, TenantId, TenantQueueDepth};

fn main() {
    let snapshot = QueryServiceAdmissionSnapshot {
        active_queries: 0,
        queued_queries: 0,
        queued_by_tenant: Vec::new(),
        next_tenant_id: None,
        protected_devices: Vec::new(),
    };
    let _ = TenantQueueDepth { tenant_id: String::new(), queued_queries: 0 };
    let _ = snapshot.queued_by_tenant;
    let _ = snapshot.next_tenant_id;
}
