use nexus_query_engine::runtime::{QueryGrantBand, QueryGrantRequest, QueryServiceTenantConfig};

fn main() {
    let _ = QueryServiceTenantConfig::new("tenant", 1024);
    let _ = QueryGrantBand { min_bytes: 1, want_bytes: 2 };
    let _ = QueryGrantRequest::new();
}
