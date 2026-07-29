use nexus_query_engine::runtime::{QueryService, QueryServiceConfig, QueryServiceTenantConfig};

fn main() {
    let _ = QueryService::new(QueryServiceConfig::default(), vec![QueryServiceTenantConfig::new("tenant", 1)]);
}
