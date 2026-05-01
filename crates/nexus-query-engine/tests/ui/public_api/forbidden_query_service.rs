use nexus_query_engine::runtime::{QueryService, QueryServiceConfig};

fn main() {
    let _ = QueryService::new(QueryServiceConfig::default());
}
