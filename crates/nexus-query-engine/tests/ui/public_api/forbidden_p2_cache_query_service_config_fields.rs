use nexus_query_engine::runtime::QueryServiceConfig;

fn main() {
    let _ = QueryServiceConfig {
        local_parquet_source_cache_max_retained_bytes: Some(1),
        resident_graph_cache_max_retained_bytes: Some(1),
        ..QueryServiceConfig::default()
    };
}
