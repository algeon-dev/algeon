use nexus_query_engine::runtime::QueryExecutionConfig;

fn main() {
    let _ = QueryExecutionConfig {
        cache_cap_bytes: 1,
        ..QueryExecutionConfig::default()
    };
}
