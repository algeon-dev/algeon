use nexus_query_engine::runtime::QueryServiceConfig;

fn main() {
    let _ = QueryServiceConfig {
        max_concurrent_queries: 1,
        max_queries_per_device: Some(1),
        max_admission_queue_len: 1,
        admission_wait_timeout: std::time::Duration::from_secs(1),
        ..QueryServiceConfig::default()
    };
}
