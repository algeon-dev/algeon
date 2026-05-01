use nexus_query_engine::runtime::{QueryExecutionConfig, QueueBounds};

fn main() {
    let _ = QueryExecutionConfig {
        queue_bounds: QueueBounds::default(),
        ..QueryExecutionConfig::default()
    };
}
