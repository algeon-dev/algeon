// Dynamic filter optimizer contracts stay in this module via include!
// so test discovery paths remain stable.
include!("dynamic_filter/fixtures.rs");
include!("dynamic_filter/source_routes.rs");
include!("dynamic_filter/residual.rs");
include!("dynamic_filter/shared_sources.rs");
include!("dynamic_filter/aggregate.rs");
include!("dynamic_filter/union.rs");
include!("dynamic_filter/payload_metrics.rs");
