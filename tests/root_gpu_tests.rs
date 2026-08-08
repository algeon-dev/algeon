#[path = "cuvs_sql_gpu_tests.rs"]
#[cfg(feature = "cuvs")]
mod cuvs_sql_gpu_tests;

#[path = "server_cugraph_tests.rs"]
#[cfg(all(feature = "server", feature = "cugraph"))]
mod server_cugraph_tests;

#[path = "server_cuvs_tests.rs"]
#[cfg(all(feature = "server", feature = "cuvs"))]
mod server_cuvs_tests;

#[path = "session_config_tests.rs"]
mod session_config_tests;

#[path = "native_metrics_tests.rs"]
mod native_metrics_tests;

#[path = "attempt_lifecycle_tests.rs"]
mod attempt_lifecycle_tests;
