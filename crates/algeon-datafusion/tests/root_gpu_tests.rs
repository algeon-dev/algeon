#[path = "root_gpu_fixture.rs"]
mod root_gpu_fixture;

#[path = "cuvs_sql_gpu_tests.rs"]
#[cfg(feature = "cuvs")]
mod cuvs_sql_gpu_tests;

#[path = "session_config_tests.rs"]
mod session_config_tests;

#[path = "native_metrics_tests.rs"]
mod native_metrics_tests;

#[path = "attempt_lifecycle_tests.rs"]
mod attempt_lifecycle_tests;
