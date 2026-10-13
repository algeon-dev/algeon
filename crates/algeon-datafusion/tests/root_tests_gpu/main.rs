mod root_gpu_fixture;

#[cfg(feature = "cuvs")]
mod cuvs_sql_gpu_tests;

#[cfg(feature = "cuml")]
mod cuml_sql_gpu_tests;

mod session_config_tests;

mod backend_io_config_tests;

mod attempt_lifecycle_tests;
