//! Benchmark and workload application surface for `datafusion-nexus`.
//!
//! This leaf crate owns the TPC-H, TPC-DS, and ClickBench workload catalogs,
//! fixture discovery, DataFusion registration helpers, session/optimizer
//! configs, report work-item builders, and the benchmark/report/triage/stress
//! binaries. The adapter contracts it consumes live in [`datafusion_nexus`].

pub mod tpch_benchmark;
pub mod workload;

#[cfg(feature = "iceberg")]
pub mod iceberg_smoke;

pub use clickbench::{
    CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE,
    CUDF_CLICKBENCH_CONTENT_FINGERPRINT_ENV, CUDF_CLICKBENCH_GENERATION_COMMAND_ENV,
    CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
    CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
    CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, CUDF_CLICKBENCH_PARQUET_DIR_ENV, ClickbenchDatasetMetadata,
    ClickbenchExecutionProgressEvent, ClickbenchLayout, ClickbenchQuery, ClickbenchWorkloadError,
    ClickbenchWorkloadResult, DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR, DEFAULT_LOCAL_CLICKBENCH_SINGLE_DIR,
    DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET, DEFAULT_NVME_CLICKBENCH_PARTITIONED_DIR,
    clickbench_datafusion_execution_workload_items, clickbench_datafusion_execution_workload_items_filtered,
    clickbench_datafusion_execution_workload_items_with_optimizer_config,
    clickbench_datafusion_planning_workload_items, clickbench_datafusion_planning_workload_items_with_optimizer_config,
    clickbench_fallback_pressure_workload_items, clickbench_native_optimizer_config, clickbench_query_ids,
    clickbench_report_session_config, clickbench_tuned_cpu_session_config, describe_clickbench_dataset,
    describe_clickbench_dataset_with_content_fingerprint, discover_clickbench_layout,
    external_clickbench_dataset_metadata, prepare_clickbench_report_dataset, register_clickbench_table,
};
pub use tpcds::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
    CUDF_TPCDS_NATIVE_STATISTICAL_AGGREGATE_EXECUTION_MODE_ENV, CUDF_TPCDS_NATIVE_STDDEV_SAMP_EXECUTION_MODE_ENV,
    CUDF_TPCDS_PARQUET_DIR_ENV, CUDF_TPCDS_SCALE_FACTOR_ENV, DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR,
    DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR, TPCDS_QUERIES, TPCDS_QUERY_COUNT, TPCDS_TABLES, TpcdsDatasetMetadata,
    TpcdsExecutionProgressEvent, TpcdsExternalEngineConfig, TpcdsExternalOracleConfig, TpcdsQuery,
    TpcdsTableFileMetadata, TpcdsTableLayout, TpcdsTableMetadata, TpcdsWorkloadError, TpcdsWorkloadResult,
    default_tpcds_sf1_parquet_root, describe_tpcds_dataset, describe_tpcds_dataset_with_content_fingerprint,
    discover_tpcds_table_layout, external_tpcds_dataset_metadata, prepare_tpcds_report_dataset,
    tiny_tpcds_dataset_metadata, tpcds_datafusion_execution_workload_items,
    tpcds_datafusion_execution_workload_items_filtered,
    tpcds_datafusion_execution_workload_items_with_optimizer_config, tpcds_datafusion_planning_workload_items,
    tpcds_datafusion_planning_workload_items_with_optimizer_config, tpcds_fallback_pressure_workload_items,
    tpcds_native_analytics_optimizer_config, tpcds_native_optimizer_config, tpcds_native_report_session_config,
    tpcds_parity_workload_items, tpcds_parity_workload_items_with_external_oracles,
    tpcds_parity_workload_items_with_session_configs, tpcds_query_ids, tpcds_report_session_config,
    write_tiny_tpcds_parquet_dataset,
};
pub use tpch::{
    ANALYTICAL_OLAP_QUERIES, ANALYTICAL_OLAP_QUERY_COUNT, AnalyticalOlapCapabilityFirstAdmission,
    AnalyticalOlapCurrentStrictAdmission, AnalyticalOlapLoweringExpectation, AnalyticalOlapQuery,
    AnalyticalOlapWorkloadClass, CUDF_TPCH_CONTENT_FINGERPRINT_ENV, CUDF_TPCH_GENERATION_COMMAND_ENV,
    CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV, DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR,
    DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR, DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR, DEFAULT_NVME_MOUNT_ROOT,
    DEFAULT_NVME_TPCH_SF1_PARQUET_DIR, DEFAULT_NVME_TPCH_SF10_PARQUET_DIR,
    DEFAULT_NVME_TPCH_SF100_OPTIMIZED_PARQUET_DIR, TPCH_QUERIES, TPCH_QUERY_COUNT, TPCH_TABLES, TpchDatasetMetadata,
    TpchDefaultScale, TpchQuery, TpchTableFileMetadata, TpchTableLayout, TpchTableMetadata, TpchWorkloadError,
    TpchWorkloadResult, analytical_olap_datafusion_planning_workload_items,
    analytical_olap_datafusion_planning_workload_items_with_optimizer_config,
    analytical_olap_fallback_pressure_workload_items, analytical_olap_query_ids, default_nvme_mount_present,
    default_nvme_tpch_dataset, default_nvme_tpch_dataset_with_fingerprint, default_tpch_parquet_root,
    describe_tpch_dataset, describe_tpch_dataset_with_content_fingerprint, discover_tpch_table_layout,
    external_tpch_dataset_metadata, prepare_tpch_report_dataset, register_tpch_table_layout,
    tiny_tpch_dataset_metadata, tpch_datafusion_execution_workload_items,
    tpch_datafusion_execution_workload_items_with_optimizer_config, tpch_datafusion_planning_workload_items,
    tpch_datafusion_planning_workload_items_with_optimizer_config, tpch_fallback_pressure_workload_items,
    tpch_native_optimizer_config, tpch_query_ids, tpch_report_session_config,
    tpch_scalar_fallback_pressure_workload_items, tpch_tuned_cpu_session_config, write_tiny_tpch_parquet_dataset,
};
pub use workload::{clickbench, tpcds, tpch};
