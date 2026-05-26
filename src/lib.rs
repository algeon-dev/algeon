//! DataFusion adapter for native cudf-rs GPU execution.
//!
//! The crate exposes the DataFusion-facing boundary around the native
//! [`cudf_query_engine`] IR:
//!
//! - [`session`] wires the native physical optimizer into a
//!   `SessionStateBuilder` while preserving DataFusion's final validation rule.
//! - [`planner`] owns native rewrite policy, fallback behavior, and the
//!   strict/adaptive planning diagnostics consumed by reports.
//! - [`native_exec`] contains the `ExecutionPlan` wrappers that either
//!   materialize native GPU results back to Arrow or pass GPU-resident
//!   fragments between native subplans.
//! - [`native_report`] defines the TSV/report contracts emitted by benchmark
//!   and diagnostic binaries. Column order and schema-version constants are
//!   part of the public surface.
//! - [`tpch`], [`tpcds`], and [`clickbench`] provide workload metadata,
//!   fixture discovery, session configs, and workload-item builders.
//!
//! DataFusion-specific planning belongs here. DataFusion-free native IR,
//! runtime policy, capability admission, and cudf-rs execution live in
//! [`cudf_query_engine`].

#[cfg(feature = "cugraph")]
#[doc(hidden)]
pub mod cugraph_parity;
pub mod error;
#[cfg(feature = "cugraph")]
pub mod graph_sql;
#[cfg(feature = "iceberg")]
pub mod iceberg;
pub mod native;
pub mod native_exec;
pub mod native_report;
pub mod planner;
/// GPU fallback policy and stable fallback reason vocabulary.
pub mod policy;
/// Native planning report candidate/outcome types.
pub mod report;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod table_format;
mod workload;

pub use clickbench::{
    CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE,
    CUDF_CLICKBENCH_CONTENT_FINGERPRINT_ENV, CUDF_CLICKBENCH_GENERATION_COMMAND_ENV,
    CUDF_CLICKBENCH_NATIVE_DEVICE_BUDGET_BYTES_ENV, CUDF_CLICKBENCH_NATIVE_MAX_RETRY_ATTEMPTS_ENV,
    CUDF_CLICKBENCH_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV, CUDF_CLICKBENCH_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV,
    CUDF_CLICKBENCH_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV, CUDF_CLICKBENCH_PARQUET_DIR_ENV, ClickbenchDatasetMetadata,
    ClickbenchExecutionProgressEvent, ClickbenchLayout, ClickbenchQuery, ClickbenchWorkloadError,
    ClickbenchWorkloadResult, DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR, DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET,
    DEFAULT_NVME_CLICKBENCH_PARTITIONED_DIR, clickbench_datafusion_execution_workload_items,
    clickbench_datafusion_execution_workload_items_filtered,
    clickbench_datafusion_execution_workload_items_with_optimizer_config,
    clickbench_datafusion_planning_workload_items, clickbench_datafusion_planning_workload_items_with_optimizer_config,
    clickbench_fallback_pressure_workload_items, clickbench_native_optimizer_config, clickbench_query_ids,
    clickbench_report_session_config, clickbench_tuned_cpu_session_config, describe_clickbench_dataset,
    describe_clickbench_dataset_with_content_fingerprint, discover_clickbench_layout,
    external_clickbench_dataset_metadata, prepare_clickbench_report_dataset, register_clickbench_table,
};
pub use cudf_query_engine::{
    AdaptiveMemoryHistory, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey, NativeOptimizerOptions,
    NativePipelineExecutionMode, SemijoinReductionOptions,
};
pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
#[cfg(feature = "cugraph")]
pub use graph_sql::{
    CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_ECG_FUNCTION,
    CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION, CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION,
    CUGRAPH_K_CORE_FUNCTION, CUGRAPH_KATZ_CENTRALITY_FUNCTION, CUGRAPH_LEIDEN_FUNCTION,
    CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION,
    CUGRAPH_OPERATION_REGISTRY, CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_PAGERANK_FUNCTION,
    CUGRAPH_PAGERANK_STRICT_FUNCTION, CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_VALIDATE_CALL_FUNCTION, CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, CugraphAlgoTableProvider,
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsParams, CugraphCoreNumberParams,
    CugraphDegreesAllParams, CugraphEcgParams, CugraphEigenvectorCentralityParams, CugraphGraphConstructionPolicy,
    CugraphGraphOptions, CugraphHitsParams, CugraphInDegreesAllParams, CugraphKCoreDegreeType, CugraphKCoreParams,
    CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams, CugraphMstParams,
    CugraphOperationDefinition, CugraphOutDegreesAllParams, CugraphPageRankParams, CugraphSccParams,
    CugraphSpectralClusteringParams, CugraphSqlConfig, CugraphSsspParams, CugraphTableReference,
    CugraphTriangleCountAllParams, CugraphWccParams, cugraph_table_functions, parse_cugraph_algorithm_spec,
    register_cugraph_udtfs, stable_result_schema,
};
#[cfg(feature = "iceberg")]
pub use iceberg::catalog::{
    CudfIcebergCatalogProvider, CudfIcebergSchemaProvider, GlueIcebergCatalogBackend, GlueIcebergCatalogConfig,
    IcebergCatalogBackend, IcebergProviderKind, build_catalog_provider_from_backend,
    build_scoped_catalog_provider_from_backend,
};
#[cfg(feature = "iceberg")]
pub use iceberg::{
    CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig, GlueIcebergTableDiagnostics,
    GlueIcebergTableRegistration, IcebergTableFormatPlanner, LoadedGlueIcebergTable,
    iceberg_file_scan_tasks_to_table_format_scan_plan, load_glue_table, register_glue_table,
};
pub use native::{NativeLoweringResult, lower_physical_plan_to_native};
pub use native_exec::{
    NATIVE_CUDF_DATAFUSION_METRIC_NAMES, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
    NativeCudfGpuProducerExec,
};
pub use native_report::{
    NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION,
    NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NATIVE_TPCDS_PARITY_TSV_HEADER,
    NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, NativeAggregateStrategyMetricRow, NativeDataFusionExecutionMetrics,
    NativeDataFusionExecutionReport, NativeDataFusionExecutionRow, NativeDataFusionExecutionWorkloadInput,
    NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
    NativeDataFusionPlanningRow, NativeDataFusionPlanningWorkloadInput, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome,
    NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureRow,
    NativeFallbackPressureWorkloadInput, NativeFallbackPressureWorkloadItem, NativeGraphRuntimeMetricRow,
    NativeJoinDiagnosticSummary, NativeJoinStrategyMetricRow, NativePlanningFailure, NativeSourceDiagnosticSummary,
    NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow, NativeTpcdsParityCpuOutcome,
    NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus, NativeTpcdsParityExternalOutputSummary,
    NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome,
    NativeTpcdsParityReport, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode,
    ParityComparisonResult, ParityFloatTolerance, ParityStageStatus, ParityStatus, WorkloadDatasetMetadata,
    compare_record_batches, sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};
pub use planner::{
    CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
    CudfTableFormatGpuOnlyLeafGuardRule, NativeAdaptivePipelineCandidateDiagnostics, NativeAdaptivePipelineDiagnostics,
    NativePlanningDiagnostics, NativeStrictBoundedAcceptedCandidateDiagnostics,
    NativeStrictBoundedCompiledGraphDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeStrictBoundedRejectedCandidateDiagnostics, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
pub use session::{
    CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME, CudfSessionStateBuilderExt, PhysicalOptimizerRules,
    cudf_native_physical_optimizer_rules, cudf_native_physical_optimizer_rules_with_rule,
};
pub use table_format::{
    CudfTableFormatScanExec, TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource,
    TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent,
    TableFormatDirectDeviceReadDisposition, TableFormatPartitionTransform, TableFormatPartitionValue,
    TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue, TableFormatScanDiagnostics,
    TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
};
pub use tpcds::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_NATIVE_DEVICE_BUDGET_BYTES_ENV,
    CUDF_TPCDS_NATIVE_MAX_RETRY_ATTEMPTS_ENV, CUDF_TPCDS_NATIVE_MAX_ROW_GROUPS_PER_CHUNK_ENV,
    CUDF_TPCDS_NATIVE_MAX_SOURCE_CHUNK_BYTES_ENV, CUDF_TPCDS_NATIVE_MIN_RETRY_CHUNK_ROW_GROUPS_ENV,
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
    tpcds_native_optimizer_config, tpcds_parity_workload_items, tpcds_parity_workload_items_with_external_oracles,
    tpcds_query_ids, tpcds_report_session_config, write_tiny_tpcds_parquet_dataset,
};
pub use tpch::{
    AD_HOC_OLAP_QUERIES, AD_HOC_OLAP_QUERY_COUNT, AdHocOlapCapabilityFirstAdmission, AdHocOlapCurrentStrictAdmission,
    AdHocOlapLoweringExpectation, AdHocOlapQuery, AdHocOlapWorkloadClass, CUDF_TPCH_CONTENT_FINGERPRINT_ENV,
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV,
    DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR, DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR,
    DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR, DEFAULT_NVME_MOUNT_ROOT, DEFAULT_NVME_TPCH_SF1_PARQUET_DIR,
    DEFAULT_NVME_TPCH_SF10_PARQUET_DIR, DEFAULT_NVME_TPCH_SF100_OPTIMIZED_PARQUET_DIR, TPCH_QUERIES, TPCH_QUERY_COUNT,
    TPCH_TABLES, TpchDatasetMetadata, TpchDefaultScale, TpchQuery, TpchTableFileMetadata, TpchTableLayout,
    TpchTableMetadata, TpchWorkloadError, TpchWorkloadResult, ad_hoc_olap_datafusion_planning_workload_items,
    ad_hoc_olap_datafusion_planning_workload_items_with_optimizer_config, ad_hoc_olap_fallback_pressure_workload_items,
    ad_hoc_olap_query_ids, default_nvme_mount_present, default_nvme_tpch_dataset,
    default_nvme_tpch_dataset_with_fingerprint, default_tpch_parquet_root, describe_tpch_dataset,
    describe_tpch_dataset_with_content_fingerprint, discover_tpch_table_layout, external_tpch_dataset_metadata,
    prepare_tpch_report_dataset, register_tpch_table_layout, tiny_tpch_dataset_metadata,
    tpch_datafusion_execution_workload_items, tpch_datafusion_execution_workload_items_with_optimizer_config,
    tpch_datafusion_planning_workload_items, tpch_datafusion_planning_workload_items_with_optimizer_config,
    tpch_fallback_pressure_workload_items, tpch_native_optimizer_config, tpch_query_ids, tpch_report_session_config,
    tpch_scalar_fallback_pressure_workload_items, tpch_tuned_cpu_session_config, write_tiny_tpch_parquet_dataset,
};
pub use workload::{clickbench, tpcds, tpch};
