//! DataFusion adapter for native cudf-rs GPU execution.
//!
//! The crate exposes the DataFusion-facing boundary around the native
//! [`nexus_query_engine`] IR:
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
//!
//! DataFusion-specific planning belongs here. DataFusion-free native IR,
//! runtime policy, capability admission, and cudf-rs execution live in
//! [`nexus_query_engine`].

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

pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
#[cfg(feature = "cugraph")]
pub use graph_sql::{
    CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION,
    CUGRAPH_ECG_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION, CUGRAPH_HITS_FUNCTION,
    CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_K_CORE_FUNCTION, CUGRAPH_KATZ_CENTRALITY_FUNCTION,
    CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION,
    CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
    CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_PAGERANK_FUNCTION, CUGRAPH_PAGERANK_STRICT_FUNCTION,
    CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_VALIDATE_CALL_FUNCTION, CUGRAPH_VALIDATE_CALL_SCHEMA_VERSION, CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
    CugraphAlgoTableProvider, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsOutputMode,
    CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsSourceVertex, CugraphCoreNumberParams,
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
    IcebergCatalogBackend, IcebergProviderKind, RestIcebergCatalogBackend, RestIcebergCatalogConfig,
    build_catalog_provider_from_backend, build_scoped_catalog_provider_from_backend,
};
#[cfg(feature = "iceberg")]
pub use iceberg::{
    CUDF_ICEBERG_S3_ACCESS_KEY_ID_ENV, CUDF_ICEBERG_S3_ENDPOINT_ENV, CUDF_ICEBERG_S3_PATH_STYLE_ENV,
    CUDF_ICEBERG_S3_REGION_ENV, CUDF_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, CUDF_ICEBERG_S3_SESSION_TOKEN_ENV,
    CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig, GlueIcebergTableDiagnostics,
    GlueIcebergTableRegistration, IcebergTableFormatPlanner, LoadedGlueIcebergTable,
    iceberg_file_scan_tasks_to_table_format_scan_plan, iceberg_schema_to_datafusion_arrow_schema, load_glue_table,
    register_glue_table,
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
    NativeDynamicFilterMeasuredReductionStatus, NativeDynamicFilterPostDecodeReductionStage,
    NativeDynamicFilterRuntimeMetricRow, NativeDynamicFilterWorkAvoidanceStatus, NativeFallbackPressureOutcome,
    NativeFallbackPressureReport, NativeFallbackPressureRow, NativeFallbackPressureWorkloadInput,
    NativeFallbackPressureWorkloadItem, NativeGraphRuntimeMetricRow, NativeJoinDiagnosticSummary,
    NativeJoinStrategyMetricRow, NativePlanningFailure, NativeSourceDiagnosticSummary,
    NativeSourceReadRuntimeMetricRow, NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow,
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus,
    NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityNativePlanningOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport,
    NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityComparisonResult,
    ParityFloatTolerance, ParityStageStatus, ParityStatus, WorkloadDatasetMetadata, compare_record_batches,
    sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};
pub use nexus_query_engine::{
    AdaptiveMemoryHistory, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey, NativeOptimizerOptions,
    NativePipelineExecutionMode, NativePlanCpuPreferenceReason, NativePlanPreferenceOptions, S3SourceStorageCredential,
    S3SourceStorageProfile, SemijoinReductionOptions, SourceStorageProfile, StatisticalAggregateExecutionMode,
    StatisticalAggregateToleranceContract,
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
    TableFormatDirectDeviceReadDisposition, TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyConfig,
    TableFormatParquetReadStrategyPreference, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
    TableFormatPartitionValue, TableFormatPlanner, TableFormatReadCapability, TableFormatScalarValue,
    TableFormatScanDiagnostics, TableFormatScanPlan, TableFormatScanRequest, TableFormatUnsupportedFeature,
    TableFormatUnsupportedFeatureCode, resolve_table_format_parquet_source,
    resolve_table_format_parquet_source_with_metadata,
};
