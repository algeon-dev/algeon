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
mod physical_plan;
pub mod planner;
/// GPU fallback policy and stable fallback reason vocabulary.
pub mod policy;
/// Native planning report candidate/outcome types.
pub mod report;
#[cfg(feature = "server")]
pub mod server;
pub mod session;
pub mod session_config;
mod source_cardinality;
pub mod table_format;

/// Public adapter error contract.
pub mod error_contracts {
    pub use crate::error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
}

/// Common adapter imports for applications that install the native optimizer
/// and execute supported DataFusion plans.
pub mod prelude {
    pub use crate::{
        CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
        CudfSessionStateBuilderExt, DataFusionNexusSessionConfig, FallbackReason, GpuFallbackPolicy, NativeCudfExec,
        NativeCudfExecutionConfig, NativeMemoryPolicyConfig, PhysicalOptimizerRules, PlanningReport,
        cudf_native_physical_optimizer_rules, cudf_native_physical_optimizer_rules_with_rule,
    };
}

/// Stable planning and runtime report contracts.
pub mod report_contracts {
    pub use crate::{
        native_report::{
            NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION,
            NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
            NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION,
            NATIVE_TPCDS_PARITY_TSV_HEADER, NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, NativeAggregateStrategyMetricRow,
            NativeDataFusionExecutionMetrics, NativeDataFusionExecutionReport, NativeDataFusionExecutionRow,
            NativeDataFusionExecutionWorkloadInput, NativeDataFusionExecutionWorkloadItem,
            NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPlanningRow,
            NativeDataFusionPlanningWorkloadInput, NativeDataFusionPlanningWorkloadItem,
            NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome,
            NativeDynamicFilterMeasuredReductionStatus, NativeDynamicFilterPostDecodeReductionStage,
            NativeDynamicFilterRuntimeMetricRow, NativeDynamicFilterWorkAvoidanceStatus, NativeFallbackPressureOutcome,
            NativeFallbackPressureReport, NativeFallbackPressureRow, NativeFallbackPressureWorkloadInput,
            NativeFallbackPressureWorkloadItem, NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow,
            NativeJoinDiagnosticSummary, NativeJoinStrategyMetricRow, NativeLakehouseCacheFillEventMetricRow,
            NativePlanningFailure, NativeSourceDiagnosticSummary, NativeSourceReadRuntimeMetricRow,
            NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow, NativeTpcdsParityCpuOutcome,
            NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus,
            NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityNativePlanningOutcome,
            NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport,
            NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityComparisonResult,
            ParityFloatTolerance, ParityStageStatus, ParityStatus, WorkloadDatasetMetadata, compare_record_batches,
            sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
        },
        report::{CandidateOutcome, GpuCandidate, PlanningReport},
    };
}

/// DataFusion-facing execution wrapper contracts.
pub mod execution_contracts {
    pub use crate::{
        native::{NativeLoweringResult, lower_physical_plan_to_native},
        native_exec::{
            NATIVE_CUDF_DATAFUSION_METRIC_NAMES, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
            NativeCudfGpuProducerExec,
        },
        planner::{
            CUDF_TABLE_FORMAT_GPU_ONLY_LEAF_GUARD_RULE_NAME, CudfNativePhysicalOptimizerRule, CudfOptimizerConfig,
            CudfTableFormatGpuOnlyLeafGuardRule, NativeAdaptivePipelineCandidateDiagnostics,
            NativeAdaptivePipelineDiagnostics, NativePlanningDiagnostics,
            NativeStrictBoundedAcceptedCandidateDiagnostics, NativeStrictBoundedCompiledGraphDiagnostics,
            NativeStrictBoundedContractDiagnostics, NativeStrictBoundedRejectedCandidateDiagnostics,
            STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
        },
        session_config::{
            DATAFUSION_NEXUS_CONFIG_PREFIX, DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS, DataFusionNexusSessionConfig,
            DataFusionNexusSessionConfigOption, NativeMemoryPolicyConfig,
        },
    };
}

/// Table-format scan and lakehouse source contracts.
pub mod table_format_contracts {
    pub use crate::{
        source_cardinality::{
            SOURCE_UNIQUE_KEYS_METADATA_KEY, register_source_unique_keys_for_path, schema_with_source_unique_keys,
        },
        table_format::{
            AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, CudfTableFormatScanExec,
            NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV, NEXUS_ICEBERG_REMOTE_DIRECT_ENV,
            NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV, NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV,
            NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV, NEXUS_LAKEHOUSE_CACHE_READ_ENV,
            NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, TableFormat, TableFormatColumnMapping, TableFormatColumnRole,
            TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat, TableFormatDeleteFile,
            TableFormatDeleteFileContent, TableFormatDirectDeviceReadDisposition, TableFormatLakehouseCachePolicy,
            TableFormatLakehouseCacheReadPolicy, TableFormatParquetReadStrategyConfig,
            TableFormatParquetReadStrategyPreference, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
            TableFormatPartitionValue, TableFormatPlanner, TableFormatPlanningDeadline, TableFormatReadCapability,
            TableFormatScalarValue, TableFormatScanDiagnostics, TableFormatScanPlan, TableFormatScanRequest,
            TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode, resolve_table_format_parquet_source,
            resolve_table_format_parquet_source_with_metadata,
        },
    };
}

/// Native engine contracts re-exported by the adapter for compatibility.
///
/// New specialist code should generally import directly from
/// `nexus_query_engine::{plan,runtime,source,capability,metrics}`.
pub mod engine_contracts {
    pub use nexus_query_engine::{
        AdaptiveMemoryHistory, DynamicFilterExecutionOptions, GpuLocalParquetSourceCache,
        GpuLocalParquetSourceCacheKey, KeyedHashJoinStrategyOptions, NativeOptimizerOptions,
        NativePipelineExecutionMode, NativePlanCpuPreferenceReason, NativePlanPreferenceOptions, RemoteParquetReadMode,
        S3SourceStorageCredential, S3SourceStorageProfile, SemijoinReductionOptions, SourceStorageProfile,
        StatisticalAggregateExecutionMode, StatisticalAggregateToleranceContract,
    };
}

#[cfg(feature = "iceberg")]
/// Optional Iceberg catalog and table-provider contracts.
pub mod iceberg_contracts {
    pub use crate::iceberg::{
        CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig, GlueIcebergTableDiagnostics,
        GlueIcebergTableRegistration, IcebergTableFormatPlanner, LoadedGlueIcebergTable,
        NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN,
        NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
        NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
        catalog::{
            CudfIcebergCatalogProvider, CudfIcebergSchemaProvider, GlueIcebergCatalogBackend, GlueIcebergCatalogConfig,
            IcebergCatalogBackend, IcebergProviderKind, RestIcebergCatalogBackend, RestIcebergCatalogConfig,
            build_catalog_provider_from_backend, build_scoped_catalog_provider_from_backend,
        },
        iceberg_file_scan_tasks_to_table_format_scan_plan, iceberg_schema_to_datafusion_arrow_schema, load_glue_table,
        register_glue_table,
    };
}

#[cfg(feature = "cugraph")]
/// Optional SQL-visible cuGraph contracts.
pub mod graph_contracts {
    pub use crate::graph_sql::{
        CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
        CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION,
        CUGRAPH_ECG_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION, CUGRAPH_FORCE_ATLAS2_FUNCTION,
        CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_K_CORE_FUNCTION,
        CUGRAPH_KATZ_CENTRALITY_FUNCTION, CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION,
        CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION, CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION,
        CUGRAPH_OPERATION_REGISTRY, CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_PAGERANK_FUNCTION,
        CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION, CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION,
        CUGRAPH_SSSP_FUNCTION, CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
        CUGRAPH_VALIDATE_CALL_FUNCTION, CUGRAPH_VALIDATE_CALL_SCHEMA_VERSION,
        CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION, CugraphAlgoTableProvider, CugraphAlgorithm,
        CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsOutputMode, CugraphBfsOutputOptions, CugraphBfsParams,
        CugraphBfsSource, CugraphBfsSourceVertex, CugraphCoreNumberParams, CugraphDegreesAllParams, CugraphEcgParams,
        CugraphEigenvectorCentralityParams, CugraphForceAtlas2Params, CugraphGraphConstructionPolicy,
        CugraphGraphOptions, CugraphHitsParams, CugraphInDegreesAllParams, CugraphKCoreDegreeType, CugraphKCoreParams,
        CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams, CugraphMstParams,
        CugraphOperationDefinition, CugraphOutDegreesAllParams, CugraphPageRankParams,
        CugraphPageRankPersonalizationTable, CugraphPersonalizedPageRankParams, CugraphSccParams,
        CugraphSpectralClusteringParams, CugraphSqlConfig, CugraphSsspParams, CugraphTableReference,
        CugraphTriangleCountAllParams, CugraphWccParams, cugraph_table_functions, parse_cugraph_algorithm_spec,
        register_cugraph_udtfs, stable_result_schema,
    };
}

#[cfg(feature = "server")]
/// Optional Flight SQL server contracts.
pub mod server_contracts {
    pub use crate::server::*;
}

pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
#[cfg(feature = "cugraph")]
pub use graph_sql::{
    CUGRAPH_BALANCED_CUT_CLUSTERING_FUNCTION, CUGRAPH_BFS_FUNCTION, CUGRAPH_CORE_NUMBER_FUNCTION,
    CUGRAPH_DEGREES_ALL_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_FUNCTION, CUGRAPH_DESCRIBE_ALGORITHM_SCHEMA_VERSION,
    CUGRAPH_ECG_FUNCTION, CUGRAPH_EIGENVECTOR_CENTRALITY_FUNCTION, CUGRAPH_FORCE_ATLAS2_FUNCTION,
    CUGRAPH_HITS_FUNCTION, CUGRAPH_IN_DEGREES_ALL_FUNCTION, CUGRAPH_K_CORE_FUNCTION, CUGRAPH_KATZ_CENTRALITY_FUNCTION,
    CUGRAPH_LEIDEN_FUNCTION, CUGRAPH_LIST_ALGORITHMS_FUNCTION, CUGRAPH_LIST_ALGORITHMS_SCHEMA_VERSION,
    CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_MINIMUM_SPANNING_TREE_FUNCTION, CUGRAPH_OPERATION_REGISTRY,
    CUGRAPH_OUT_DEGREES_ALL_FUNCTION, CUGRAPH_PAGERANK_FUNCTION, CUGRAPH_PERSONALIZED_PAGERANK_FUNCTION,
    CUGRAPH_SPECTRAL_MODULARITY_MAXIMIZATION_FUNCTION, CUGRAPH_SSSP_FUNCTION,
    CUGRAPH_STRONGLY_CONNECTED_COMPONENTS_FUNCTION, CUGRAPH_TRIANGLE_COUNT_ALL_FUNCTION,
    CUGRAPH_VALIDATE_CALL_FUNCTION, CUGRAPH_VALIDATE_CALL_SCHEMA_VERSION, CUGRAPH_WEAKLY_CONNECTED_COMPONENTS_FUNCTION,
    CugraphAlgoTableProvider, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsOutputMode,
    CugraphBfsOutputOptions, CugraphBfsParams, CugraphBfsSource, CugraphBfsSourceVertex, CugraphCoreNumberParams,
    CugraphDegreesAllParams, CugraphEcgParams, CugraphEigenvectorCentralityParams, CugraphForceAtlas2Params,
    CugraphGraphConstructionPolicy, CugraphGraphOptions, CugraphHitsParams, CugraphInDegreesAllParams,
    CugraphKCoreDegreeType, CugraphKCoreParams, CugraphKatzCentralityParams, CugraphLeidenParams, CugraphLouvainParams,
    CugraphMstParams, CugraphOperationDefinition, CugraphOutDegreesAllParams, CugraphPageRankParams,
    CugraphPageRankPersonalizationTable, CugraphPersonalizedPageRankParams, CugraphSccParams,
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
    CudfIcebergTableProvider, GlueIcebergCredentialSource, GlueIcebergTableConfig, GlueIcebergTableDiagnostics,
    GlueIcebergTableRegistration, IcebergTableFormatPlanner, LoadedGlueIcebergTable,
    NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV, NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN,
    NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV, NEXUS_ICEBERG_S3_ENDPOINT_ENV, NEXUS_ICEBERG_S3_PATH_STYLE_ENV,
    NEXUS_ICEBERG_S3_REGION_ENV, NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV, NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV,
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
    NativeFallbackPressureWorkloadItem, NativeGraphReservationMetricRow, NativeGraphRuntimeMetricRow,
    NativeJoinDiagnosticSummary, NativeJoinStrategyMetricRow, NativeLakehouseCacheFillEventMetricRow,
    NativePlanningFailure, NativeSourceDiagnosticSummary, NativeSourceReadRuntimeMetricRow,
    NativeTaskOperatorMetricRow, NativeTaskRuntimeMetricRow, NativeTpcdsParityCpuOutcome,
    NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus, NativeTpcdsParityExternalOutputSummary,
    NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome,
    NativeTpcdsParityReport, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode,
    ParityComparisonResult, ParityFloatTolerance, ParityStageStatus, ParityStatus, WorkloadDatasetMetadata,
    compare_record_batches, sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
};
pub use nexus_query_engine::{
    AdaptiveMemoryHistory, DynamicFilterExecutionOptions, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey,
    KeyedHashJoinStrategyOptions, NativeOptimizerOptions, NativePipelineExecutionMode, NativePlanCpuPreferenceReason,
    NativePlanPreferenceOptions, RemoteParquetReadMode, S3SourceStorageCredential, S3SourceStorageProfile,
    SemijoinReductionOptions, SourceStorageProfile, StatisticalAggregateExecutionMode,
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
pub use session_config::{
    DATAFUSION_NEXUS_CONFIG_PREFIX, DATAFUSION_NEXUS_SESSION_CONFIG_OPTIONS, DataFusionNexusSessionConfig,
    DataFusionNexusSessionConfigOption, NativeMemoryPolicyConfig,
};
pub use source_cardinality::{
    SOURCE_UNIQUE_KEYS_METADATA_KEY, register_source_unique_keys_for_path, schema_with_source_unique_keys,
};
pub use table_format::{
    AWS_ENDPOINT_URL_ENV, AWS_ENDPOINT_URL_S3_ENV, CudfTableFormatScanExec, NEXUS_ICEBERG_PARQUET_READ_STRATEGY_ENV,
    NEXUS_ICEBERG_REMOTE_DIRECT_ENV, NEXUS_ICEBERG_REMOTE_DIRECT_READ_MODE_ENV,
    NEXUS_ICEBERG_REMOTE_HOST_BUFFER_RANGED_ENV, NEXUS_LAKEHOUSE_CACHE_MAX_BYTES_ENV, NEXUS_LAKEHOUSE_CACHE_POLICY_ENV,
    NEXUS_LAKEHOUSE_CACHE_READ_ENV, NEXUS_LAKEHOUSE_CACHE_ROOT_ENV, TableFormat, TableFormatColumnMapping,
    TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat,
    TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatDirectDeviceReadDisposition,
    TableFormatLakehouseCachePolicy, TableFormatLakehouseCacheReadPolicy, TableFormatParquetReadStrategyConfig,
    TableFormatParquetReadStrategyPreference, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
    TableFormatPartitionValue, TableFormatPlanner, TableFormatPlanningDeadline, TableFormatReadCapability,
    TableFormatScalarValue, TableFormatScanDiagnostics, TableFormatScanPlan, TableFormatScanRequest,
    TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode, resolve_table_format_parquet_source,
    resolve_table_format_parquet_source_with_metadata,
};
