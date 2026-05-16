//! DataFusion adapter for native cudf-rs GPU execution.

pub mod error;
pub mod native;
pub mod native_exec;
pub mod native_report;
pub mod planner;
pub mod policy;
pub mod report;
pub mod session;
mod workload;

pub use cudf_query_engine::{
    AdaptiveMemoryHistory, GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey, NativePipelineExecutionMode,
};
pub use error::{Error, ErrorCode, ErrorKind, ErrorStatus, Facts, Result};
pub use native::{NativeLoweringResult, lower_physical_plan_to_native};
pub use native_exec::{
    NATIVE_CUDF_DATAFUSION_METRIC_NAMES, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
    NativeCudfGpuProducerExec,
};
pub use native_report::{
    NATIVE_DATAFUSION_EXECUTION_TSV_HEADER, NATIVE_DATAFUSION_EXECUTION_TSV_SCHEMA_VERSION,
    NATIVE_DATAFUSION_PLANNING_TSV_HEADER, NATIVE_DATAFUSION_PLANNING_TSV_SCHEMA_VERSION,
    NATIVE_FALLBACK_PRESSURE_TSV_HEADER, NATIVE_FALLBACK_PRESSURE_TSV_SCHEMA_VERSION, NativeDataFusionExecutionMetrics,
    NativeDataFusionExecutionReport, NativeDataFusionExecutionRow, NativeDataFusionExecutionWorkloadInput,
    NativeDataFusionExecutionWorkloadItem, NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport,
    NativeDataFusionPlanningRow, NativeDataFusionPlanningWorkloadInput, NativeDataFusionPlanningWorkloadItem,
    NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntime, NativeDataFusionRuntimeOutcome,
    NativeFallbackPressureOutcome, NativeFallbackPressureReport, NativeFallbackPressureRow,
    NativeFallbackPressureWorkloadInput, NativeFallbackPressureWorkloadItem, NativeJoinDiagnosticSummary,
    NativeJoinStrategyMetricRow, NativePlanningFailure, NativeSourceDiagnosticSummary, NativeTaskOperatorMetricRow,
    NativeTaskRuntimeMetricRow, WorkloadDatasetMetadata,
};
pub use planner::{
    CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, NativeAdaptivePipelineCandidateDiagnostics,
    NativeAdaptivePipelineDiagnostics, NativePlanningDiagnostics, NativeStrictBoundedAcceptedCandidateDiagnostics,
    NativeStrictBoundedCompiledGraphDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeStrictBoundedRejectedCandidateDiagnostics, STRICT_BOUNDED_NON_JOIN_OUTSIDE_REPORT_SCOPE_REASON,
};
pub use policy::{FallbackReason, GpuFallbackPolicy};
pub use report::{CandidateOutcome, GpuCandidate, PlanningReport};
pub use session::{
    CUDF_NATIVE_PHYSICAL_OPTIMIZER_RULE_NAME, CudfSessionStateBuilderExt, PhysicalOptimizerRules,
    cudf_native_physical_optimizer_rules, cudf_native_physical_optimizer_rules_with_rule,
};
pub use tpcds::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_PARQUET_DIR_ENV,
    CUDF_TPCDS_SCALE_FACTOR_ENV, TPCDS_QUERIES, TPCDS_QUERY_COUNT, TPCDS_TABLES, TpcdsDatasetMetadata, TpcdsQuery,
    TpcdsTableFileMetadata, TpcdsTableLayout, TpcdsTableMetadata, TpcdsWorkloadError, TpcdsWorkloadResult,
    describe_tpcds_dataset, describe_tpcds_dataset_with_content_fingerprint, discover_tpcds_table_layout,
    external_tpcds_dataset_metadata, prepare_tpcds_report_dataset, tiny_tpcds_dataset_metadata,
    tpcds_datafusion_execution_workload_items, tpcds_datafusion_execution_workload_items_with_optimizer_config,
    tpcds_datafusion_planning_workload_items, tpcds_datafusion_planning_workload_items_with_optimizer_config,
    tpcds_fallback_pressure_workload_items, tpcds_native_optimizer_config, tpcds_query_ids,
    tpcds_report_session_config, write_tiny_tpcds_parquet_dataset,
};
pub use tpch::{
    AD_HOC_OLAP_QUERIES, AD_HOC_OLAP_QUERY_COUNT, AdHocOlapCapabilityFirstAdmission, AdHocOlapCurrentStrictAdmission,
    AdHocOlapLoweringExpectation, AdHocOlapQuery, AdHocOlapWorkloadClass, CUDF_TPCH_CONTENT_FINGERPRINT_ENV,
    CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV, CUDF_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES,
    TPCH_QUERY_COUNT, TPCH_TABLES, TpchDatasetMetadata, TpchQuery, TpchTableFileMetadata, TpchTableLayout,
    TpchTableMetadata, TpchWorkloadError, TpchWorkloadResult, ad_hoc_olap_datafusion_planning_workload_items,
    ad_hoc_olap_datafusion_planning_workload_items_with_optimizer_config, ad_hoc_olap_fallback_pressure_workload_items,
    ad_hoc_olap_query_ids, describe_tpch_dataset, describe_tpch_dataset_with_content_fingerprint,
    discover_tpch_table_layout, external_tpch_dataset_metadata, prepare_tpch_report_dataset,
    register_tpch_table_layout, tiny_tpch_dataset_metadata, tpch_datafusion_execution_workload_items,
    tpch_datafusion_execution_workload_items_with_optimizer_config, tpch_datafusion_planning_workload_items,
    tpch_datafusion_planning_workload_items_with_optimizer_config, tpch_fallback_pressure_workload_items,
    tpch_native_optimizer_config, tpch_query_ids, tpch_report_session_config,
    tpch_scalar_fallback_pressure_workload_items, tpch_tuned_cpu_session_config, write_tiny_tpch_parquet_dataset,
};
pub use workload::{tpcds, tpch};
