//! DataFusion adapter for native cudf-rs GPU execution.

pub mod error;
pub mod native;
pub mod native_exec;
pub mod native_report;
pub mod planner;
pub mod policy;
pub mod report;
pub mod session;
pub mod tpch;
mod tpch_queries;

pub use cudf_query_engine::{GpuLocalParquetSourceCache, GpuLocalParquetSourceCacheKey};
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
    NativePlanningFailure, NativeSourceDiagnosticSummary,
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
pub use tpch::{
    CUDF_TPCH_CONTENT_FINGERPRINT_ENV, CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_PARQUET_DIR_ENV,
    CUDF_TPCH_SCALE_FACTOR_ENV, TPCH_QUERIES, TPCH_QUERY_COUNT, TPCH_TABLES, TpchDatasetMetadata, TpchQuery,
    TpchTableFileMetadata, TpchTableMetadata, TpchWorkloadError, TpchWorkloadResult, describe_tpch_dataset,
    describe_tpch_dataset_with_content_fingerprint, external_tpch_dataset_metadata, prepare_tpch_report_dataset,
    tiny_tpch_dataset_metadata, tpch_datafusion_execution_workload_items, tpch_datafusion_planning_workload_items,
    tpch_fallback_pressure_workload_items, tpch_native_optimizer_config, tpch_query_ids, tpch_report_session_config,
    tpch_scalar_fallback_pressure_workload_items, write_tiny_tpch_parquet_dataset,
};
