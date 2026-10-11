//! TPC-DS parity report contracts.
//!
//! This module compares CPU DataFusion output, native GPU output, and optional
//! DuckDB oracle output for the same TPC-DS query. Column order comes from
//! [`NATIVE_TPCDS_PARITY_TSV_HEADER`], missing optional values serialize as `none`,
//! and status enums serialize through their `as_str` methods.
//!
//! Output fingerprints use a stable FNV-1a hash over schema metadata and
//! canonicalized row values. They are diagnostic identifiers. They are not
//! cryptographic hashes. Unordered comparisons sort canonical rows before
//! comparing values; ordered comparisons preserve batch row order.

use arrow::{
    array::{
        Array, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array, Decimal128Array, Decimal256Array,
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        StringArray, StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Schema, SchemaRef, TimeUnit, i256},
    record_batch::RecordBatch,
};
use datafusion::physical_plan::{ExecutionPlan, metrics::MetricsSet};
use query_planning::mode::StatisticalAggregateExecutionMode;

use super::{
    CandidateOutcomeKind, FinalPlanDisposition, NativeDataFusionExecutionMetrics, NativeDataFusionRuntimeOutcome,
    NativePlanningFailure, WorkloadDatasetMetadata,
    diagnostics::{post_native_plan_coverage, unsupported_category},
    tsv::{TsvRow, stable_tsv_field, write_workload_dataset_metadata},
};
use crate::planner::NativePlanningDiagnostics;

mod canonical;
mod compare;
mod fingerprint;
mod report;

// Private `use` items are visible to this module and its descendants. The
// submodules import these names with `use super::*;`.
use canonical::{CanonicalField, collect_canonical_rows, compare_canonical_rows};
use compare::{ParityCompareOptions, compare_record_batches_with_options};
pub use compare::{
    ParityComparisonResult, compare_record_batches, sql_has_top_level_order_by,
    sql_top_level_order_by_covers_output_columns,
};
use fingerprint::{fingerprint_batches, fingerprint_rows, fingerprint_schema_only};
pub use report::{
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus,
    NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityNativePlanningOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityReport, NativeTpcdsParityRow,
    NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityFloatTolerance, ParityStageStatus, ParityStatus,
};

/// TPC-DS parity TSV column names in wire/report order.
pub(crate) const NATIVE_TPCDS_PARITY_TSV_COLUMNS: &[&str] = &[
    "query_id",
    "cpu_planning_status",
    "cpu_runtime_status",
    "native_planning_outcome",
    "native_reason_code",
    "native_unsupported_category",
    "post_native_execution_path",
    "native_plan_fragment_count",
    "host_datafusion_boundary_count",
    "native_runtime_started",
    "native_runtime_outcome",
    "parity_status",
    "parity_detail",
    "cpu_output_rows",
    "native_output_rows",
    "cpu_output_batches",
    "native_output_batches",
    "cpu_output_fingerprint",
    "native_output_fingerprint",
    "comparison_mode",
    "duckdb_output_status",
    "duckdb_cpu_parity_status",
    "duckdb_cpu_parity_detail",
    "duckdb_output_rows",
    "duckdb_output_batches",
    "duckdb_output_fingerprint",
    "native_retry_count",
    "native_runtime_failure",
];
/// TPC-DS parity TSV header.
pub const NATIVE_TPCDS_PARITY_TSV_HEADER: &str = "query_id\tcpu_planning_status\tcpu_runtime_status\tnative_planning_outcome\tnative_reason_code\tnative_unsupported_category\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\tnative_runtime_started\tnative_runtime_outcome\tparity_status\tparity_detail\tcpu_output_rows\tnative_output_rows\tcpu_output_batches\tnative_output_batches\tcpu_output_fingerprint\tnative_output_fingerprint\tcomparison_mode\tduckdb_output_status\tduckdb_cpu_parity_status\tduckdb_cpu_parity_detail\tduckdb_output_rows\tduckdb_output_batches\tduckdb_output_fingerprint\tnative_retry_count\tnative_runtime_failure";
