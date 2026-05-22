//! TPC-DS parity report contracts.
//!
//! This module compares CPU DataFusion output, native GPU output, and optional
//! external-oracle output for the same TPC-DS query. The TSV surface is stable
//! within [`NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION`]: column order comes from
//! [`NATIVE_TPCDS_PARITY_TSV_HEADER`], missing optional values serialize as
//! `none`, and status enums serialize through their `as_str` methods.
//!
//! Output fingerprints use a stable FNV-1a hash over schema metadata and
//! canonicalized row values. They are diagnostic identifiers, not
//! cryptographic hashes. Unordered comparisons sort canonical rows before
//! comparing values; ordered comparisons preserve batch row order.

#![deny(missing_docs)]

use std::{collections::HashMap, fmt::Write, sync::Arc};

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

use super::{
    NativeDataFusionExecutionMetrics, NativeDataFusionPlanningOutcome, NativeDataFusionPostNativeExecutionPath,
    NativeDataFusionRuntimeOutcome, NativePlanningFailure, WorkloadDatasetMetadata,
    diagnostics::{post_native_plan_coverage, unsupported_category},
    tsv::{stable_tsv_field, write_workload_dataset_metadata},
};
use crate::planner::NativePlanningDiagnostics;

/// TPC-DS parity TSV schema version.
///
/// Bump this when a column is added, removed, renamed, reordered, or given a
/// new meaning that would change downstream parsing.
pub const NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION: u32 = 3;
/// TPC-DS parity TSV header. Column order is stable for schema version 3.
pub const NATIVE_TPCDS_PARITY_TSV_HEADER: &str = "query_id\tcpu_planning_status\tcpu_runtime_status\tnative_planning_outcome\tnative_reason_code\tnative_unsupported_category\tpost_native_execution_path\tnative_plan_fragment_count\thost_datafusion_boundary_count\tnative_runtime_started\tnative_runtime_outcome\tparity_status\tparity_detail\tcpu_output_rows\tnative_output_rows\tcpu_output_batches\tnative_output_batches\tcpu_output_fingerprint\tnative_output_fingerprint\tcomparison_mode\tduckdb_output_status\tduckdb_cpu_parity_status\tduckdb_cpu_parity_detail\tduckdb_output_rows\tduckdb_output_batches\tduckdb_output_fingerprint\texternal-engine_output_status\texternal-engine_cpu_parity_status\texternal-engine_cpu_parity_detail\texternal-engine_output_rows\texternal-engine_output_batches\texternal-engine_output_fingerprint\tnative_reservation_live_peak_bytes\tnative_reservation_denied_count\tnative_retry_count\tnative_runtime_failure";

mod canonical;
mod compare;
mod fingerprint;
mod report;

// Private sibling-visibility imports. Submodules pick up these names via
// `use super::*;` because child modules can see their parent's private uses.
use canonical::{CanonicalField, CanonicalRow, collect_canonical_rows, compare_canonical_rows};
use compare::{ParityCompareOptions, compare_record_batches_with_options};
use fingerprint::{fingerprint_batches, fingerprint_rows, fingerprint_schema_only};

pub use compare::{ParityComparisonResult, compare_record_batches, sql_has_top_level_order_by};
pub use report::{
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputStatus,
    NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityNativePlanningOutcome,
    NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport,
    NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode, ParityFloatTolerance, ParityStageStatus,
    ParityStatus,
};
