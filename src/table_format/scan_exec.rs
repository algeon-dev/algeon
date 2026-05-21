#![deny(missing_docs)]

use std::{any::Any, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::context::TaskContext,
    physical_expr::{EquivalenceProperties, Partitioning},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
    },
};

use crate::{
    Error, ErrorCode,
    table_format::{TableFormat, TableFormatScanDiagnostics, TableFormatScanPlan},
};

const EXECUTE_OPERATION: &str = "datafusion_cudf.table_format.scan_exec.execute";

/// DataFusion physical scan leaf for adapter-owned table-format scan facts.
///
/// This node is a planning boundary. It deliberately does not read table data
/// through DataFusion Arrow readers; native lowering must consume the
/// structured [`TableFormatScanPlan`] instead.
#[derive(Debug, Clone)]
pub struct CudfTableFormatScanExec {
    scan_plan: TableFormatScanPlan,
    cache: Arc<PlanProperties>,
}

impl CudfTableFormatScanExec {
    /// Creates a table-format planning leaf with conservative DataFusion plan properties.
    pub fn new(scan_plan: TableFormatScanPlan, schema: SchemaRef) -> Self {
        Self { cache: conservative_properties(schema), scan_plan }
    }

    /// Returns the metadata-only table-format scan plan.
    pub fn scan_plan(&self) -> &TableFormatScanPlan {
        &self.scan_plan
    }

    /// Returns the table format that produced the scan facts.
    pub fn table_format(&self) -> &TableFormat {
        &self.scan_plan.format
    }

    /// Returns credential-free diagnostics derived from the scan plan.
    pub fn diagnostics(&self) -> TableFormatScanDiagnostics {
        self.scan_plan.diagnostics()
    }
}

impl DisplayAs for CudfTableFormatScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let diagnostics = self.diagnostics();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CudfTableFormatScanExec: table_format={}, snapshot_id={}, catalog_type={}, planned_data_file_count={}, planned_delete_file_count={}, admitted_native_source_file_count={}, native_source_read_capability={}, direct_device_read_disposition={}",
                diagnostics.table_format,
                diagnostics.snapshot_id.as_deref().unwrap_or_default(),
                diagnostics.catalog_type.as_deref().unwrap_or_default(),
                diagnostics.planned_data_file_count,
                diagnostics.planned_delete_file_count,
                diagnostics.admitted_native_source_file_count,
                diagnostics.native_source_read_capability,
                diagnostics.direct_device_read_disposition.code(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "table_format={}", diagnostics.table_format)?;
                writeln!(f, "snapshot_id={}", diagnostics.snapshot_id.as_deref().unwrap_or_default())?;
                writeln!(f, "catalog_type={}", diagnostics.catalog_type.as_deref().unwrap_or_default())?;
                writeln!(f, "planned_data_file_count={}", diagnostics.planned_data_file_count)?;
                writeln!(f, "planned_delete_file_count={}", diagnostics.planned_delete_file_count)?;
                writeln!(f, "admitted_native_source_file_count={}", diagnostics.admitted_native_source_file_count)?;
                writeln!(f, "native_source_read_capability={}", diagnostics.native_source_read_capability)?;
                writeln!(f, "direct_device_read_disposition={}", diagnostics.direct_device_read_disposition.code())
            },
        }
    }
}

impl ExecutionPlan for CudfTableFormatScanExec {
    fn name(&self) -> &str {
        "CudfTableFormatScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        Vec::new()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return Err(DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidChildCount,
                    "datafusion_cudf.table_format.scan_exec.with_new_children",
                    "table-format scan exec expected no children",
                )
                .with_fact("actual_child_count", children.len()),
            ));
        }
        Ok(self)
    }

    fn execute(&self, partition: usize, _context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        Err(DataFusionError::from(
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                EXECUTE_OPERATION,
                "table-format scan exec is a native planning leaf and does not execute a CPU Arrow reader",
            )
            .with_fact("partition", partition)
            .with_fact("table_format", self.scan_plan.format.code())
            .with_fact("cpu_arrow_reader_fallback", false),
        ))
    }
}

fn conservative_properties(schema: SchemaRef) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(1),
        EmissionType::Final,
        Boundedness::Bounded,
    ))
}
