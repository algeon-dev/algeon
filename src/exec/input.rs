use std::sync::Arc;

use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use super::{normalize::NormalizedGpuIslandPlan, parquet};
use crate::error::Result as AdapterResult;

#[derive(Debug, Clone)]
pub(super) enum GpuIslandInput {
    ChildRecordBatches(Arc<dyn ExecutionPlan>),
    LocalParquetScan(parquet::CudfParquetScan),
}

impl GpuIslandInput {
    pub(super) fn try_from_normalized(plan: &NormalizedGpuIslandPlan) -> AdapterResult<Self> {
        if let Ok(Some(scan)) = parquet::try_local_parquet_scan(plan) {
            return Ok(Self::LocalParquetScan(scan));
        }
        Ok(Self::ChildRecordBatches(Arc::clone(&plan.input)))
    }

    pub(super) fn child(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        match self {
            Self::ChildRecordBatches(input) => Some(input),
            Self::LocalParquetScan(_) => None,
        }
    }

    pub(super) fn output_partition_count(&self) -> usize {
        match self {
            Self::ChildRecordBatches(input) => input.output_partitioning().partition_count(),
            Self::LocalParquetScan(scan) => scan.partition_count(),
        }
    }

    pub(super) fn display_name(&self) -> &'static str {
        match self {
            Self::ChildRecordBatches(_) => "child_record_batches",
            Self::LocalParquetScan(_) => "local_parquet_scan",
        }
    }

    pub(super) fn input_boundary(&self) -> &'static str {
        match self {
            Self::ChildRecordBatches(_) => "host_to_device",
            Self::LocalParquetScan(_) => "cudf_parquet_reader",
        }
    }
}
