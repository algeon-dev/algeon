use std::sync::Arc;

use arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use cudf::{DataFrame, ExecutionPlacement, frame::dataframe::GpuRecordBatch};

use crate::{
    error::{QueryError, Result},
    metrics::QueryMetrics,
    plan::ResultSink,
    runtime::{DevicePlacement, ExecutionContext},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    GpuResident,
    MaterializeToHost,
}

impl ExecutionMode {
    pub fn for_sink(sink: &ResultSink) -> Self {
        match sink {
            ResultSink::GpuResident { .. } => Self::GpuResident,
            ResultSink::HostArrow { .. } => Self::MaterializeToHost,
        }
    }
}

#[derive(Clone)]
pub struct GpuDataFrame {
    frame: DataFrame,
    placement: ExecutionPlacement,
}

impl GpuDataFrame {
    pub fn new(frame: DataFrame) -> Self {
        let placement = frame.placement();
        Self { frame, placement }
    }

    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    pub fn into_frame(self) -> DataFrame {
        debug_assert_eq!(self.placement, self.frame.placement());
        self.frame
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.placement
    }

    pub fn num_rows(&self) -> usize {
        self.frame.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.frame.num_columns()
    }
}

#[derive(Clone)]
pub struct GpuResultHandle {
    batch: GpuRecordBatch,
    runtime: Arc<cudf::Runtime>,
    device_placement: DevicePlacement,
}

impl GpuResultHandle {
    pub fn new(context: &ExecutionContext, frame: GpuDataFrame, schema: SchemaRef) -> Result<Self> {
        let batch = GpuRecordBatch::new(frame.into_frame(), schema).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.gpu_result_handle",
                "failed to bind native GPU result to its Arrow schema contract",
                error,
            )
        })?;
        Ok(Self { batch, runtime: context.shared_runtime(), device_placement: context.device_placement() })
    }

    pub fn frame(&self) -> &DataFrame {
        self.batch.frame()
    }

    pub fn into_gpu_frame(self) -> GpuDataFrame {
        GpuDataFrame::new(self.batch.into_frame())
    }

    pub fn schema(&self) -> &SchemaRef {
        self.batch.schema()
    }

    pub fn runtime(&self) -> &Arc<cudf::Runtime> {
        &self.runtime
    }

    pub fn device_placement(&self) -> DevicePlacement {
        self.device_placement
    }

    pub fn placement(&self) -> ExecutionPlacement {
        self.batch.placement()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn num_columns(&self) -> usize {
        self.batch.num_columns()
    }

    pub fn to_host_record_batch(&self) -> Result<RecordBatch> {
        self.batch.to_host_record_batch().map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.host_arrow.materialize",
                "failed to materialize native cudf output to host Arrow",
                error,
            )
        })
    }
}

pub enum QueryOutput {
    GpuResident(GpuResultHandle),
    HostArrow(RecordBatch),
}

impl QueryOutput {
    pub fn mode(&self) -> ExecutionMode {
        match self {
            Self::GpuResident(_) => ExecutionMode::GpuResident,
            Self::HostArrow(_) => ExecutionMode::MaterializeToHost,
        }
    }

    pub fn as_host_arrow(&self) -> Option<&RecordBatch> {
        match self {
            Self::HostArrow(batch) => Some(batch),
            Self::GpuResident(_) => None,
        }
    }

    pub fn as_gpu_resident(&self) -> Option<&GpuResultHandle> {
        match self {
            Self::GpuResident(frame) => Some(frame),
            Self::HostArrow(_) => None,
        }
    }
}

pub struct QueryExecutionResult {
    output: QueryOutput,
    metrics: QueryMetrics,
}

impl QueryExecutionResult {
    pub fn new(output: QueryOutput, metrics: QueryMetrics) -> Self {
        Self { output, metrics }
    }

    pub fn output(&self) -> &QueryOutput {
        &self.output
    }

    pub fn metrics(&self) -> &QueryMetrics {
        &self.metrics
    }

    pub fn into_parts(self) -> (QueryOutput, QueryMetrics) {
        (self.output, self.metrics)
    }
}
