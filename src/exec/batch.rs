use arrow::record_batch::RecordBatch;
use cudf::{ExecutionOverrides, ExecutionPlacement, Runtime};
use datafusion::common::{Result as DataFusionResult, utils::memory::get_record_batch_memory_size};

use super::metrics::CudfGpuIslandMetrics;
use crate::error::Error;

#[derive(Debug)]
pub(super) struct CudfBatch {
    frame: cudf::DataFrame,
    placement: ExecutionPlacement,
}

impl CudfBatch {
    pub(super) fn new(frame: cudf::DataFrame) -> Self {
        let placement = frame.placement();
        Self { frame, placement }
    }

    pub(super) fn into_frame(self) -> cudf::DataFrame {
        debug_assert_eq!(self.placement, self.frame.placement());
        self.frame
    }

    pub(super) fn concat_many(
        self,
        ctx: &ExecutionOverrides<'_>,
        other_batches: &[Self],
        metrics: &CudfGpuIslandMetrics,
    ) -> DataFusionResult<Self> {
        if other_batches.is_empty() {
            return Ok(self);
        }

        let other_frames = other_batches.iter().map(|batch| &batch.frame).collect::<Vec<_>>();
        let operator_timer = metrics.cudf_operator_time.timer();
        let frame = self.frame.with_execution(ctx).concat_many(&other_frames).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.input.concat",
                "failed to concatenate uploaded CudfBatches inside GPU island",
                error,
            )
        })?;
        operator_timer.done();
        Ok(Self::new(frame))
    }

    pub(super) fn materialize(self, metrics: &CudfGpuIslandMetrics) -> DataFusionResult<RecordBatch> {
        debug_assert_eq!(self.placement, self.frame.placement());
        let materialization_timer = metrics.record_batch_materialization_time.timer();
        let output = self.frame.to_record_batch().map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.record_batch.materialize",
                "failed to materialize CudfBatch to RecordBatch",
                error,
            )
        })?;
        materialization_timer.done();
        metrics.device_to_host_materializations.add(1);
        metrics.device_to_host_materialized_rows.add(output.num_rows());
        metrics.device_to_host_materialized_bytes.add(get_record_batch_memory_size(&output));
        Ok(output)
    }
}

pub(super) fn upload_batch(
    runtime: &Runtime,
    batch: RecordBatch,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<CudfBatch> {
    let input_rows = batch.num_rows();
    let input_bytes = get_record_batch_memory_size(&batch);
    metrics.input_batches.add(1);
    metrics.input_rows.add(input_rows);
    upload_batch_with_size(runtime, batch, input_rows, input_bytes, metrics)
}

pub(super) fn upload_batch_with_size(
    runtime: &Runtime,
    batch: RecordBatch,
    input_rows: usize,
    input_bytes: usize,
    metrics: &CudfGpuIslandMetrics,
) -> DataFusionResult<CudfBatch> {
    let upload_timer = metrics.record_batch_upload_time.timer();
    let frame = cudf::DataFrame::from_record_batch(runtime, &batch).map_err(|error| {
        Error::from_cudf("datafusion_cudf.record_batch.upload", "failed to upload RecordBatch to cudf", error)
    })?;
    upload_timer.done();
    metrics.host_to_device_uploads.add(1);
    metrics.host_to_device_uploaded_rows.add(input_rows);
    metrics.host_to_device_uploaded_bytes.add(input_bytes);
    Ok(CudfBatch::new(frame))
}
