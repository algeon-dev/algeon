use super::{
    BoundedExecutionAttempt, GpuDataFrame, Instant, NodeId, ResidentGpuFrame, Result, cudf_query_error,
    observe_device_memory_bytes,
};

/// Concatenates chunk frames, in order, into one resident frame.
///
/// A single chunk passes through. Otherwise one cuDF concat allocates the full output
/// while every input stays alive until the call returns; the peak is the inputs plus the
/// output. On error the inputs drop.
pub(super) fn concatenate_join_source_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    mut chunks: Vec<(GpuDataFrame, u64)>,
) -> Result<ResidentGpuFrame> {
    if chunks.len() == 1 {
        let (frame, device_bytes) = chunks.pop().expect("single chunk checked");
        return Ok(ResidentGpuFrame::with_device_bytes(frame, device_bytes));
    }
    let mut frames = Vec::with_capacity(chunks.len());
    for (frame, _) in chunks {
        frames.push(frame.into_frame());
    }
    let chunk_count = frames.len();
    let first_frame = frames.remove(0);
    let rest_frame_refs = frames.iter().collect::<Vec<_>>();
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let timer = Instant::now();
    let output = match first_frame.on(overrides).concat_many(&rest_frame_refs) {
        Ok(output) => output,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.join.source_concat",
                "failed to concatenate bounded join leaf local parquet source chunks",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("chunk_count", chunk_count));
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());

    let output_frame = GpuDataFrame::new(output);
    let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics())?;
    Ok(ResidentGpuFrame::with_device_bytes(output_frame, output_bytes))
}
