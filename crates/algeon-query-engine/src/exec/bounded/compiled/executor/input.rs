use query_planning::{
    error::{QueryError, Result},
    pipeline::{PipelineTask, ResidentDataForm, ResidentResourceId, ResourceSharing},
};

use super::{CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineState, CompiledTaskExecutor};
use crate::exec::bounded::{
    ResidentGpuFrame,
    compiled::{helpers::invalid, resource::CompiledGpuResourcePayload},
    materialize_bounded_join_leaf_local_parquet_source_stream,
};

impl CompiledGpuTaskExecutor<'_> {
    /// Hands `task` one input under the resource's compiled sharing contract.
    ///
    /// The sole consumer of a resource takes the stored payload. A consumer of
    /// a shared resource gets its own view and leaves the payload stored for
    /// the others: a frame snapshot over the same device storage, or a stream
    /// alias over the producer's retained storage. A deferred source read is
    /// copied for a consumer only when the contract declares independent
    /// cursors, because each copy reads the source again.
    pub(super) fn take_task_input(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
        resource: ResidentResourceId,
    ) -> Result<CompiledGpuResourcePayload> {
        if state.can_consume_task_input_resource(task, resource) {
            return Ok(state.consume_task_input_resource(task, resource)?.payload);
        }
        self.apply_ready_dynamic_filters_to_shared_source_resource(state, resource)?;
        let stored = state.resource_payload(resource)?;
        let sharing = self.resource_sharing(resource);
        match &stored.payload {
            CompiledGpuResourcePayload::ResidentFrame { frame, applied_limit } => {
                Ok(CompiledGpuResourcePayload::ResidentFrame {
                    frame: Box::new(ResidentGpuFrame::with_device_bytes(frame.frame.clone(), frame.device_bytes())),
                    applied_limit: *applied_limit,
                })
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) => {
                stream.try_alias().map(CompiledGpuResourcePayload::ResidentFrameStream).ok_or_else(|| {
                    shared_input_error(
                        "compiled shared frame stream cannot open a consumer cursor",
                        task,
                        resource,
                        stored,
                    )
                })
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream)
                if sharing == ResourceSharing::IndependentCursorStream =>
            {
                Ok(CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream.clone()))
            },
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(_)
            | CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(_) => Err(shared_input_error(
                "compiled shared resource holds a deferred source read its consumers may not replay",
                task,
                resource,
                stored,
            )),
        }
    }

    /// Takes an input of a task that reads one complete resident frame.
    pub(super) fn take_frame_input(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<CompiledGpuResource>,
        resource: ResidentResourceId,
    ) -> Result<ResidentGpuFrame> {
        match self.take_task_input(task, state, resource)? {
            CompiledGpuResourcePayload::ResidentFrame { frame, .. } => Ok(*frame),
            payload => Err(self.reject_task_input(task, resource, payload)?),
        }
    }

    /// Releases an input whose form the task cannot read and returns the
    /// contract error. The compiled contract gives every consumer a form it
    /// accepts, so reaching this means a producer broke its declared form.
    pub(super) fn reject_task_input(
        &mut self,
        task: &PipelineTask,
        resource: ResidentResourceId,
        payload: CompiledGpuResourcePayload,
    ) -> Result<QueryError> {
        let payload = CompiledGpuResource { payload };
        let error = invalid("compiled task received an input form it does not accept")
            .with_fact("compiled_task_id", task.id())
            .with_fact("pipeline_task_kind", task.kind().as_str())
            .with_fact("resident_resource_id", resource)
            .with_fact("payload_form", payload.form_name());
        <Self as CompiledTaskExecutor>::release_resource(self, resource, payload)?;
        Ok(error)
    }

    /// Settles a task output into the form its compiled resource contract
    /// declares before the output is stored.
    ///
    /// A deferred source read the contract does not let consumers take as is
    /// is read once here into one resident frame, and a lazy stream that
    /// several consumers share immutably is collected once into its chunks.
    /// Neither leaves a consumer to replay the producer.
    pub(super) fn settle_task_output(
        &mut self,
        task: &PipelineTask,
        resource: ResidentResourceId,
        output: CompiledGpuResource,
    ) -> Result<CompiledGpuResource> {
        let form = self.resource_data_form(resource);
        let sharing = self.resource_sharing(resource);
        match output.payload {
            CompiledGpuResourcePayload::JoinLeafLocalParquetSource(stream)
                if form != ResidentDataForm::SourceStream =>
            {
                let stream = self.apply_dynamic_filters_to_source_stream(*stream)?;
                materialize_bounded_join_leaf_local_parquet_source_stream(&mut self.execution, stream)
                    .map(CompiledGpuResource::new)
            },
            CompiledGpuResourcePayload::PendingJoinLeafLocalParquetSource(prefetch)
                if form != ResidentDataForm::SourceStream || sharing != ResourceSharing::SingleConsume =>
            {
                prefetch.finish(&mut self.execution).map(CompiledGpuResource::new)
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream) if form == ResidentDataForm::Frame => {
                let output = CompiledGpuResource::frame_stream_payload(stream);
                let error = invalid("compiled task produced a frame stream for a frame resource")
                    .with_fact("compiled_task_id", task.id())
                    .with_fact("pipeline_task_kind", task.kind().as_str())
                    .with_fact("resident_resource_id", resource);
                <Self as CompiledTaskExecutor>::release_resource(self, resource, output)?;
                Err(error)
            },
            CompiledGpuResourcePayload::ResidentFrameStream(stream)
                if sharing == ResourceSharing::SharedImmutableFrame && stream.is_lazy() =>
            {
                stream.into_remaining_frames_owned(&mut self.execution).map(CompiledGpuResource::frame_stream)
            },
            payload => Ok(CompiledGpuResource { payload }),
        }
    }
}

fn shared_input_error(
    message: &'static str,
    task: &PipelineTask,
    resource: ResidentResourceId,
    stored: &CompiledGpuResource,
) -> QueryError {
    invalid(message)
        .with_fact("compiled_task_id", task.id())
        .with_fact("pipeline_task_kind", task.kind().as_str())
        .with_fact("resident_resource_id", resource)
        .with_fact("payload_form", stored.form_name())
}
