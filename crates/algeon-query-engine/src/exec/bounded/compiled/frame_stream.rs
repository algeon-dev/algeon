//! Resident frame streams: the `FrameStream` payload form of a compiled
//! resource.
//!
//! A stream lets a consumer pull frames one at a time, so the producer's
//! whole output need not be resident at once. Frames already yielded belong to the
//! consumer; storage the producer still retains (unread inputs, merge state,
//! finalized frames not yet pulled) belongs to the producer until its
//! `release` runs. Release needs the execution attempt, so a stream payload
//! is never released by plain `Drop`: `CompiledGpuTaskExecutor::release_resource`
//! calls `release_all` for every stream the scheduler returns.

use std::collections::VecDeque;

use query_planning::{
    dynamic_filter::{DynamicFilter, DynamicFilterRoute},
    error::Result,
    plan::{FilterNode, NodeId, ProjectionNode},
};
use query_runtime::runtime::resident::GpuDataFrame;

use crate::exec::{
    ExecutionMetrics,
    bounded::{BoundedExecutionAttempt, ResidentGpuFrame, execute_bounded_projection, run_budgeted},
    operators::execute_filter_with_budget,
};

/// Frames a compiled consumer pulls in order.
///
/// Eager payloads hold every frame in `frames`. Lazy payloads hold a
/// `producer` and may also hold a prefix of frames; `next_frame` drains the
/// prefix before asking the producer. Row and byte figures are hints taken at
/// construction and are not updated as frames are pulled.
pub(in crate::exec::bounded) struct ResidentFrameStreamPayload {
    /// Advisory device bytes for one yielded frame. Eager payloads carry
    /// their largest observed frame; producers derive the hint from their
    /// inputs, and `None` when their byte model cannot bound the output.
    /// Join probe sizing and `from_producer` chains read it.
    max_frame_bytes_hint: Option<u64>,
    /// Aggregate byte hint for the whole stream, `None` when unknown. Lazy
    /// transformations retain their input's hint until output frames are
    /// observed; this is not an exact final-output allocation measurement.
    total_bytes_hint: Option<u64>,
    aggregate_num_rows: u64,
    num_columns: usize,
    frames: VecDeque<ResidentGpuFrame>,
    producer: Option<ResidentFrameStreamProducer>,
}

/// Lazy producer behind a resident frame stream payload.
///
/// A yielded frame belongs to the caller. Storage the source still retains,
/// such as unread inputs, build state, or pending outputs, belongs to the
/// source until [`release`](Self::release) runs.
pub(in crate::exec::bounded) trait ResidentFrameStreamSource: Send {
    /// Produces the next frame.
    ///
    /// `Some(Ok(frame))` transfers the frame and its device bytes to the
    /// caller. `Some(Err(_))` reports a failure while the source may still
    /// retain storage. `None` reports exhaustion; some sources release their
    /// retained state before returning `None`, but `None` alone does not mean
    /// the source has released it.
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>>;

    /// Releases storage the source still retains, within the execution attempt
    /// that owns it.
    ///
    /// The owning payload calls this after draining the stream, when a
    /// consumer stops early, and on error paths, so it must be idempotent,
    /// including after the source released itself at exhaustion. Dropping a
    /// source without this call runs only ordinary `Drop` for its fields;
    /// release work that needs the execution attempt does not happen. The
    /// default retains nothing.
    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {}

    /// Creates an independent cursor sharing retained input storage. Output
    /// transformations may be replayed instead of retaining finalized frames.
    fn alias(&self) -> Option<Box<dyn ResidentFrameStreamSource>> {
        None
    }

    /// Retained state carrying the stream's complete group-key set before any
    /// per-frame finalization runs. Producers answer only when the retained
    /// state already names the operator's output group-key columns; the
    /// aggregate columns of those frames are still merge state.
    fn complete_group_key_state(&self) -> Option<Vec<GpuDataFrame>> {
        None
    }
}

/// Wrapper that lets the payload release its source once by value
/// while `next_frame` keeps borrowing it.
struct ResidentFrameStreamProducer {
    inner: Option<Box<dyn ResidentFrameStreamSource>>,
}

impl ResidentFrameStreamProducer {
    fn new(inner: Box<dyn ResidentFrameStreamSource>) -> Self {
        Self { inner: Some(inner) }
    }

    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        self.inner.as_mut()?.next_frame(execution)
    }

    fn release(mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if let Some(mut inner) = self.inner.take() {
            inner.release(execution);
        }
    }
}

impl ResidentFrameStreamPayload {
    pub(in crate::exec::bounded) fn new(frames: Vec<ResidentGpuFrame>) -> Self {
        let total_bytes_hint = frames.iter().fold(0u64, |total, frame| total.saturating_add(frame.device_bytes()));
        let max_frame_bytes_hint = frames.iter().fold(0u64, |max_bytes, frame| max_bytes.max(frame.device_bytes()));
        let aggregate_num_rows =
            frames.iter().fold(0u64, |total, frame| total.saturating_add(frame.frame.num_rows() as u64));
        let num_columns = frames.first().map_or(0, |frame| frame.frame.num_columns());
        Self {
            max_frame_bytes_hint: Some(max_frame_bytes_hint),
            total_bytes_hint: Some(total_bytes_hint),
            aggregate_num_rows,
            num_columns,
            frames: frames.into(),
            producer: None,
        }
    }

    pub(in crate::exec::bounded) fn from_producer(
        max_frame_bytes_hint: Option<u64>,
        total_bytes_hint: Option<u64>,
        aggregate_num_rows: u64,
        num_columns: usize,
        source: Box<dyn ResidentFrameStreamSource>,
    ) -> Self {
        Self {
            max_frame_bytes_hint,
            total_bytes_hint,
            aggregate_num_rows,
            num_columns,
            frames: VecDeque::new(),
            producer: Some(ResidentFrameStreamProducer::new(source)),
        }
    }

    pub(in crate::exec::bounded) fn max_frame_bytes_hint(&self) -> Option<u64> {
        self.max_frame_bytes_hint
    }

    pub(in crate::exec::bounded) fn total_bytes_hint(&self) -> Option<u64> {
        self.total_bytes_hint
    }

    pub(in crate::exec::bounded) fn num_rows(&self) -> u64 {
        self.aggregate_num_rows
    }

    pub(in crate::exec::bounded) fn num_columns(&self) -> usize {
        self.num_columns
    }

    /// The stream's complete group-key set, readable before any frame is
    /// pulled. Eager payloads answer with their already finalized frames; lazy
    /// payloads answer from producer-retained state, so a slow consumer never
    /// forces the whole output resident just to publish a key snapshot.
    pub(in crate::exec::bounded) fn complete_group_key_state(&self) -> Option<Vec<GpuDataFrame>> {
        match &self.producer {
            Some(producer) => producer.inner.as_ref()?.complete_group_key_state(),
            None => Some(self.frames.iter().map(|frame| frame.frame.clone()).collect()),
        }
    }

    /// An independent cursor over the same content for another consumer.
    ///
    /// An eager alias clones each frame handle, which shares the device
    /// buffers; the copied `device_bytes` therefore counts the same storage
    /// twice as a hint. A lazy alias asks the source for its own cursor and is
    /// `None` when the source cannot replay.
    pub(super) fn try_alias(&self) -> Option<Self> {
        if let Some(producer) = &self.producer {
            return Some(Self::from_producer(
                self.max_frame_bytes_hint,
                self.total_bytes_hint,
                self.aggregate_num_rows,
                self.num_columns,
                producer.inner.as_ref()?.alias()?,
            ));
        }
        let frames = self
            .frames
            .iter()
            .map(|frame| ResidentGpuFrame { frame: frame.frame.clone(), device_bytes: frame.device_bytes() })
            .collect::<Vec<_>>();
        Some(Self::new(frames))
    }

    /// Whether frames are still produced on demand rather than all resident.
    pub(in crate::exec::bounded) fn is_lazy(&self) -> bool {
        self.producer.is_some()
    }

    /// `true` only when nothing can be pulled; a lazy stream with no frames
    /// yet is not empty.
    pub(in crate::exec::bounded) fn is_empty(&self) -> bool {
        self.frames.is_empty() && self.producer.is_none()
    }

    /// Frames already resident in the payload. Frames a producer has not
    /// yielded yet are not counted.
    pub(in crate::exec::bounded) fn len(&self) -> usize {
        self.frames.len()
    }

    pub(in crate::exec::bounded) fn next_frame(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Option<Result<ResidentGpuFrame>> {
        if let Some(frame) = self.frames.pop_front() {
            return Some(Ok(frame));
        }
        self.producer.as_mut()?.next_frame(execution)
    }

    pub(in crate::exec::bounded) fn collect_remaining_frames_owned(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let mut accumulated = Vec::new();
        loop {
            match self.next_frame(execution) {
                Some(Ok(frame)) => accumulated.push(frame),
                Some(Err(error)) => return Err(error),
                None => return Ok(accumulated),
            }
        }
    }

    /// Drains the stream into owned frames and releases the producer on both
    /// the success and the error path, so a failed drain leaves no retained
    /// producer storage behind.
    pub(in crate::exec::bounded) fn into_remaining_frames_owned(
        mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let result = self.collect_remaining_frames_owned(execution);
        self.release_producer(execution);
        result
    }

    /// Drops the resident frames. Their device memory is released by the
    /// frame handles' own `Drop`, so no execution attempt is needed.
    pub(in crate::exec::bounded) fn release_remaining_frames(&mut self) {
        self.frames.clear();
    }

    /// Releases producer-retained storage. Idempotent; later calls find no
    /// producer.
    pub(in crate::exec::bounded) fn release_producer(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        let Some(producer) = self.producer.take() else { return };
        producer.release(execution);
    }

    /// Releases everything the payload still holds. This is the release hook
    /// the executor runs for a stream the scheduler returns.
    pub(in crate::exec::bounded) fn release_all(mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.release_remaining_frames();
        self.release_producer(execution);
    }
}

#[derive(Clone)]
enum UnaryMappedResidentFrameStreamOperator {
    Filter(FilterNode),
    /// A filter whose predicate came from a dynamic filter route; executes
    /// like `Filter` and additionally reports post-decode row counts for the
    /// route.
    DynamicFilter {
        filter: FilterNode,
        route: DynamicFilterRoute,
        dynamic_filter: DynamicFilter,
    },
    Projection(ProjectionNode),
}

/// Lazy source that applies one filter or projection to each frame of an
/// input stream as it is pulled, so the operator output is never fully
/// resident unless a consumer collects it.
struct UnaryMappedResidentFrameStreamSource {
    node_id: NodeId,
    /// `None` after release. Taken by value so the input's own producer is
    /// released through the attempt; a plain drop would skip that release.
    input: Option<ResidentFrameStreamPayload>,
    operator: UnaryMappedResidentFrameStreamOperator,
}

impl UnaryMappedResidentFrameStreamSource {
    fn from_filter(node_id: NodeId, input: ResidentFrameStreamPayload, filter: FilterNode) -> Self {
        Self { node_id, input: Some(input), operator: UnaryMappedResidentFrameStreamOperator::Filter(filter) }
    }

    fn from_dynamic_filter(
        node_id: NodeId,
        input: ResidentFrameStreamPayload,
        filter: FilterNode,
        route: DynamicFilterRoute,
        dynamic_filter: DynamicFilter,
    ) -> Self {
        Self {
            node_id,
            input: Some(input),
            operator: UnaryMappedResidentFrameStreamOperator::DynamicFilter { filter, route, dynamic_filter },
        }
    }

    fn from_projection(node_id: NodeId, input: ResidentFrameStreamPayload, projection: ProjectionNode) -> Self {
        Self { node_id, input: Some(input), operator: UnaryMappedResidentFrameStreamOperator::Projection(projection) }
    }

    fn execute(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        input: ResidentGpuFrame,
    ) -> Result<ResidentGpuFrame> {
        match &self.operator {
            UnaryMappedResidentFrameStreamOperator::Filter(filter)
            | UnaryMappedResidentFrameStreamOperator::DynamicFilter { filter, .. } => run_budgeted(
                execution,
                self.node_id,
                "compiled_filter",
                input.frame,
                |context, input, token, metrics| execute_filter_with_budget(context, filter, input, token, metrics),
            ),
            UnaryMappedResidentFrameStreamOperator::Projection(projection) => {
                execute_bounded_projection(execution, self.node_id, "compiled_projection", projection, input)
            },
        }
    }

    fn record_metrics(&self, input_rows: u64, output: &ResidentGpuFrame, metrics: &ExecutionMetrics) {
        if let UnaryMappedResidentFrameStreamOperator::DynamicFilter { route, dynamic_filter, .. } = &self.operator {
            metrics.record_dynamic_filter_post_decode_rows(
                route,
                dynamic_filter,
                Some(input_rows),
                Some(output.frame.num_rows() as u64),
            );
        }
    }
}

impl ResidentFrameStreamSource for UnaryMappedResidentFrameStreamSource {
    fn alias(&self) -> Option<Box<dyn ResidentFrameStreamSource>> {
        Some(Box::new(Self {
            node_id: self.node_id,
            input: Some(self.input.as_ref()?.try_alias()?),
            operator: self.operator.clone(),
        }))
    }

    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let input = self.input.as_mut()?;
        match input.next_frame(execution) {
            Some(Ok(frame)) => {
                let input_rows = frame.frame.num_rows() as u64;
                Some(self.execute(execution, frame).inspect(|output| {
                    self.record_metrics(input_rows, output, execution.metrics());
                }))
            },
            Some(Err(error)) => Some(Err(error)),
            None => {
                // Exhaustion releases the input at once so its retained
                // storage does not wait for the owning payload's release.
                self.release(execution);
                None
            },
        }
    }

    fn release(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if let Some(input) = self.input.take() {
            input.release_all(execution);
        }
    }
}

/// Wraps `stream` in a lazy filter. The input's row and byte hints are kept:
/// a filter can only shrink its input, so they remain upper bounds for the
/// consumers that size work from them.
pub(super) fn map_filter_resident_frame_stream(
    node_id: NodeId,
    stream: ResidentFrameStreamPayload,
    filter: FilterNode,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = stream.max_frame_bytes_hint();
    let total_bytes_hint = stream.total_bytes_hint();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = stream.num_columns();
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(UnaryMappedResidentFrameStreamSource::from_filter(node_id, stream, filter)),
    )
}

pub(super) fn map_dynamic_filter_resident_frame_stream(
    node_id: NodeId,
    stream: ResidentFrameStreamPayload,
    filter: FilterNode,
    route: DynamicFilterRoute,
    dynamic_filter: DynamicFilter,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = stream.max_frame_bytes_hint();
    let total_bytes_hint = stream.total_bytes_hint();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = stream.num_columns();
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(UnaryMappedResidentFrameStreamSource::from_dynamic_filter(
            node_id,
            stream,
            filter,
            route,
            dynamic_filter,
        )),
    )
}

/// Wraps `stream` in a lazy projection. Row hints carry over unchanged; the
/// column count comes from the projection schema, while the byte hints stay
/// the input's because the projected width is not modelled here.
pub(super) fn map_projection_resident_frame_stream(
    node_id: NodeId,
    stream: ResidentFrameStreamPayload,
    projection: ProjectionNode,
) -> ResidentFrameStreamPayload {
    let max_frame_bytes_hint = stream.max_frame_bytes_hint();
    let total_bytes_hint = stream.total_bytes_hint();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = projection.output_schema().fields().len();
    ResidentFrameStreamPayload::from_producer(
        max_frame_bytes_hint,
        total_bytes_hint,
        aggregate_num_rows,
        num_columns,
        Box::new(UnaryMappedResidentFrameStreamSource::from_projection(node_id, stream, projection)),
    )
}
