use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::{
    dynamic_filter::{DynamicFilter, DynamicFilterRoute},
    error::{QueryError, QueryErrorCode, Result},
    exec::{
        ExecutionMetrics,
        bounded::{BoundedExecutionAttempt, ResidentGpuFrame, execute_bounded_projection, run_budgeted},
        operators::execute_filter_with_budget,
    },
    plan::{FilterNode, NodeId, ProjectionNode},
    runtime::resident::GpuDataFrame,
};

pub(in crate::exec::bounded) struct ResidentFrameStreamPayload {
    /// Advisory device bytes for one yielded frame. Eager payloads carry
    /// their largest observed frame; producers derive the hint from their
    /// inputs. Join probe sizing and `from_producer` chains read it.
    max_frame_bytes_hint: u64,
    /// Aggregate byte hint for the whole stream. Lazy transformations retain
    /// their input's hint until output frames are observed; this is not an
    /// exact final-output allocation measurement.
    total_bytes_hint: u64,
    /// Observed device bytes of the yielded frames that no consumer has taken
    /// yet. Vec-backed payloads start at the sum of their frames' observed
    /// bytes because every frame is already live; producer-backed payloads
    /// start at 0. Storage the producer still retains is not counted here and
    /// is added by `live_observed_device_bytes()`.
    live_observed_device_bytes: Arc<AtomicU64>,
    aggregate_num_rows: u64,
    num_columns: usize,
    frames: VecDeque<ResidentGpuFrame>,
    producer: Option<ResidentFrameStreamProducer>,
}

pub(in crate::exec::bounded) struct YieldedFrame {
    frame: Option<ResidentGpuFrame>,
    release_guard: YieldedFrameReleaseGuard,
}

impl YieldedFrame {
    pub(in crate::exec::bounded) fn yield_from_producer(frame: ResidentGpuFrame, live_counter: Arc<AtomicU64>) -> Self {
        let contributed_bytes = frame.device_bytes();
        Self { frame: Some(frame), release_guard: YieldedFrameReleaseGuard::new(live_counter, contributed_bytes, true) }
    }

    fn from_existing_payload_frame(frame: ResidentGpuFrame, live_counter: Arc<AtomicU64>) -> Self {
        let contributed_bytes = frame.device_bytes();
        Self {
            frame: Some(frame),
            release_guard: YieldedFrameReleaseGuard::new(live_counter, contributed_bytes, false),
        }
    }

    pub(in crate::exec::bounded) fn take_for_release(&mut self) -> ResidentGpuFrame {
        let frame = self.frame.take().expect("YieldedFrame::take_for_release called twice");
        self.release_guard.take_for_release("YieldedFrame::take_for_release called twice");
        frame
    }
}

pub(crate) struct YieldedFrameReleaseGuard {
    live_counter: Arc<AtomicU64>,
    contributed_bytes: u64,
    drained: bool,
}

impl YieldedFrameReleaseGuard {
    pub(in crate::exec::bounded) fn new(
        live_counter: Arc<AtomicU64>,
        contributed_bytes: u64,
        increment_on_create: bool,
    ) -> Self {
        if increment_on_create {
            // Relaxed ordering: this counter is accounting only, never a
            // happens-before edge; saturation preserves the serial semantics.
            let _ = live_counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
                Some(live.saturating_add(contributed_bytes))
            });
        }
        Self { live_counter, contributed_bytes, drained: false }
    }

    pub(crate) fn take_for_release(&mut self, double_take_message: &'static str) {
        assert!(!self.drained, "{double_take_message}");
        let _ = self.live_counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            Some(live.saturating_sub(self.contributed_bytes))
        });
        self.drained = true;
    }
}

impl Drop for YieldedFrameReleaseGuard {
    fn drop(&mut self) {
        if self.drained {
            return;
        }
        let _ = self.live_counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            Some(live.saturating_sub(self.contributed_bytes))
        });
        debug_assert!(
            std::thread::panicking(),
            "YieldedFrame dropped without take_for_release; its {} observed device bytes left the live counter \
             without a consumer taking the frame",
            self.contributed_bytes
        );
    }
}

#[cfg(test)]
pub(crate) struct YieldedFrameCounterTestHarness {
    live_counter: Arc<AtomicU64>,
}

#[cfg(test)]
impl Default for YieldedFrameCounterTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl YieldedFrameCounterTestHarness {
    pub(crate) fn new() -> Self {
        Self { live_counter: Arc::new(AtomicU64::new(0)) }
    }

    pub(crate) fn live_bytes(&self) -> u64 {
        self.live_counter.load(Ordering::Relaxed)
    }

    pub(crate) fn yield_bytes(&self, contributed_bytes: u64) -> YieldedFrameReleaseGuard {
        YieldedFrameReleaseGuard::new(Arc::clone(&self.live_counter), contributed_bytes, true)
    }
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

    /// Device storage still owned by the producer, excluding frames already yielded.
    fn retained_device_bytes(&self) -> u64 {
        0
    }

    /// Retained state carrying the stream's complete group-key set before any
    /// per-frame finalization runs. Producers answer only when the retained
    /// state already names the operator's output group-key columns; the
    /// aggregate columns of those frames are still merge state.
    fn complete_group_key_state(&self) -> Option<Vec<GpuDataFrame>> {
        None
    }
}

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
            max_frame_bytes_hint,
            total_bytes_hint,
            live_observed_device_bytes: Arc::new(AtomicU64::new(total_bytes_hint)),
            aggregate_num_rows,
            num_columns,
            frames: frames.into(),
            producer: None,
        }
    }

    pub(in crate::exec::bounded) fn from_producer(
        max_frame_bytes_hint: u64,
        total_bytes_hint: u64,
        aggregate_num_rows: u64,
        num_columns: usize,
        source: Box<dyn ResidentFrameStreamSource>,
    ) -> Self {
        Self {
            max_frame_bytes_hint,
            total_bytes_hint,
            live_observed_device_bytes: Arc::new(AtomicU64::new(0)),
            aggregate_num_rows,
            num_columns,
            frames: VecDeque::new(),
            producer: Some(ResidentFrameStreamProducer::new(source)),
        }
    }

    pub(in crate::exec::bounded) fn max_frame_bytes_hint(&self) -> u64 {
        self.max_frame_bytes_hint
    }

    pub(in crate::exec::bounded) fn total_bytes_hint(&self) -> u64 {
        self.total_bytes_hint
    }

    pub(in crate::exec::bounded) fn live_observed_device_bytes(&self) -> u64 {
        self.live_observed_device_bytes.load(Ordering::Relaxed).saturating_add(
            self.producer
                .as_ref()
                .and_then(|producer| producer.inner.as_ref())
                .map_or(0, |source| source.retained_device_bytes()),
        )
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

    pub(in crate::exec::bounded) fn frames(&self) -> impl Iterator<Item = &ResidentGpuFrame> {
        assert!(self.producer.is_none(), "lazy frame streams cannot expose an eager frame slice");
        self.frames.iter()
    }

    pub(in crate::exec::bounded) fn alias(&self, node_id: NodeId, memory_boundary: &'static str) -> Result<Self> {
        self.try_alias().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.exec.bounded.compiled.frame_stream",
                "resident frame stream producer cannot create an independent consumer",
            )
            .with_fact("node_id", node_id)
            .with_fact("memory_boundary", memory_boundary)
        })
    }

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

    pub(in crate::exec::bounded) fn into_shared(self, execution: &mut BoundedExecutionAttempt<'_>) -> Result<Self> {
        if let Some(alias) = self.try_alias() {
            return Ok(alias);
        }
        self.into_remaining_frames_owned(execution).map(Self::new)
    }

    pub(in crate::exec::bounded) fn is_empty(&self) -> bool {
        self.frames.is_empty() && self.producer.is_none()
    }

    pub(in crate::exec::bounded) fn len(&self) -> usize {
        self.frames.len()
    }

    pub(in crate::exec::bounded) fn next_yielded_frame(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Option<Result<YieldedFrame>> {
        if !self.frames.is_empty() {
            let frame = self.frames.pop_front().expect("non-empty frame stream had no front frame");
            return Some(Ok(YieldedFrame::from_existing_payload_frame(
                frame,
                Arc::clone(&self.live_observed_device_bytes),
            )));
        }
        let producer = self.producer.as_mut()?;
        producer.next_frame(execution).map(|result| {
            result.map(|frame| YieldedFrame::yield_from_producer(frame, Arc::clone(&self.live_observed_device_bytes)))
        })
    }

    pub(in crate::exec::bounded) fn collect_remaining_frames_owned(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let mut accumulated = Vec::new();
        loop {
            match self.next_yielded_frame(execution) {
                Some(Ok(mut yielded)) => accumulated.push(yielded.take_for_release()),
                Some(Err(error)) => return Err(error),
                None => return Ok(accumulated),
            }
        }
    }

    pub(in crate::exec::bounded) fn into_remaining_frames_owned(
        mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let result = self.collect_remaining_frames_owned(execution);
        self.release_producer(execution);
        result
    }

    pub(in crate::exec::bounded) fn release_remaining_frames(&mut self) {
        self.frames.clear();
        self.live_observed_device_bytes.store(0, Ordering::Relaxed);
    }

    pub(in crate::exec::bounded) fn release_producer(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        let Some(producer) = self.producer.take() else { return };
        producer.release(execution);
    }

    pub(in crate::exec::bounded) fn release_all(mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.release_remaining_frames();
        self.release_producer(execution);
    }
}

#[derive(Clone)]
enum UnaryMappedResidentFrameStreamOperator {
    Filter(FilterNode),
    DynamicFilter { filter: FilterNode, route: DynamicFilterRoute, dynamic_filter: DynamicFilter },
    Projection(ProjectionNode),
}

struct UnaryMappedResidentFrameStreamSource {
    node_id: NodeId,
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

    fn retained_device_bytes(&self) -> u64 {
        self.input.as_ref().map_or(0, ResidentFrameStreamPayload::live_observed_device_bytes)
    }

    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let input = self.input.as_mut()?;
        match input.next_yielded_frame(execution) {
            Some(Ok(mut yielded)) => {
                let frame = yielded.take_for_release();
                let input_rows = frame.frame.num_rows() as u64;
                Some(self.execute(execution, frame).inspect(|output| {
                    self.record_metrics(input_rows, output, execution.metrics());
                }))
            },
            Some(Err(error)) => Some(Err(error)),
            None => {
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
