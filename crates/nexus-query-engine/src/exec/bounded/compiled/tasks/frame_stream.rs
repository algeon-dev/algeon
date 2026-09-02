use super::*;

pub(in crate::exec::bounded) struct ResidentFrameStreamPayload {
    /// Worst-case bytes a single yielded "step" can occupy on the
    /// device. Consumed by `resource_admission_estimate_bytes` (the
    /// scheduler's admission input sizing).
    max_per_frame_reservation_bytes: u64,
    /// Sum of per-frame reservation bytes across the whole stream.
    /// Consumed by `aggregate_reservation_bytes()` and start-of-stream
    /// contract accounting for producer-backed payloads.
    aggregate_reservation_bytes: u64,
    /// Currently-live frame reservation bytes. Read by
    /// `reclaimable_input_bytes` (`tasks.rs:reclaimable_input_bytes`).
    /// For Vec-backed payloads this equals `aggregate_reservation_bytes`
    /// because all frames are immediately live.
    live_reservation_bytes: Rc<Cell<u64>>,
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
    pub(in crate::exec::bounded) fn yield_from_producer(frame: ResidentGpuFrame, live_counter: Rc<Cell<u64>>) -> Self {
        let contributed_bytes = frame.reservation.bytes();
        Self { frame: Some(frame), release_guard: YieldedFrameReleaseGuard::new(live_counter, contributed_bytes, true) }
    }

    fn from_existing_payload_frame(frame: ResidentGpuFrame, live_counter: Rc<Cell<u64>>) -> Self {
        let contributed_bytes = frame.reservation.bytes();
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

struct YieldedFrameReleaseGuard {
    live_counter: Rc<Cell<u64>>,
    contributed_bytes: u64,
    drained: bool,
}

impl YieldedFrameReleaseGuard {
    fn new(live_counter: Rc<Cell<u64>>, contributed_bytes: u64, increment_on_create: bool) -> Self {
        if increment_on_create {
            live_counter.set(live_counter.get().saturating_add(contributed_bytes));
        }
        Self { live_counter, contributed_bytes, drained: false }
    }

    fn take_for_release(&mut self, double_take_message: &'static str) {
        assert!(!self.drained, "{double_take_message}");
        self.live_counter.set(self.live_counter.get().saturating_sub(self.contributed_bytes));
        self.drained = true;
    }
}

impl Drop for YieldedFrameReleaseGuard {
    fn drop(&mut self) {
        if self.drained {
            return;
        }
        self.live_counter.set(self.live_counter.get().saturating_sub(self.contributed_bytes));
        debug_assert!(
            false,
            "YieldedFrame dropped without explicit release; reservation bytes={} now leak the device pool",
            self.contributed_bytes
        );
    }
}

#[cfg(test)]
pub(crate) struct YieldedFrameCounterTestHarness {
    live_counter: Rc<Cell<u64>>,
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
        Self { live_counter: Rc::new(Cell::new(0)) }
    }

    pub(crate) fn live_bytes(&self) -> u64 {
        self.live_counter.get()
    }

    pub(crate) fn yield_bytes(&self, contributed_bytes: u64) -> YieldedFrameCounterTestGuard {
        YieldedFrameCounterTestGuard {
            release_guard: YieldedFrameReleaseGuard::new(Rc::clone(&self.live_counter), contributed_bytes, true),
        }
    }
}

#[cfg(test)]
pub(crate) struct YieldedFrameCounterTestGuard {
    release_guard: YieldedFrameReleaseGuard,
}

#[cfg(test)]
impl YieldedFrameCounterTestGuard {
    pub(crate) fn take_for_release(&mut self) {
        self.release_guard.take_for_release("YieldedFrameCounterTestGuard::take_for_release called twice");
    }
}

pub(in crate::exec::bounded) trait ResidentFrameStreamSource {
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>>;

    fn release(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) {}
}

struct ResidentFrameStreamProducer {
    producer_reservations: Vec<ActiveReservation>,
    inner: Option<Box<dyn ResidentFrameStreamSource>>,
}

impl ResidentFrameStreamProducer {
    fn new(producer_reservations: Vec<ActiveReservation>, inner: Box<dyn ResidentFrameStreamSource>) -> Self {
        Self { producer_reservations, inner: Some(inner) }
    }

    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        self.inner.as_mut()?.next_frame(execution)
    }

    fn release(mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        if let Some(mut inner) = self.inner.take() {
            inner.release(execution);
        }
        for reservation in self.producer_reservations.drain(..).rev() {
            execution.release(reservation);
        }
    }
}

impl Drop for ResidentFrameStreamProducer {
    fn drop(&mut self) {
        if self.producer_reservations.is_empty() {
            return;
        }
        debug_assert!(
            false,
            "ResidentFrameStreamProducer dropped without explicit release; producer_reservation_count={}",
            self.producer_reservations.len()
        );
    }
}

impl ResidentFrameStreamPayload {
    pub(in crate::exec::bounded) fn new(frames: Vec<ResidentGpuFrame>) -> Self {
        let aggregate_reservation_bytes =
            frames.iter().fold(0u64, |total, frame| total.saturating_add(frame.reservation.bytes()));
        let max_per_frame_reservation_bytes =
            frames.iter().fold(0u64, |max_bytes, frame| max_bytes.max(frame.reservation.bytes()));
        let aggregate_num_rows =
            frames.iter().fold(0u64, |total, frame| total.saturating_add(frame.frame.num_rows() as u64));
        let num_columns = frames.first().map_or(0, |frame| frame.frame.num_columns());
        Self {
            max_per_frame_reservation_bytes,
            aggregate_reservation_bytes,
            live_reservation_bytes: Rc::new(Cell::new(aggregate_reservation_bytes)),
            aggregate_num_rows,
            num_columns,
            frames: frames.into(),
            producer: None,
        }
    }

    pub(in crate::exec::bounded) fn from_producer(
        max_per_frame_reservation_bytes: u64,
        aggregate_reservation_bytes: u64,
        aggregate_num_rows: u64,
        num_columns: usize,
        producer_reservations: Vec<ActiveReservation>,
        source: Box<dyn ResidentFrameStreamSource>,
    ) -> Self {
        Self {
            max_per_frame_reservation_bytes,
            aggregate_reservation_bytes,
            live_reservation_bytes: Rc::new(Cell::new(0)),
            aggregate_num_rows,
            num_columns,
            frames: VecDeque::new(),
            producer: Some(ResidentFrameStreamProducer::new(producer_reservations, source)),
        }
    }

    pub(in crate::exec::bounded) fn max_per_frame_reservation_bytes(&self) -> u64 {
        self.max_per_frame_reservation_bytes
    }

    pub(in crate::exec::bounded) fn aggregate_reservation_bytes(&self) -> u64 {
        self.aggregate_reservation_bytes
    }

    pub(in crate::exec::bounded) fn live_reservation_bytes(&self) -> u64 {
        self.live_reservation_bytes.get()
    }

    pub(in crate::exec::bounded) fn num_rows(&self) -> u64 {
        self.aggregate_num_rows
    }

    pub(in crate::exec::bounded) fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub(in crate::exec::bounded) fn frames(&self) -> impl Iterator<Item = &ResidentGpuFrame> {
        assert!(self.producer.is_none(), "lazy frame streams cannot expose an eager frame slice");
        self.frames.iter()
    }

    pub(in crate::exec::bounded) fn is_eager(&self) -> bool {
        self.producer.is_none()
    }

    pub(in crate::exec::bounded) fn alias_eager(
        &self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node_id: NodeId,
        reservation_kind: ReservationKind,
        memory_boundary: &'static str,
    ) -> Result<Self> {
        if !self.is_eager() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.exec.bounded.compiled.frame_stream",
                "lazy resident frame streams cannot be shared by alias",
            )
            .with_fact("node_id", node_id)
            .with_fact("memory_boundary", memory_boundary));
        }
        let frames = self
            .frames
            .iter()
            .map(|frame| ResidentGpuFrame {
                frame: frame.frame.clone(),
                reservation: execution.alias_reservation(
                    &frame.reservation,
                    MemoryReservationRequest::new(
                        reservation_kind,
                        frame.reservation.bytes(),
                        execution.reservations().budget(),
                    )
                    .with_memory_boundary(memory_boundary)
                    .with_node_id(node_id)
                    .with_stage(memory_boundary),
                ),
            })
            .collect::<Vec<_>>();
        Ok(Self::new(frames))
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
            return Some(Ok(YieldedFrame::from_existing_payload_frame(frame, Rc::clone(&self.live_reservation_bytes))));
        }
        let producer = self.producer.as_mut()?;
        producer.next_frame(execution).map(|result| {
            result.map(|frame| YieldedFrame::yield_from_producer(frame, Rc::clone(&self.live_reservation_bytes)))
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
                Some(Err(error)) => {
                    while let Some(frame) = accumulated.pop() {
                        execution.release(frame.reservation);
                    }
                    return Err(error);
                },
                None => return Ok(accumulated),
            }
        }
    }

    pub(in crate::exec::bounded) fn into_remaining_frames_owned(
        mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) -> Result<Vec<ResidentGpuFrame>> {
        let result = self.collect_remaining_frames_owned(execution);
        self.release_producer_reservations(execution);
        result
    }

    pub(in crate::exec::bounded) fn release_remaining_frames(&mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        while let Some(frame) = self.frames.pop_back() {
            execution.release(frame.reservation);
        }
        self.live_reservation_bytes.set(0);
    }

    pub(in crate::exec::bounded) fn release_producer_reservations(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
    ) {
        let Some(producer) = self.producer.take() else { return };
        producer.release(execution);
    }

    pub(in crate::exec::bounded) fn release_all(mut self, execution: &mut BoundedExecutionAttempt<'_>) {
        self.release_remaining_frames(execution);
        self.release_producer_reservations(execution);
    }
}

#[cfg(test)]
struct EmptyResidentFrameStreamSource;

#[cfg(test)]
impl ResidentFrameStreamSource for EmptyResidentFrameStreamSource {
    fn next_frame(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        None
    }
}

#[cfg(test)]
struct ErrorResidentFrameStreamSource {
    emitted: bool,
}

#[cfg(test)]
impl ResidentFrameStreamSource for ErrorResidentFrameStreamSource {
    fn next_frame(&mut self, _execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        if self.emitted {
            return None;
        }
        self.emitted = true;
        Some(Err(QueryError::new(
            QueryErrorCode::Internal,
            "nexus_query_engine.exec.bounded.compiled.resident_frame_stream_test",
            "injected resident frame stream source failure",
        )))
    }
}

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
            | UnaryMappedResidentFrameStreamOperator::DynamicFilter { filter, .. } => execute_bounded_generic_unary(
                execution,
                self.node_id,
                "compiled_filter",
                "compiled_filter_output",
                input,
                true,
                |context, input, token, metrics| {
                    crate::exec::operators::execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for compiled bounded filter stages"),
                        metrics,
                    )
                },
            ),
            UnaryMappedResidentFrameStreamOperator::Projection(projection) => execute_bounded_generic_projection(
                execution,
                self.node_id,
                "compiled_projection",
                "compiled_projection_output",
                projection,
                input,
            ),
        }
    }

    fn record_metrics(&self, input_rows: u64, output: &ResidentGpuFrame, metrics: &mut QueryMetrics) {
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
    fn next_frame(&mut self, execution: &mut BoundedExecutionAttempt<'_>) -> Option<Result<ResidentGpuFrame>> {
        let input = self.input.as_mut()?;
        match input.next_yielded_frame(execution) {
            Some(Ok(mut yielded)) => {
                let frame = yielded.take_for_release();
                let input_rows = frame.frame.num_rows() as u64;
                Some(self.execute(execution, frame).inspect(|output| {
                    self.record_metrics(input_rows, output, execution.metrics_mut());
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
    let max_per_frame_reservation_bytes = stream.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = stream.aggregate_reservation_bytes();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = stream.num_columns();
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
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
    let max_per_frame_reservation_bytes = stream.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = stream.aggregate_reservation_bytes();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = stream.num_columns();
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
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
    let max_per_frame_reservation_bytes = stream.max_per_frame_reservation_bytes();
    let aggregate_reservation_bytes = stream.aggregate_reservation_bytes();
    let aggregate_num_rows = stream.num_rows();
    let num_columns = projection.output_schema().fields().len();
    ResidentFrameStreamPayload::from_producer(
        max_per_frame_reservation_bytes,
        aggregate_reservation_bytes,
        aggregate_num_rows,
        num_columns,
        Vec::new(),
        Box::new(UnaryMappedResidentFrameStreamSource::from_projection(node_id, stream, projection)),
    )
}

#[cfg(test)]
pub(crate) struct ResidentFrameStreamProducerTestHarness<'a> {
    payload: Option<ResidentFrameStreamPayload>,
    execution: BoundedExecutionAttempt<'a>,
}

#[cfg(test)]
impl<'a> ResidentFrameStreamProducerTestHarness<'a> {
    pub(crate) fn new(
        context: &'a ExecutionContext,
        metrics: &'a mut QueryMetrics,
        producer_reservation_bytes: &[u64],
    ) -> Result<Self> {
        Self::with_source(context, metrics, producer_reservation_bytes, Box::new(EmptyResidentFrameStreamSource))
    }

    pub(crate) fn new_erroring(
        context: &'a ExecutionContext,
        metrics: &'a mut QueryMetrics,
        producer_reservation_bytes: &[u64],
    ) -> Result<Self> {
        Self::with_source(
            context,
            metrics,
            producer_reservation_bytes,
            Box::new(ErrorResidentFrameStreamSource { emitted: false }),
        )
    }

    fn with_source(
        context: &'a ExecutionContext,
        metrics: &'a mut QueryMetrics,
        producer_reservation_bytes: &[u64],
        source: Box<dyn ResidentFrameStreamSource>,
    ) -> Result<Self> {
        let budget = producer_reservation_bytes.iter().fold(1u64, |total, bytes| total.saturating_add(*bytes)).max(1);
        let mut execution = BoundedExecutionAttempt::new(context, budget, metrics, None)?;
        let mut producer_reservations = Vec::with_capacity(producer_reservation_bytes.len());
        for (index, bytes) in producer_reservation_bytes.iter().enumerate() {
            let request = MemoryReservationRequest::new(ReservationKind::OperatorOutput, *bytes, budget)
                .with_memory_boundary("resident_frame_stream_producer_test")
                .with_stage("resident_frame_stream_producer_test")
                .with_chunk_index(index);
            producer_reservations.push(execution.try_reserve_owned_resident(request)?);
        }
        let aggregate_reservation_bytes =
            producer_reservation_bytes.iter().fold(0u64, |total, bytes| total.saturating_add(*bytes));
        let max_per_frame_reservation_bytes = producer_reservation_bytes.iter().copied().max().unwrap_or(0);
        let payload = ResidentFrameStreamPayload::from_producer(
            max_per_frame_reservation_bytes,
            aggregate_reservation_bytes,
            0,
            0,
            producer_reservations,
            source,
        );
        Ok(Self { payload: Some(payload), execution })
    }

    pub(crate) fn reserved_bytes(&self) -> u64 {
        self.execution.reservations().reserved_bytes()
    }

    pub(crate) fn release_producer_reservations(&mut self) {
        if let Some(payload) = self.payload.as_mut() {
            payload.release_producer_reservations(&mut self.execution);
        }
    }

    pub(crate) fn collect_remaining_frames_owned_then_release(&mut self) -> Result<usize> {
        let result = self
            .payload
            .as_mut()
            .expect("producer test payload already dropped")
            .collect_remaining_frames_owned(&mut self.execution)
            .map(|frames| frames.len());
        self.release_producer_reservations();
        result
    }

    pub(crate) fn drop_payload_without_release(&mut self) {
        let _ = self.payload.take();
    }
}
