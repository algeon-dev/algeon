use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use arrow::array::new_empty_array;
use cudf::DataFrame;
use futures::{StreamExt, stream::iter};
use query_engine::{
    error::QueryErrorCode, native_error::cudf_query_error, plan::QueryPlan, runtime::ExecutionCancellation,
};

use super::{
    AlgeonNativeComposedExec, AlgeonNativeExec, AlgeonNativeMetrics, BoxFuture, ChildrenPropertiesMode,
    CugraphAlgorithmSpec, DataFusionError, Error, ErrorCode, ExecutionContext, ExecutionPlan, GpuDataFrame,
    GpuResidentInputMap, GpuResultHandle, GraphBfsPublishedResult, GraphExecutionMetrics, HostBatchCursorConfig,
    MemoryConsumer, MemoryReservation, QueryExecutionRequest, QueryOutput, RecordBatch, RecordBatchStream,
    RecordOutput, ReplaceChildrenOptions, Result, SchemaRef, Stream, TaskContext, TryStreamExt,
    as_algeon_native_composed_exec, as_algeon_native_exec, counted_cudf_runtime, edge_source_output_rows,
    execute_gpu_resident_producer_with_context, execute_query_plan, finalize_graph_query_outcome,
    get_record_batch_memory_size, gpu_resident_producer_input_name, native_gpu_resident_producer,
    plan_contains_algeon_native, prepare_gpu_resident_producer_for_execution, resident_side_input_output_rows,
};
use crate::native_exec::{lakehouse::resolve_query_plan, validate_host_wrapped_gpu_execution};

const CUGRAPH_HOST_INPUT_MEMORY_CONSUMER: &str = "CugraphAlgoExec[host-input-staging]";
const CUGRAPH_HOST_OUTPUT_MEMORY_CONSUMER: &str = "CugraphAlgoExec[host-output-staging]";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostReservationEvidence {
    input_limit_bytes: usize,
    output_limit_bytes: usize,
    input_staging_peak_bytes: usize,
    aggregate_peak_bytes: usize,
}

impl HostReservationEvidence {
    pub(super) fn new(input_limit_bytes: usize, output_limit_bytes: usize) -> Self {
        Self {
            input_limit_bytes,
            output_limit_bytes,
            input_staging_peak_bytes: 0,
            aggregate_peak_bytes: output_limit_bytes,
        }
    }

    pub(super) fn record_input_peak(&mut self, reservation: &MemoryReservation) -> Result<()> {
        if reservation.size() > self.input_limit_bytes {
            return Err(host_staging_limit_error(
                "graph_input",
                "input_evidence",
                self.input_limit_bytes,
                reservation.size() - self.input_limit_bytes,
                self.input_limit_bytes,
            ));
        }
        self.input_staging_peak_bytes = self.input_staging_peak_bytes.max(reservation.size());
        self.aggregate_peak_bytes =
            self.input_staging_peak_bytes.checked_add(self.output_limit_bytes).ok_or_else(|| {
                Error::new(
                    ErrorCode::SizeOverflow,
                    "algeon_datafusion.cugraph_sql.host_staging_evidence",
                    "host staging evidence aggregate bound overflowed usize",
                )
                .with_fact("reason", "host_staging_evidence_overflow")
                .with_fact("input_peak_bytes", self.input_staging_peak_bytes)
                .with_fact("output_limit_bytes", self.output_limit_bytes)
            })?;
        Ok(())
    }

    pub(super) fn output_limit_bytes(&self) -> usize {
        self.output_limit_bytes
    }

    #[cfg(test)]
    pub(super) fn input_limit_bytes(&self) -> usize {
        self.input_limit_bytes
    }

    #[cfg(test)]
    pub(super) fn input_staging_peak_bytes(&self) -> usize {
        self.input_staging_peak_bytes
    }

    #[cfg(test)]
    pub(super) fn aggregate_peak_bytes(&self) -> usize {
        self.aggregate_peak_bytes
    }

    pub(super) fn check_output_growth(
        &self,
        boundary: &'static str,
        current_bytes: usize,
        requested_bytes: usize,
    ) -> Result<()> {
        let next_bytes = current_bytes.checked_add(requested_bytes).ok_or_else(|| {
            host_staging_limit_error("graph_result", boundary, current_bytes, requested_bytes, self.output_limit_bytes)
        })?;
        let next_total = self.input_staging_peak_bytes.checked_add(next_bytes).ok_or_else(|| {
            host_staging_limit_error(
                "graph_result",
                boundary,
                self.input_staging_peak_bytes,
                next_bytes,
                self.aggregate_peak_bytes,
            )
        })?;
        if next_bytes > self.output_limit_bytes || next_total > self.aggregate_peak_bytes {
            return Err(host_staging_limit_error(
                "graph_result",
                boundary,
                current_bytes,
                requested_bytes,
                self.aggregate_peak_bytes,
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(super) enum StagedCugraphSource {
    Native { schema: SchemaRef, plan: Arc<dyn ExecutionPlan> },
    Host { schema: SchemaRef, batches: Vec<RecordBatch> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CugraphInputRowBounds {
    edge: Option<u64>,
    predicates: Vec<Option<u64>>,
}

impl CugraphInputRowBounds {
    pub(super) fn from_execution_plans(
        edge_source: &dyn ExecutionPlan,
        predicate_sources: &[Arc<dyn ExecutionPlan>],
    ) -> Result<Self> {
        Ok(Self {
            edge: edge_source_output_rows(edge_source)?,
            predicates: predicate_sources
                .iter()
                .map(|source| resident_side_input_output_rows(source.as_ref()))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    pub(super) fn edge_row_upper_bound(&self) -> Option<u64> {
        self.edge
    }

    pub(super) fn predicate_row_upper_bounds(&self) -> &[Option<u64>] {
        &self.predicates
    }
}

#[derive(Debug)]
pub(super) struct StagedCugraphInputs {
    pub(super) edge: StagedCugraphSource,
    pub(super) predicates: Vec<StagedCugraphSource>,
    pub(super) admitted_graph_plan: Option<QueryPlan>,
    pub(super) admitted_edge_source_gpu_plan: Option<QueryPlan>,
    pub(super) host_evidence: HostReservationEvidence,
    pub(super) reservation: Option<MemoryReservation>,
}

#[derive(Debug)]
pub(super) struct CollectedCugraphInputs {
    edge: StagedCugraphSource,
    predicates: Vec<StagedCugraphSource>,
    host_evidence: HostReservationEvidence,
    reservation: MemoryReservation,
}

pub(super) struct UploadedCugraphInputs {
    pub(super) edge_schema: SchemaRef,
    pub(super) edge_frame: GpuDataFrame,
    pub(super) predicate_frames: Vec<CugraphGpuFrame>,
    pub(super) host_evidence: HostReservationEvidence,
    pub(super) reservation: Option<MemoryReservation>,
}

#[cfg(test)]
pub(super) async fn stage_cugraph_inputs(
    spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    max_staging_bytes: usize,
) -> Result<StagedCugraphInputs> {
    collect_cugraph_inputs(
        spec,
        edge_source,
        predicate_sources,
        task_context,
        cancellation,
        max_staging_bytes,
        max_staging_bytes,
    )
    .await?
    .prepare_logical_inputs(spec)
}

pub(super) async fn collect_cugraph_inputs(
    _spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    max_input_staging_bytes: usize,
    max_output_staging_bytes: usize,
) -> Result<CollectedCugraphInputs> {
    cancellation.check("algeon_datafusion.cugraph_sql.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.host_input_staging",
            "cuGraph host-input staging was cancelled before execution",
            error,
        ))
    })?;
    let reservation = MemoryConsumer::new(CUGRAPH_HOST_INPUT_MEMORY_CONSUMER)
        .with_can_spill(false)
        .register(task_context.memory_pool());
    let staged_bytes = AtomicUsize::new(0);
    let host_evidence = HostReservationEvidence::new(max_input_staging_bytes, max_output_staging_bytes);
    let edge = stage_cugraph_source(
        "edge",
        edge_source,
        Arc::clone(&task_context),
        cancellation,
        max_input_staging_bytes,
        &staged_bytes,
        &reservation,
    )
    .await?;
    let mut predicates = Vec::with_capacity(predicate_sources.len());
    for source in predicate_sources {
        predicates.push(
            stage_cugraph_source(
                "side_input",
                source,
                Arc::clone(&task_context),
                cancellation,
                max_input_staging_bytes,
                &staged_bytes,
                &reservation,
            )
            .await?,
        );
    }
    Ok(CollectedCugraphInputs { edge, predicates, host_evidence, reservation })
}

async fn stage_cugraph_source(
    source_kind: &'static str,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    cancellation: &ExecutionCancellation,
    max_staging_bytes: usize,
    staged_bytes: &AtomicUsize,
    reservation: &MemoryReservation,
) -> Result<StagedCugraphSource> {
    if native_gpu_resident_producer(plan.as_ref()).is_some() {
        let plan = prepare_gpu_resident_producer_for_execution(plan, Arc::clone(&task_context)).await?;
        return Ok(StagedCugraphSource::Native { schema: plan.schema(), plan });
    }
    let schema = plan.schema();
    if as_algeon_native_exec(plan.as_ref()).is_some() || as_algeon_native_composed_exec(plan.as_ref()).is_some() {
        return Ok(StagedCugraphSource::Native { schema, plan });
    }
    if plan_contains_algeon_native(plan.as_ref()) {
        validate_host_wrapped_gpu_execution(plan.as_ref(), task_context.as_ref())?;
    }

    // Every partition stays in flight: a DataFusion operator can hold one output partition until
    // its siblings progress (a CollectLeft hash join's dynamic filter waits for every probe
    // partition to report build bounds), so draining partitions one at a time deadlocks.
    // `buffered` still yields partitions in ordinal order, keeping the staged edge order stable.
    let partition_count = plan.properties().output_partitioning().partition_count();
    let partitions = iter(0..partition_count)
        .map(|partition| {
            let plan = &plan;
            let task_context = Arc::clone(&task_context);
            async move {
                check_host_staging_cancellation(cancellation)?;
                let mut stream = plan.execute(partition, task_context)?;
                let mut batches = Vec::new();
                while let Some(batch) = stream.try_next().await? {
                    check_host_staging_cancellation(cancellation)?;
                    reserve_host_staging_batch(
                        source_kind,
                        "input",
                        &batch,
                        max_staging_bytes,
                        staged_bytes,
                        reservation,
                    )?;
                    batches.push(batch);
                }
                Ok::<_, DataFusionError>(batches)
            }
        })
        .buffered(partition_count.max(1))
        .try_collect::<Vec<_>>()
        .await?;
    Ok(StagedCugraphSource::Host { schema, batches: partitions.into_iter().flatten().collect() })
}

pub(super) fn check_host_staging_cancellation(cancellation: &ExecutionCancellation) -> Result<()> {
    cancellation.check("algeon_datafusion.cugraph_sql.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.host_input_staging",
            "cuGraph host-input staging was cancelled",
            error,
        ))
    })
}

fn reserve_host_staging_batch(
    source_kind: &'static str,
    boundary: &'static str,
    batch: &RecordBatch,
    max_staging_bytes: usize,
    staged_bytes: &AtomicUsize,
    reservation: &MemoryReservation,
) -> Result<()> {
    let batch_bytes = get_record_batch_memory_size(batch);
    // Claim through compare-exchange so concurrently polled partitions cannot jointly overshoot
    // the cumulative cap; a denied pool reservation below returns the claim.
    let staged = loop {
        let staged = staged_bytes.load(Ordering::Relaxed);
        let next = staged
            .checked_add(batch_bytes)
            .filter(|next| *next <= max_staging_bytes)
            .ok_or_else(|| host_staging_limit_error(source_kind, boundary, staged, batch_bytes, max_staging_bytes))?;
        if staged_bytes.compare_exchange(staged, next, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            break staged;
        }
    };
    reservation.try_grow(batch_bytes).map_err(|source| {
        staged_bytes.fetch_sub(batch_bytes, Ordering::Relaxed);
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "algeon_datafusion.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied cuGraph staging",
            )
            .with_fact("boundary", boundary)
            .with_fact("source_kind", source_kind)
            .with_fact("requested_bytes", batch_bytes)
            .with_fact("staged_bytes", staged)
            .with_source(source),
        )
    })?;
    Ok(())
}

pub(super) fn host_staging_limit_error(
    source_kind: &'static str,
    boundary: &'static str,
    staged_bytes: usize,
    next_batch_bytes: usize,
    max_staging_bytes: usize,
) -> DataFusionError {
    Error::new(
        ErrorCode::HostStagingLimitExceeded,
        "algeon_datafusion.cugraph_sql.host_staging_limit",
        "cuGraph GPU-island host staging exceeded its configured cumulative byte limit",
    )
    .with_fact("boundary", boundary)
    .with_fact("source_kind", source_kind)
    .with_fact("staged_bytes", staged_bytes)
    .with_fact("next_batch_bytes", next_batch_bytes)
    .with_fact("max_staging_bytes", max_staging_bytes)
    .into()
}

impl CollectedCugraphInputs {
    pub(super) fn row_bounds(&self) -> Result<CugraphInputRowBounds> {
        cugraph_input_row_bounds(&self.edge, &self.predicates)
    }

    pub(super) fn prepare_logical_inputs(self, _spec: &CugraphAlgorithmSpec) -> Result<StagedCugraphInputs> {
        let Self { edge, predicates, mut host_evidence, reservation } = self;
        host_evidence.record_input_peak(&reservation)?;
        Ok(StagedCugraphInputs {
            edge,
            predicates,
            admitted_graph_plan: None,
            admitted_edge_source_gpu_plan: None,
            host_evidence,
            reservation: Some(reservation),
        })
    }
}

impl StagedCugraphInputs {
    pub(super) fn row_bounds(&self) -> Result<CugraphInputRowBounds> {
        cugraph_input_row_bounds(&self.edge, &self.predicates)
    }

    pub(super) fn validate_admission(&self, context: &ExecutionContext) -> Result<()> {
        if !context.is_query_service_admitted() {
            return Ok(());
        }
        let graph_plan = self.admitted_graph_plan.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.admission",
                "cuGraph staged inputs are missing the admission-time graph execution plan",
            )
            .with_fact("reason", "admission_graph_plan_missing")
        })?;
        context.validate_admitted_plan_capabilities(graph_plan).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cugraph_sql.admission",
                "the admitted cuGraph graph plan does not match runtime execution",
                error,
            ))
        })
    }

    pub(super) async fn upload(
        self,
        context: &ExecutionContext,
        metrics: &mut GraphExecutionMetrics<'_>,
        task_context: Arc<TaskContext>,
    ) -> Result<UploadedCugraphInputs> {
        self.validate_admission(context)?;
        let Self {
            edge,
            predicates,
            admitted_graph_plan: _,
            admitted_edge_source_gpu_plan,
            host_evidence,
            reservation,
        } = self;
        let edge_schema = edge.schema();
        let edge_frame =
            edge.upload(context, metrics, Arc::clone(&task_context), admitted_edge_source_gpu_plan).await?;
        let mut predicate_frames = Vec::with_capacity(predicates.len());
        for source in predicates {
            let schema = source.schema();
            let frame = source.upload(context, metrics, Arc::clone(&task_context), None).await?;
            predicate_frames.push(CugraphGpuFrame { schema, frame });
        }
        let _reservation = reservation;
        Ok(UploadedCugraphInputs { edge_schema, edge_frame, predicate_frames, host_evidence, reservation: None })
    }
}

fn cugraph_input_row_bounds(
    edge: &StagedCugraphSource,
    predicates: &[StagedCugraphSource],
) -> Result<CugraphInputRowBounds> {
    Ok(CugraphInputRowBounds {
        edge: edge.row_upper_bound(StagedCugraphSourceRole::Edge)?,
        predicates: predicates
            .iter()
            .enumerate()
            .map(|(index, source)| source.row_upper_bound(StagedCugraphSourceRole::Predicate(index)))
            .collect::<Result<Vec<_>>>()?,
    })
}

#[derive(Clone, Copy)]
enum StagedCugraphSourceRole {
    Edge,
    Predicate(usize),
}

impl StagedCugraphSource {
    pub(super) fn schema(&self) -> SchemaRef {
        match self {
            Self::Native { schema, .. } | Self::Host { schema, .. } => Arc::clone(schema),
        }
    }

    fn row_upper_bound(&self, role: StagedCugraphSourceRole) -> Result<Option<u64>> {
        match self {
            Self::Native { plan, .. } => match role {
                StagedCugraphSourceRole::Edge => edge_source_output_rows(plan.as_ref()),
                StagedCugraphSourceRole::Predicate(_) => resident_side_input_output_rows(plan.as_ref()),
            },
            Self::Host { batches, .. } => checked_staged_host_rows(batches, role).map(Some),
        }
    }

    async fn upload(
        self,
        context: &ExecutionContext,
        metrics: &mut GraphExecutionMetrics<'_>,
        task_context: Arc<TaskContext>,
        admitted_edge_source_gpu_plan: Option<QueryPlan>,
    ) -> Result<GpuDataFrame> {
        match self {
            Self::Native { plan, .. } => {
                execute_edge_source_to_gpu(context, metrics, plan, task_context, admitted_edge_source_gpu_plan).await
            },
            Self::Host { schema, batches } => record_batches_to_gpu(context, schema, &batches),
        }
    }
}

fn checked_staged_host_rows(batches: &[RecordBatch], role: StagedCugraphSourceRole) -> Result<u64> {
    batches.iter().try_fold(0_u64, |rows, batch| {
        let batch_rows = u64::try_from(batch.num_rows()).map_err(|_| staged_host_row_overflow(role, rows, u64::MAX))?;
        rows.checked_add(batch_rows).ok_or_else(|| staged_host_row_overflow(role, rows, batch_rows))
    })
}

fn staged_host_row_overflow(
    role: StagedCugraphSourceRole,
    accumulated_rows: u64,
    next_batch_rows: u64,
) -> DataFusionError {
    let mut error = Error::new(
        ErrorCode::SizeOverflow,
        "algeon_datafusion.cugraph_sql.host_input_rows",
        "staged cuGraph host-input row count overflowed the native cardinality contract",
    )
    .with_fact(
        "source_kind",
        match role {
            StagedCugraphSourceRole::Edge => "edge",
            StagedCugraphSourceRole::Predicate(_) => "side_input",
        },
    )
    .with_fact("accumulated_rows", accumulated_rows)
    .with_fact("next_batch_rows", next_batch_rows);
    if let StagedCugraphSourceRole::Predicate(index) = role {
        error = error.with_fact("predicate_index", index);
    }
    error.into()
}

#[derive(Clone, Debug)]
pub(super) struct EagerCugraphHostOutput {
    schema: SchemaRef,
    batches: Arc<Vec<RecordBatch>>,
    next_batch: usize,
    reservation: Arc<MemoryReservation>,
    host_evidence: Option<HostReservationEvidence>,
}

impl EagerCugraphHostOutput {
    pub(super) fn into_parts_with_evidence(
        self,
    ) -> (Arc<Vec<RecordBatch>>, Arc<MemoryReservation>, Option<HostReservationEvidence>) {
        (self.batches, self.reservation, self.host_evidence)
    }

    pub(super) fn shared_parts(&self) -> (Arc<Vec<RecordBatch>>, Arc<MemoryReservation>) {
        (Arc::clone(&self.batches), Arc::clone(&self.reservation))
    }

    pub(super) fn from_shared_parts(
        schema: SchemaRef,
        batches: Arc<Vec<RecordBatch>>,
        reservation: Arc<MemoryReservation>,
    ) -> Self {
        Self { schema, batches, next_batch: 0, reservation, host_evidence: None }
    }

    pub(super) fn from_shared_parts_with_evidence(
        schema: SchemaRef,
        batches: Arc<Vec<RecordBatch>>,
        reservation: Arc<MemoryReservation>,
        host_evidence: Option<HostReservationEvidence>,
    ) -> Self {
        Self { schema, batches, next_batch: 0, reservation, host_evidence }
    }

    pub(super) fn published_result(&self) -> GraphBfsPublishedResult {
        let (batches, reservation) = self.shared_parts();
        GraphBfsPublishedResult::new(batches, reservation)
    }

    pub(super) fn from_published_result(schema: SchemaRef, result: &GraphBfsPublishedResult) -> Option<Self> {
        let (batches, reservation) = result.host_shared_parts()?;
        Some(Self::from_shared_parts(schema, batches, reservation))
    }

    pub(super) fn record_datafusion_output(self, metrics: &AlgeonNativeMetrics) -> Self {
        let schema = Arc::clone(&self.schema);
        let (batches, reservation, host_evidence) = self.into_parts_with_evidence();
        let batches = batches.iter().cloned().map(|batch| batch.record_output(&metrics.baseline)).collect();
        Self::from_shared_parts_with_evidence(schema, Arc::new(batches), reservation, host_evidence)
    }
}

impl Stream for EagerCugraphHostOutput {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let batch = self.batches.get(self.next_batch).cloned();
        self.next_batch = self.next_batch.saturating_add(usize::from(batch.is_some()));
        Poll::Ready(batch.map(Ok))
    }
}

impl RecordBatchStream for EagerCugraphHostOutput {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

pub(super) fn materialize_gpu_output_eagerly(
    handle: GpuResultHandle,
    task_context: &TaskContext,
    cancellation: ExecutionCancellation,
    max_staging_bytes: usize,
    host_evidence: Option<&HostReservationEvidence>,
    metrics: &mut GraphExecutionMetrics<'_>,
) -> Result<EagerCugraphHostOutput> {
    let output_limit_bytes =
        host_evidence.map_or(max_staging_bytes, |evidence| evidence.output_limit_bytes().min(max_staging_bytes));
    let schema = Arc::clone(handle.schema());
    let max_materialization_bytes = output_limit_bytes.min(HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES);
    let target_materialization_bytes =
        output_limit_bytes.min(HostBatchCursorConfig::DEFAULT_TARGET_MATERIALIZATION_BYTES);
    let config = HostBatchCursorConfig {
        target_materialization_bytes,
        max_materialization_bytes,
        ..HostBatchCursorConfig::default()
    };
    let mut cursor = handle.into_host_batch_cursor(cancellation, config).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.host_output_cursor",
            "failed to create bounded cuGraph host-output cursor",
            error,
        ))
    })?;
    let reservation = MemoryConsumer::new(CUGRAPH_HOST_OUTPUT_MEMORY_CONSUMER)
        .with_can_spill(false)
        .register(task_context.memory_pool());
    let mut staged_bytes = 0usize;
    let mut batches = Vec::new();
    while let Some(prepared) = cursor.prepare_next_batch().map_err(|error| {
        if error.code() == QueryErrorCode::HostMaterializationLimitExceeded {
            return DataFusionError::from(
                Error::new(
                    ErrorCode::HostStagingLimitExceeded,
                    "algeon_datafusion.cugraph_sql.host_staging_limit",
                    "one cuGraph host-output row exceeded the configured staging limit",
                )
                .with_fact("boundary", "output")
                .with_fact("source_kind", "graph_result")
                .with_fact("staged_bytes", staged_bytes)
                .with_fact("max_staging_bytes", output_limit_bytes)
                .with_source(error),
            );
        }
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.host_output_cursor",
            "failed to size the next bounded cuGraph host-output batch",
            error,
        ))
    })? {
        let estimated_bytes = prepared.estimated_materialization_bytes();
        let Some(next_estimated_bytes) = staged_bytes.checked_add(estimated_bytes) else {
            return Err(host_staging_limit_error(
                "graph_result",
                "output",
                staged_bytes,
                estimated_bytes,
                output_limit_bytes,
            ));
        };
        if let Some(evidence) = host_evidence {
            evidence.check_output_growth("output", staged_bytes, estimated_bytes)?;
        } else if next_estimated_bytes > output_limit_bytes {
            return Err(host_staging_limit_error(
                "graph_result",
                "output",
                staged_bytes,
                estimated_bytes,
                output_limit_bytes,
            ));
        }
        reservation.try_grow(estimated_bytes).map_err(|source| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::MemoryReservationFailed,
                    "algeon_datafusion.cugraph_sql.host_staging_reservation",
                    "DataFusion host memory pool denied cuGraph output staging",
                )
                .with_fact("boundary", "output")
                .with_fact("requested_bytes", estimated_bytes)
                .with_fact("staged_bytes", staged_bytes)
                .with_source(source),
            )
        })?;
        let batch = cursor.materialize_prepared_batch(prepared).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cugraph_sql.host_output_cursor",
                "failed to materialize a reserved cuGraph host-output batch",
                error,
            ))
        })?;
        let actual_bytes = get_record_batch_memory_size(&batch);
        let surplus = estimated_bytes.checked_sub(actual_bytes).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::CudfInvariantViolation,
                    "algeon_datafusion.cugraph_sql.host_output_cursor",
                    "materialized cuGraph Arrow output exceeded its pre-copy estimate",
                )
                .with_fact("estimated_bytes", estimated_bytes)
                .with_fact("actual_bytes", actual_bytes),
            )
        })?;
        if surplus != 0 {
            reservation.shrink(surplus);
        }
        staged_bytes = staged_bytes.checked_add(actual_bytes).ok_or_else(|| {
            host_staging_limit_error("graph_result", "output", staged_bytes, actual_bytes, output_limit_bytes)
        })?;
        metrics.record_device_to_host_materialization(&batch);
        batches.push(batch);
    }
    Ok(EagerCugraphHostOutput::from_shared_parts_with_evidence(
        schema,
        Arc::new(batches),
        Arc::new(reservation),
        host_evidence.copied(),
    ))
}

pub(super) fn prepare_cugraph_source_metadata(
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> BoxFuture<'static, Result<Arc<dyn ExecutionPlan>>> {
    Box::pin(async move {
        if let Some(native) = plan.downcast_ref::<AlgeonNativeExec>() {
            return Ok(
                Arc::new(native.prepare_lakehouse_metadata(task_context.as_ref()).await) as Arc<dyn ExecutionPlan>
            );
        }
        if let Some(native) = plan.downcast_ref::<AlgeonNativeComposedExec>() {
            return Ok(
                Arc::new(native.prepare_lakehouse_metadata(Arc::clone(&task_context)).await?) as Arc<dyn ExecutionPlan>
            );
        }

        let children = plan.children();
        if children.is_empty() {
            return Ok(plan);
        }
        let mut prepared_children = Vec::with_capacity(children.len());
        let mut changed = false;
        for child in children {
            let prepared = prepare_cugraph_source_metadata(Arc::clone(child), Arc::clone(&task_context)).await?;
            changed |= !Arc::ptr_eq(child, &prepared);
            prepared_children.push(prepared);
        }
        if !changed {
            return Ok(plan);
        }
        let original = Arc::clone(&plan);
        Ok(plan
            .replace_children(prepared_children, ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
            .unwrap_or(original))
    })
}

pub(super) async fn execute_edge_source_to_gpu(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    admitted_composed_plan: Option<QueryPlan>,
) -> Result<GpuDataFrame> {
    if native_gpu_resident_producer(plan.as_ref()).is_some() {
        let (_, handle) =
            execute_gpu_resident_producer_with_context(plan.as_ref(), context, 0, Arc::clone(&task_context)).await?;
        return Ok(handle.into_gpu_frame());
    }
    if let Some(native) = as_algeon_native_exec(plan.as_ref()) {
        let native_plan =
            resolve_query_plan(native.execution_config(), native.native_plan().clone(), task_context.as_ref())
                .await
                .clone_with_gpu_resident_sink();
        let outcome = execute_query_plan(context, QueryExecutionRequest::new(&native_plan));
        let output = finalize_graph_query_outcome(
            metrics,
            outcome,
            "algeon_datafusion.cugraph_sql.edge_source_native",
            "native cuGraph edge-source execution failed",
        )?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.edge_source_native",
                "native cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }
    if let Some(native) = as_algeon_native_composed_exec(plan.as_ref()) {
        let native_plan = match admitted_composed_plan {
            Some(plan) => plan,
            None => resolve_query_plan(native.execution_config(), native.native_plan().clone(), task_context.as_ref())
                .await
                .clone_with_gpu_resident_sink(),
        };
        let runtime_occurrence_ids = native
            .gpu_inputs()
            .iter()
            .map(|input| {
                gpu_resident_producer_input_name(input.as_ref()).ok_or_else(|| {
                    DataFusionError::from(
                        Error::new(
                            ErrorCode::InvalidExecutionPlan,
                            "algeon_datafusion.cugraph_sql.edge_source_composed",
                            "the composed cuGraph edge source is missing a top-level producer occurrence identity",
                        )
                        .with_fact("reason", "composed_admission_occurrence_missing"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        context.validate_admitted_composed_attempt(&native_plan, &runtime_occurrence_ids).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cugraph_sql.edge_source_composed.admission",
                "the admitted composed edge source does not match runtime execution",
                error,
            ))
        })?;
        let mut gpu_inputs = GpuResidentInputMap::new();
        for child in native.gpu_inputs() {
            let Some(producer) = native_gpu_resident_producer(child.as_ref()) else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "algeon_datafusion.cugraph_sql.edge_source_composed",
                    "native composed cuGraph edge-source child was not a GPU-resident producer",
                )
                .with_fact("child_name", child.name())
                .into());
            };
            let (input_name, handle) =
                producer.execute_gpu_result_with_context(context, 0, Arc::clone(&task_context)).await?;
            gpu_inputs.insert(input_name, handle);
        }
        let outcome =
            execute_query_plan(context, QueryExecutionRequest::new(&native_plan).with_gpu_inputs(&gpu_inputs));
        let output = finalize_graph_query_outcome(
            metrics,
            outcome,
            "algeon_datafusion.cugraph_sql.edge_source_composed",
            "native composed cuGraph edge-source execution failed",
        )?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "algeon_datafusion.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }

    Err(Error::new(
        ErrorCode::InvalidExecutionPlan,
        "algeon_datafusion.cugraph_sql.edge_source_host_import",
        "host cuGraph inputs must be staged before native context acquisition",
    )
    .with_fact("source_plan", plan.name())
    .into())
}

pub(super) struct CugraphGpuFrame {
    pub(super) schema: SchemaRef,
    pub(super) frame: GpuDataFrame,
}

pub(super) fn record_batches_to_gpu(
    context: &ExecutionContext,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<GpuDataFrame> {
    let runtime = counted_cudf_runtime(
        context,
        "algeon_datafusion.cugraph_sql.edge_source_host_import.runtime",
        "query service closed before cuGraph edge import acquired its runtime",
    )?;
    let Some(first_batch) = batches.first() else {
        let arrays = schema.fields().iter().map(|field| new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema, arrays).map_err(DataFusionError::from)?;
        return DataFrame::from_record_batch(&runtime, &batch).map(GpuDataFrame::new).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "algeon_datafusion.cugraph_sql.edge_source_host_import",
                "failed to import empty host Arrow edge-source batch into cudf",
                cudf_query_error(
                    "algeon_datafusion.cugraph_sql.edge_source_host_import",
                    "failed to import empty host Arrow edge-source batch into cudf",
                    error,
                ),
            ))
        });
    };
    let imported = DataFrame::from_record_batches(&runtime, batches).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "algeon_datafusion.cugraph_sql.edge_source_host_import",
            "failed to import host Arrow edge-source batches into cudf",
            cudf_query_error(
                "algeon_datafusion.cugraph_sql.edge_source_host_import",
                "failed to import host Arrow edge-source batches into cudf",
                error,
            ),
        ))
    })?;
    if imported.num_columns() != first_batch.num_columns() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "algeon_datafusion.cugraph_sql.edge_source_host_import",
            "cuGraph edge-source import changed column count",
        )
        .with_fact("expected_column_count", first_batch.num_columns())
        .with_fact("actual_column_count", imported.num_columns())
        .into());
    }
    Ok(GpuDataFrame::new(imported))
}
