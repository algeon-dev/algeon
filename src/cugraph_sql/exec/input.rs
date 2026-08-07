use super::*;

const CUGRAPH_HOST_INPUT_MEMORY_CONSUMER: &str = "CugraphAlgoExec[host-input-staging]";
const CUGRAPH_HOST_OUTPUT_MEMORY_CONSUMER: &str = "CugraphAlgoExec[host-output-staging]";

#[derive(Debug)]
pub(super) enum StagedCugraphSource {
    Native { schema: SchemaRef, plan: Arc<dyn ExecutionPlan> },
    Host { schema: SchemaRef, batches: Vec<RecordBatch> },
}

#[derive(Debug)]
pub(super) struct StagedCugraphInputs {
    edge: StagedCugraphSource,
    logical_literal_source: Option<StagedCugraphSource>,
    predicates: Vec<StagedCugraphSource>,
    reservation: MemoryReservation,
}

pub(super) struct UploadedCugraphInputs {
    pub(super) edge_schema: SchemaRef,
    pub(super) edge_frame: GpuDataFrame,
    pub(super) logical_literal_source: Option<CugraphGpuFrame>,
    pub(super) predicate_frames: Vec<CugraphGpuFrame>,
}

pub(super) async fn stage_cugraph_inputs(
    spec: &CugraphAlgorithmSpec,
    edge_source: Arc<dyn ExecutionPlan>,
    predicate_sources: Vec<Arc<dyn ExecutionPlan>>,
    task_context: Arc<TaskContext>,
    cancellation: &nexus_query_engine::runtime::ExecutionCancellation,
    max_staging_bytes: usize,
) -> Result<StagedCugraphInputs> {
    cancellation.check("datafusion_nexus.cugraph_sql.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.host_input_staging",
            "cuGraph host-input staging was cancelled before execution",
            error,
        ))
    })?;
    let mut reservation = MemoryConsumer::new(CUGRAPH_HOST_INPUT_MEMORY_CONSUMER)
        .with_can_spill(false)
        .register(task_context.memory_pool());
    let mut staged_bytes = 0usize;
    let edge = stage_cugraph_source(
        "edge",
        edge_source,
        Arc::clone(&task_context),
        cancellation,
        max_staging_bytes,
        &mut staged_bytes,
        &mut reservation,
    )
    .await?;
    let logical_literal_source = if let Some(batch) = bfs_literal_string_source_batch(spec)? {
        reserve_host_staging_batch(
            "source_vertices",
            "input",
            &batch,
            max_staging_bytes,
            &mut staged_bytes,
            &mut reservation,
        )?;
        let schema = batch.schema();
        Some(StagedCugraphSource::Host { schema, batches: vec![batch] })
    } else {
        None
    };
    let mut predicates = Vec::with_capacity(predicate_sources.len());
    for source in predicate_sources {
        predicates.push(
            stage_cugraph_source(
                "side_input",
                source,
                Arc::clone(&task_context),
                cancellation,
                max_staging_bytes,
                &mut staged_bytes,
                &mut reservation,
            )
            .await?,
        );
    }
    Ok(StagedCugraphInputs { edge, logical_literal_source, predicates, reservation })
}

async fn stage_cugraph_source(
    source_kind: &'static str,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    cancellation: &nexus_query_engine::runtime::ExecutionCancellation,
    max_staging_bytes: usize,
    staged_bytes: &mut usize,
    reservation: &mut MemoryReservation,
) -> Result<StagedCugraphSource> {
    let schema = plan.schema();
    if as_nexus_native_exec(plan.as_ref()).is_some() || as_nexus_native_composed_exec(plan.as_ref()).is_some() {
        return Ok(StagedCugraphSource::Native { schema, plan });
    }
    if plan_contains_nexus_native(plan.as_ref()) {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.host_input_staging",
            "cuGraph host staging cannot hide a native GPU child below a CPU execution wrapper",
        )
        .with_fact("source_kind", source_kind)
        .with_fact("source_plan", plan.name())
        .into());
    }

    let partition_count = plan.properties().output_partitioning().partition_count();
    let mut batches = Vec::new();
    for partition in 0..partition_count {
        check_host_staging_cancellation(cancellation)?;
        let mut stream = plan.execute(partition, Arc::clone(&task_context))?;
        while let Some(batch) = stream.try_next().await? {
            check_host_staging_cancellation(cancellation)?;
            reserve_host_staging_batch(source_kind, "input", &batch, max_staging_bytes, staged_bytes, reservation)?;
            batches.push(batch);
        }
    }
    Ok(StagedCugraphSource::Host { schema, batches })
}

fn check_host_staging_cancellation(cancellation: &nexus_query_engine::runtime::ExecutionCancellation) -> Result<()> {
    cancellation.check("datafusion_nexus.cugraph_sql.host_input_staging").map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.host_input_staging",
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
    staged_bytes: &mut usize,
    reservation: &mut MemoryReservation,
) -> Result<()> {
    let batch_bytes = get_record_batch_memory_size(batch);
    let Some(next_staged_bytes) = staged_bytes.checked_add(batch_bytes) else {
        return Err(host_staging_limit_error(source_kind, boundary, *staged_bytes, batch_bytes, max_staging_bytes));
    };
    if next_staged_bytes > max_staging_bytes {
        return Err(host_staging_limit_error(source_kind, boundary, *staged_bytes, batch_bytes, max_staging_bytes));
    }
    reservation.try_grow(batch_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied cuGraph staging",
            )
            .with_fact("boundary", boundary)
            .with_fact("source_kind", source_kind)
            .with_fact("requested_bytes", batch_bytes)
            .with_fact("staged_bytes", *staged_bytes)
            .with_source(source),
        )
    })?;
    *staged_bytes = next_staged_bytes;
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
        "datafusion_nexus.cugraph_sql.host_staging_limit",
        "cuGraph GPU-island host staging exceeded its configured cumulative byte limit",
    )
    .with_fact("boundary", boundary)
    .with_fact("source_kind", source_kind)
    .with_fact("staged_bytes", staged_bytes)
    .with_fact("next_batch_bytes", next_batch_bytes)
    .with_fact("max_staging_bytes", max_staging_bytes)
    .into()
}

impl StagedCugraphInputs {
    pub(super) async fn upload(
        self,
        context: &ExecutionContext,
        metrics: &mut GraphExecutionMetrics<'_>,
        task_context: Arc<TaskContext>,
    ) -> Result<UploadedCugraphInputs> {
        let Self { edge, logical_literal_source, predicates, reservation } = self;
        let edge_schema = edge.schema();
        let edge_frame = edge.upload(context, metrics, Arc::clone(&task_context)).await?;
        let logical_literal_source = match logical_literal_source {
            Some(source) => {
                let schema = source.schema();
                let frame = source.upload(context, metrics, Arc::clone(&task_context)).await?;
                Some(CugraphGpuFrame { schema, frame })
            },
            None => None,
        };
        let mut predicate_frames = Vec::with_capacity(predicates.len());
        for source in predicates {
            let schema = source.schema();
            let frame = source.upload(context, metrics, Arc::clone(&task_context)).await?;
            predicate_frames.push(CugraphGpuFrame { schema, frame });
        }
        drop(reservation);
        Ok(UploadedCugraphInputs { edge_schema, edge_frame, logical_literal_source, predicate_frames })
    }
}

impl StagedCugraphSource {
    fn schema(&self) -> SchemaRef {
        match self {
            Self::Native { schema, .. } | Self::Host { schema, .. } => Arc::clone(schema),
        }
    }

    async fn upload(
        self,
        context: &ExecutionContext,
        metrics: &mut GraphExecutionMetrics<'_>,
        task_context: Arc<TaskContext>,
    ) -> Result<GpuDataFrame> {
        match self {
            Self::Native { plan, .. } => execute_edge_source_to_gpu(context, metrics, plan, task_context).await,
            Self::Host { schema, batches } => record_batches_to_gpu(context, schema, &batches),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct EagerCugraphHostOutput {
    schema: SchemaRef,
    batches: Arc<Vec<RecordBatch>>,
    next_batch: usize,
    reservation: Arc<MemoryReservation>,
}

impl EagerCugraphHostOutput {
    pub(super) fn into_parts(self) -> (Arc<Vec<RecordBatch>>, Arc<MemoryReservation>) {
        (self.batches, self.reservation)
    }

    pub(super) fn shared_parts(&self) -> (Arc<Vec<RecordBatch>>, Arc<MemoryReservation>) {
        (Arc::clone(&self.batches), Arc::clone(&self.reservation))
    }

    pub(super) fn from_parts(schema: SchemaRef, batches: Vec<RecordBatch>, reservation: MemoryReservation) -> Self {
        Self::from_shared_parts(schema, Arc::new(batches), Arc::new(reservation))
    }

    pub(super) fn from_shared_parts(
        schema: SchemaRef,
        batches: Arc<Vec<RecordBatch>>,
        reservation: Arc<MemoryReservation>,
    ) -> Self {
        Self { schema, batches, next_batch: 0, reservation }
    }

    pub(super) fn published_result(&self) -> GraphBfsPublishedResult {
        let (batches, reservation) = self.shared_parts();
        GraphBfsPublishedResult::new(batches, reservation)
    }

    pub(super) fn from_published_result(schema: SchemaRef, result: &GraphBfsPublishedResult) -> Option<Self> {
        let (batches, reservation) = result.host_shared_parts()?;
        Some(Self::from_shared_parts(schema, batches, reservation))
    }

    pub(super) fn record_datafusion_output(self, metrics: &NexusNativeMetrics) -> Self {
        let schema = Arc::clone(&self.schema);
        let (batches, reservation) = self.into_parts();
        let batches = batches.iter().cloned().map(|batch| batch.record_output(&metrics.baseline)).collect();
        Self::from_shared_parts(schema, Arc::new(batches), reservation)
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
    cancellation: nexus_query_engine::runtime::ExecutionCancellation,
    max_staging_bytes: usize,
    metrics: &mut GraphExecutionMetrics<'_>,
) -> Result<EagerCugraphHostOutput> {
    let schema = Arc::clone(handle.schema());
    let max_materialization_bytes = max_staging_bytes.min(HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES);
    let target_materialization_bytes =
        max_staging_bytes.min(HostBatchCursorConfig::DEFAULT_TARGET_MATERIALIZATION_BYTES);
    let config = HostBatchCursorConfig {
        target_materialization_bytes,
        max_materialization_bytes,
        ..HostBatchCursorConfig::default()
    };
    let mut cursor = handle.into_host_batch_cursor(cancellation, config).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.host_output_cursor",
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
        if error.code() == nexus_query_engine::error::QueryErrorCode::HostMaterializationLimitExceeded {
            return DataFusionError::from(
                Error::new(
                    ErrorCode::HostStagingLimitExceeded,
                    "datafusion_nexus.cugraph_sql.host_staging_limit",
                    "one cuGraph host-output row exceeded the configured staging limit",
                )
                .with_fact("boundary", "output")
                .with_fact("source_kind", "graph_result")
                .with_fact("staged_bytes", staged_bytes)
                .with_fact("max_staging_bytes", max_staging_bytes)
                .with_source(error),
            );
        }
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.host_output_cursor",
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
                max_staging_bytes,
            ));
        };
        if next_estimated_bytes > max_staging_bytes {
            return Err(host_staging_limit_error(
                "graph_result",
                "output",
                staged_bytes,
                estimated_bytes,
                max_staging_bytes,
            ));
        }
        reservation.try_grow(estimated_bytes).map_err(|source| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::MemoryReservationFailed,
                    "datafusion_nexus.cugraph_sql.host_staging_reservation",
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
                "datafusion_nexus.cugraph_sql.host_output_cursor",
                "failed to materialize a reserved cuGraph host-output batch",
                error,
            ))
        })?;
        let actual_bytes = get_record_batch_memory_size(&batch);
        let surplus = estimated_bytes.checked_sub(actual_bytes).ok_or_else(|| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::CudfInvariantViolation,
                    "datafusion_nexus.cugraph_sql.host_output_cursor",
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
            host_staging_limit_error("graph_result", "output", staged_bytes, actual_bytes, max_staging_bytes)
        })?;
        metrics.record_device_to_host_materialization(&batch);
        batches.push(batch);
    }
    Ok(EagerCugraphHostOutput::from_parts(schema, batches, reservation))
}

pub(super) fn prepare_cugraph_source_metadata(
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> BoxFuture<'static, Result<Arc<dyn ExecutionPlan>>> {
    Box::pin(async move {
        if let Some(native) = plan.downcast_ref::<NexusNativeExec>() {
            return Ok(
                Arc::new(native.prepare_lakehouse_metadata(task_context.as_ref()).await) as Arc<dyn ExecutionPlan>
            );
        }
        if let Some(native) = plan.downcast_ref::<NexusNativeComposedExec>() {
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
        Ok(plan.with_new_children(prepared_children).unwrap_or(original))
    })
}

pub(super) async fn execute_edge_source_to_gpu(
    context: &ExecutionContext,
    metrics: &mut GraphExecutionMetrics<'_>,
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
) -> Result<GpuDataFrame> {
    if let Some(native) = as_nexus_native_exec(plan.as_ref()) {
        let native_plan = crate::native_exec::lakehouse::resolve_query_plan(
            native.execution_config(),
            native.native_plan().clone(),
            task_context.as_ref(),
        )
        .await
        .clone_with_gpu_resident_sink();
        let outcome = execute_query_plan_observed(context, &native_plan, metrics.runtime_observation_config());
        let output = finalize_graph_query_outcome(
            metrics,
            outcome,
            "datafusion_nexus.cugraph_sql.edge_source_native",
            "native cuGraph edge-source execution failed",
        )?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_native",
                "native cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }
    if let Some(native) = as_nexus_native_composed_exec(plan.as_ref()) {
        let native_plan = crate::native_exec::lakehouse::resolve_query_plan(
            native.execution_config(),
            native.native_plan().clone(),
            task_context.as_ref(),
        )
        .await
        .clone_with_gpu_resident_sink();
        let mut gpu_inputs = GpuResidentInputMap::new();
        for child in native.gpu_inputs() {
            let Some(producer) = native_gpu_resident_producer(child.as_ref()) else {
                return Err(Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    "datafusion_nexus.cugraph_sql.edge_source_composed",
                    "native composed cuGraph edge-source child was not a GPU-resident producer",
                )
                .with_fact("child_name", child.name())
                .into());
            };
            let (input_name, handle) =
                producer.execute_gpu_result_with_context(context, 0, Arc::clone(&task_context)).await?;
            gpu_inputs.insert(input_name, handle);
        }
        let outcome = execute_query_plan_with_gpu_inputs_observed(
            context,
            &native_plan,
            &gpu_inputs,
            metrics.runtime_observation_config(),
        );
        let output = finalize_graph_query_outcome(
            metrics,
            outcome,
            "datafusion_nexus.cugraph_sql.edge_source_composed",
            "native composed cuGraph edge-source execution failed",
        )?;
        let QueryOutput::GpuResident(handle) = output else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                "datafusion_nexus.cugraph_sql.edge_source_composed",
                "native composed cuGraph edge source did not produce a GPU-resident result",
            )
            .into());
        };
        return Ok(handle.into_gpu_frame());
    }

    Err(Error::new(
        ErrorCode::InvalidExecutionPlan,
        "datafusion_nexus.cugraph_sql.edge_source_host_import",
        "host cuGraph inputs must be staged before native context acquisition",
    )
    .with_fact("source_plan", plan.name())
    .into())
}

pub(super) struct CugraphGpuFrame {
    pub(super) schema: SchemaRef,
    pub(super) frame: GpuDataFrame,
}

pub(super) fn cudf_logical_mapping_error(error: cudf::Error) -> DataFusionError {
    Error::from_native(
        "datafusion_nexus.cugraph_sql.logical_mapping",
        "cuGraph logical BFS GPU mapping failed",
        nexus_query_engine::error::QueryError::from_cudf(
            "datafusion_nexus.cugraph_sql.logical_mapping",
            "cuGraph logical BFS GPU mapping failed",
            error,
        ),
    )
    .into()
}

pub(super) fn cudf_series(frame: &DataFrame, column: &str) -> Result<Series> {
    frame.series(column).map_err(cudf_logical_mapping_error)
}

pub(super) fn record_batches_to_gpu(
    context: &ExecutionContext,
    schema: SchemaRef,
    batches: &[RecordBatch],
) -> Result<GpuDataFrame> {
    let runtime = counted_cudf_runtime(
        context,
        "datafusion_nexus.cugraph_sql.edge_source_host_import.runtime",
        "query service closed before cuGraph edge import acquired its runtime",
    )?;
    let Some(first_batch) = batches.first() else {
        let arrays =
            schema.fields().iter().map(|field| arrow::array::new_empty_array(field.data_type())).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(schema, arrays).map_err(DataFusionError::from)?;
        return cudf::DataFrame::from_record_batch(&runtime, &batch).map(GpuDataFrame::new).map_err(|error| {
            DataFusionError::from(Error::from_native(
                "datafusion_nexus.cugraph_sql.edge_source_host_import",
                "failed to import empty host Arrow edge-source batch into cudf",
                nexus_query_engine::error::QueryError::from_cudf(
                    "datafusion_nexus.cugraph_sql.edge_source_host_import",
                    "failed to import empty host Arrow edge-source batch into cudf",
                    error,
                ),
            ))
        });
    };
    let imported = cudf::DataFrame::from_record_batches(&runtime, batches).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "failed to import host Arrow edge-source batches into cudf",
            nexus_query_engine::error::QueryError::from_cudf(
                "datafusion_nexus.cugraph_sql.edge_source_host_import",
                "failed to import host Arrow edge-source batches into cudf",
                error,
            ),
        ))
    })?;
    if imported.num_columns() != first_batch.num_columns() {
        return Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_nexus.cugraph_sql.edge_source_host_import",
            "cuGraph edge-source import changed column count",
        )
        .with_fact("expected_column_count", first_batch.num_columns())
        .with_fact("actual_column_count", imported.num_columns())
        .into());
    }
    Ok(GpuDataFrame::new(imported))
}
