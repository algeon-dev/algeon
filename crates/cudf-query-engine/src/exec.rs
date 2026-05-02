use std::time::Instant;

use arrow::{array::Array, datatypes::DataType as ArrowDataType, record_batch::RecordBatch};
use cudf::{
    AggSpec as CudfAggSpec, DataFrame, ExecutionOverrides, ExecutionPlacement, GroupByAgg,
    SortOptions as CudfSortOptions, TypeId,
    frame::dataframe::{InfoDetail, ParquetReadOptions},
    ops::{col, lit},
};

use crate::{
    chunk::{SourceChunk, SourceChunkPlan, SourceChunkPlanningPolicy, plan_local_parquet_source_chunks},
    error::{QueryError, QueryErrorCode, Result},
    expr::{AggregateFunction, AggregateSpec, BinaryOp, Expr, LimitSpec, LiteralValue, SortSpec},
    metrics::QueryMetrics,
    plan::{AggregateNode, FilterNode, LimitNode, NodeId, PlanNode, ProjectionNode, QueryPlan, ResultSink, SortNode},
    runtime::{ExecutionContext, MemoryControlPolicy, MemoryReservationRequest, ReservationKind, RetryFacts},
    source::{LocalParquetFile, LocalParquetSource},
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

pub enum QueryOutput {
    GpuResident(GpuDataFrame),
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

    pub fn as_gpu_resident(&self) -> Option<&GpuDataFrame> {
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

pub fn execute_query_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<QueryExecutionResult> {
    let mut metrics =
        QueryMetrics { phase_c_memory_control_policy: context.memory_control_policy(), ..Default::default() };
    let output = if context.memory_control_policy().is_bounded() {
        execute_bounded_query_plan(context, plan, &mut metrics)?
    } else {
        let frame = execute_node(context, plan, plan.sink().input(), &mut metrics)?;
        match plan.sink() {
            ResultSink::GpuResident { .. } => QueryOutput::GpuResident(frame),
            ResultSink::HostArrow { .. } => QueryOutput::HostArrow(materialize_to_host(frame, &mut metrics)?),
        }
    };
    Ok(QueryExecutionResult::new(output, metrics))
}

pub fn validate_local_parquet_source(source: &LocalParquetSource) -> Result<()> {
    if source.files.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source must contain at least one file",
        )
        .with_fact("reason", "empty_files"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        if file.path.as_os_str().is_empty() {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.local_parquet_source.validate",
                "local parquet source path must not be empty",
            )
            .with_fact("file_index", file_index));
        }
    }

    if source.read_columns.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.local_parquet_source.validate",
            "local parquet source must name at least one required read column",
        )
        .with_fact("reason", "empty_read_columns"));
    }

    for (file_index, file) in source.files.iter().enumerate() {
        for (index, row_group) in file.row_groups.iter().flatten().enumerate() {
            if i32::try_from(*row_group).is_err() {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.local_parquet_source.validate",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", file.path.display())
                .with_fact("file_index", file_index)
                .with_fact("row_group_position", index)
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX));
            }
        }
    }

    Ok(())
}

pub fn read_local_parquet_source(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    validate_local_parquet_source(source)?;

    let columns = source.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut frames = Vec::with_capacity(source.files.len());
    for file in &source.files {
        frames.push(read_local_parquet_file(context, file, &columns, metrics)?);
    }

    let mut frames = frames.into_iter();
    let frame = frames.next().expect("validated local parquet source contains at least one file");
    let other_frames = frames.collect::<Vec<_>>();
    if other_frames.is_empty() {
        return Ok(GpuDataFrame::new(frame));
    }

    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.concat",
            "failed to concatenate native local parquet files",
            error,
        )
        .with_fact("file_count", source.file_count())
        .with_fact("read_columns", source.read_column_count())
    })?;
    metrics.operator_time += timer.elapsed();

    Ok(GpuDataFrame::new(output))
}

fn read_local_parquet_file(
    context: &ExecutionContext,
    file: &LocalParquetFile,
    columns: &[&str],
    metrics: &mut QueryMetrics,
) -> Result<DataFrame> {
    let row_groups = cudf_row_groups(file)?;
    let read_timer = Instant::now();
    let options = if let Some(row_groups) = &row_groups {
        ParquetReadOptions::builder()
            .filepath(&file.path)
            .columns(columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&file.path).columns(columns).with_runtime(context.runtime()).build()
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_source.read",
            "failed to read native local parquet source with cudf",
            error,
        )
        .with_fact("path", file.path.display())
        .with_fact("read_columns", columns.len())
        .with_fact("row_groups_selected", file.selected_row_group_count())
    })?;
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.parquet_files += 1;
    metrics.parquet_bytes_known += file.file_size_bytes.unwrap_or(0);
    metrics.row_groups_selected += file.selected_row_group_count() as u64;
    metrics.read_columns += columns.len() as u64;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(frame)
}

fn cudf_row_groups(file: &LocalParquetFile) -> Result<Option<Vec<i32>>> {
    let Some(row_groups) = &file.row_groups else {
        return Ok(None);
    };

    if row_groups.is_empty() {
        return Ok(None);
    }

    row_groups
        .iter()
        .map(|row_group| {
            i32::try_from(*row_group).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.local_parquet_source.row_groups",
                    "local parquet row group index exceeds the cudf reader range",
                )
                .with_fact("path", file.path.display())
                .with_fact("row_group", row_group)
                .with_fact("max_row_group", i32::MAX)
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn execute_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let node = plan.node(id).ok_or_else(|| invalid_node_reference(id, "query plan references a missing node"))?;

    let frame = match &node.kind {
        PlanNode::Source(source) => match source {
            crate::source::SourceNode::LocalParquet(source) => read_local_parquet_source(context, source, metrics),
        },
        PlanNode::Filter(filter) => {
            let input = execute_node(context, plan, filter.input, metrics)?;
            execute_filter(context, filter, input, metrics)
        },
        PlanNode::Projection(projection) => {
            let input = execute_node(context, plan, projection.input, metrics)?;
            execute_projection(context, projection, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Aggregate(aggregate) => {
            let input = execute_node(context, plan, aggregate.input, metrics)?;
            execute_aggregate(context, aggregate, input, &node.facts.output_schema, metrics)
        },
        PlanNode::Sort(sort) => {
            let input = execute_node(context, plan, sort.input, metrics)?;
            execute_sort(context, sort, input, metrics)
        },
        PlanNode::Limit(limit) => {
            let input = execute_node(context, plan, limit.input, metrics)?;
            execute_limit(context, limit, input, metrics)
        },
    }?;

    record_device_memory_observation(&frame, metrics)?;
    Ok(frame)
}

fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let bounded = BoundedPlan::analyze(plan)?;
    validate_local_parquet_source(bounded.source)?;
    bounded.validate_capabilities()?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let chunk_plan = plan_local_parquet_source_chunks(
        bounded.source_id,
        bounded.source,
        SourceChunkPlanningPolicy::from(context.memory_control_policy()),
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.bounded.plan",
            "bounded execution requires at least one planned source chunk",
        )
        .with_fact("source_node_id", bounded.source_id));
    }
    record_bounded_parquet_source_metrics(bounded.source, &chunk_plan, metrics);

    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut partials = Vec::with_capacity(chunk_plan.chunks.len());
    for chunk in &chunk_plan.chunks {
        partials.extend(execute_bounded_chunk_with_retry(context, &bounded, chunk, &mut reservations, metrics)?);
    }

    let frame = execute_final_aggregate_merge(context, &bounded, partials, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => Ok(QueryOutput::GpuResident(frame.frame)),
        ResultSink::HostArrow { .. } => {
            let output = materialize_bounded_to_host(frame, plan.sink().input(), &mut reservations, metrics)?;
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_chunk_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, min_retry_chunk_row_groups, .. } =
        context.memory_control_policy()
    else {
        unreachable!("execute_bounded_chunk_with_retry is called only for bounded policies");
    };

    execute_bounded_chunk_retry_attempt(
        context,
        bounded,
        chunk.clone(),
        reservations,
        metrics,
        1,
        max_retry_attempts.max(1),
        min_retry_chunk_row_groups.max(1),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_bounded_chunk_retry_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
    min_retry_chunk_row_groups: usize,
) -> Result<Vec<CommittedPartial>> {
    match execute_bounded_chunk_attempt(context, bounded, &chunk, reservations, metrics, attempt, max_attempts) {
        Ok(partial) => Ok(vec![partial]),
        Err(error) if is_retryable_chunk_error(&error) => {
            record_chunk_retry_error(&error, metrics);
            if attempt < max_attempts
                && let Some((left, right)) = split_retryable_chunk(&chunk, min_retry_chunk_row_groups)
            {
                metrics.record_source_chunk_retry();
                let mut partials = execute_bounded_chunk_retry_attempt(
                    context,
                    bounded,
                    left,
                    reservations,
                    metrics,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?;
                partials.extend(execute_bounded_chunk_retry_attempt(
                    context,
                    bounded,
                    right,
                    reservations,
                    metrics,
                    attempt + 1,
                    max_attempts,
                    min_retry_chunk_row_groups,
                )?);
                return Ok(partials);
            }

            Err(chunk_retry_facts(&chunk, metrics).attach_to_error(error))
        },
        Err(error) => Err(error),
    }
}

fn execute_bounded_chunk_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    chunk: &SourceChunk,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    attempt: usize,
    max_attempts: usize,
) -> Result<CommittedPartial> {
    metrics.record_source_chunk_attempt();

    let source_request = chunk_reservation_request(
        ReservationKind::SourceRead,
        "source_chunk_read",
        chunk_requested_bytes(context.memory_control_policy(), chunk),
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let source_reservation = reservations.try_reserve(source_request, metrics)?;
    let mut frame = match read_local_parquet_chunk(context, chunk, metrics) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(error);
        },
    };
    let source_observed_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(error);
        },
    };

    let operator_request = chunk_reservation_request(
        ReservationKind::ChunkOperator,
        "chunk_operator",
        source_observed_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let operator_reservation = match reservations.try_reserve(operator_request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(error);
        },
    };

    let chunk_operator_timer = Instant::now();
    let chunk_output = (|| {
        if let Some(filter) = bounded.filter {
            frame = execute_filter(context, filter, frame, metrics)?;
        }
        if let Some((projection, output_schema)) = bounded.projection {
            frame = execute_projection(context, projection, frame, output_schema, metrics)?;
        }
        execute_aggregate(context, bounded.aggregate.0, frame, bounded.aggregate.1, metrics)
    })();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation);
            reservations.release(source_reservation);
            return Err(error);
        },
    };

    reservations.release(operator_reservation);
    reservations.release(source_reservation);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let partial_request = chunk_reservation_request(
        ReservationKind::PartialState,
        "partial_state",
        partial_bytes,
        reservations.budget(),
        chunk,
        attempt,
        max_attempts,
    );
    let partial_reservation = reservations.try_reserve(partial_request, metrics)?;
    metrics.record_source_chunk_commit(chunk.row_group_count() as u64);
    metrics.record_partial_aggregate(frame.num_rows() as u64);

    Ok(CommittedPartial { frame, reservation: partial_reservation })
}

fn read_local_parquet_chunk(
    context: &ExecutionContext,
    chunk: &SourceChunk,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_groups = if chunk.reads_all_row_groups {
        None
    } else {
        Some(
            chunk
                .row_group_indexes
                .iter()
                .map(|row_group| {
                    i32::try_from(*row_group).map_err(|_| {
                        QueryError::new(
                            QueryErrorCode::InvalidPlan,
                            "cudf_query_engine.local_parquet_chunk.row_groups",
                            "local parquet row group index exceeds the cudf reader range",
                        )
                        .with_fact("path", chunk.path.display())
                        .with_fact("row_group", row_group)
                        .with_fact("max_row_group", i32::MAX)
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        )
    };
    let columns = chunk.read_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let read_timer = Instant::now();
    let options = if let Some(row_groups) = &row_groups {
        ParquetReadOptions::builder()
            .filepath(&chunk.path)
            .columns(&columns)
            .row_groups(row_groups)
            .with_runtime(context.runtime())
            .build()
    } else {
        ParquetReadOptions::builder().filepath(&chunk.path).columns(&columns).with_runtime(context.runtime()).build()
    };

    let frame = DataFrame::read_parquet_with_options(options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.local_parquet_chunk.read",
            "failed to read bounded native local parquet source chunk with cudf",
            error,
        )
        .with_fact("path", chunk.path.display())
        .with_fact("chunk_index", chunk.chunk_index)
        .with_fact("file_index", chunk.file_index)
        .with_fact("read_columns", columns.len())
        .with_fact("row_groups", format_row_groups(chunk))
    })?;
    let read_elapsed = read_timer.elapsed();
    metrics.parquet_read_time += read_elapsed;
    metrics.source_read_time += read_elapsed;
    metrics.record_input_batch(frame.num_rows() as u64);
    Ok(GpuDataFrame::new(frame))
}

fn record_bounded_parquet_source_metrics(
    source: &LocalParquetSource,
    chunk_plan: &SourceChunkPlan,
    metrics: &mut QueryMetrics,
) {
    let known_file_bytes = source.files.iter().filter_map(|file| file.file_size_bytes).sum::<u64>();
    let planned_chunk_bytes = chunk_plan
        .chunks
        .iter()
        .map(|chunk| {
            chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(0)
        })
        .sum::<u64>();

    metrics.parquet_files = metrics.parquet_files.saturating_add(source.file_count() as u64);
    metrics.parquet_bytes_known = metrics.parquet_bytes_known.saturating_add(known_file_bytes.max(planned_chunk_bytes));
    metrics.row_groups_selected = metrics.row_groups_selected.saturating_add(chunk_plan.row_group_count() as u64);
    metrics.read_columns =
        metrics.read_columns.saturating_add((source.file_count() * source.read_column_count()) as u64);
}

#[derive(Clone, Copy)]
struct BoundedPlan<'a> {
    source_id: NodeId,
    source: &'a LocalParquetSource,
    filter: Option<&'a FilterNode>,
    projection: Option<(&'a ProjectionNode, &'a arrow::datatypes::SchemaRef)>,
    aggregate_id: NodeId,
    aggregate: (&'a AggregateNode, &'a arrow::datatypes::SchemaRef),
    sort: Option<&'a SortNode>,
    limit: Option<&'a LimitNode>,
}

impl<'a> BoundedPlan<'a> {
    fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let mut current = plan.sink().input();
        let limit = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Limit(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let sort = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Sort(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };
        if limit.is_some() && sort.is_none() {
            return Err(unsupported_bounded_shape(
                "bounded execution does not support limit without a preceding global sort",
                "limit_without_sort",
            ));
        }

        let aggregate_id = current;
        let aggregate_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing aggregate node"))?;
        let PlanNode::Aggregate(aggregate) = &aggregate_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently requires a grouped sum aggregate after source-local operators",
                "missing_grouped_aggregate",
            ));
        };
        current = aggregate.input;

        let projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => None,
            },
            None => return Err(invalid_node_reference(current, "bounded query plan references a missing input node")),
        };

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let source_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing source node"))?;
        let PlanNode::Source(crate::source::SourceNode::LocalParquet(source)) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };

        Ok(Self {
            source_id: current,
            source,
            filter,
            projection,
            aggregate_id,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            sort,
            limit,
        })
    }

    fn validate_capabilities(&self) -> Result<()> {
        if let Some(filter) = self.filter {
            lower_expr(&filter.predicate)?;
        }

        if let Some((projection, output_schema)) = self.projection {
            validate_projection_capability(projection, output_schema)?;
        }

        validate_aggregate_capability(self.aggregate.0, self.aggregate.1)?;
        validate_final_aggregate_capability(self.aggregate.0, self.aggregate.1)?;

        if let Some(sort) = self.sort {
            lower_sort_spec(&sort.spec)?;
        }
        if let Some(limit) = self.limit {
            topk_candidate_rows(limit.spec)?;
        }

        Ok(())
    }
}

fn validate_projection_capability(node: &ProjectionNode, output_schema: &arrow::datatypes::SchemaRef) -> Result<()> {
    let output_field_count = output_schema.fields().len();
    if node.expressions.len() != output_field_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection expression count must match output schema field count",
        )
        .with_fact("expression_count", node.expressions.len())
        .with_fact("output_field_count", output_field_count));
    }

    for expr in &node.expressions {
        lower_expr(expr)?;
    }

    Ok(())
}

fn validate_aggregate_capability(node: &AggregateNode, output_schema: &arrow::datatypes::SchemaRef) -> Result<()> {
    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )
}

fn validate_final_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    let aggregate = lower_final_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.final_merge.validate",
        "final aggregate output names must match output schema fields",
    )
}

fn validate_aggregate_output_names(
    aggregate: &LoweredAggregate,
    output_schema: &arrow::datatypes::SchemaRef,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if aggregate.output_names != expected_output_names {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("aggregate_output_names", aggregate.output_names.join(","))
            .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ActiveReservation {
    bytes: u64,
}

#[derive(Debug)]
struct ReservationTracker {
    budget_bytes: u64,
    reserved_bytes: u64,
}

impl ReservationTracker {
    fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes, reserved_bytes: 0 }
    }

    fn budget(&self) -> u64 {
        self.budget_bytes
    }

    fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let remaining = self.budget_bytes.saturating_sub(self.reserved_bytes);
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied(request.requested_bytes);
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes = self.reserved_bytes.saturating_add(request.requested_bytes);
        metrics.record_reservation_granted(request.requested_bytes, request.requested_bytes);
        Ok(ActiveReservation { bytes: request.requested_bytes })
    }

    fn release(&mut self, reservation: ActiveReservation) {
        self.reserved_bytes = self.reserved_bytes.saturating_sub(reservation.bytes);
    }
}

struct CommittedPartial {
    frame: GpuDataFrame,
    reservation: ActiveReservation,
}

fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(fallback)
}

fn chunk_reservation_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    chunk: &SourceChunk,
    attempt: usize,
    max_attempts: usize,
) -> MemoryReservationRequest {
    let mut request = MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(chunk.source_node_id)
        .with_stage(memory_boundary)
        .with_chunk_index(chunk.chunk_index)
        .with_file_index(chunk.file_index)
        .with_path(chunk.path.display())
        .with_attempt(attempt, max_attempts);
    if !chunk.reads_all_row_groups {
        request = request.with_row_groups(chunk.row_group_indexes.clone());
    }
    request
}

fn is_retryable_chunk_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_chunk_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn split_retryable_chunk(chunk: &SourceChunk, min_retry_chunk_row_groups: usize) -> Option<(SourceChunk, SourceChunk)> {
    if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= min_retry_chunk_row_groups {
        return None;
    }

    let split_at = chunk.row_group_indexes.len() / 2;
    if split_at == 0 || split_at == chunk.row_group_indexes.len() {
        return None;
    }

    let left_row_groups = chunk.row_group_indexes[..split_at].to_vec();
    let right_row_groups = chunk.row_group_indexes[split_at..].to_vec();
    let left = retry_child_chunk(chunk, left_row_groups);
    let right = retry_child_chunk(chunk, right_row_groups);
    Some((left, right))
}

fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
    let parent_row_group_count = parent.row_group_indexes.len();
    let known_row_group_bytes =
        proportional_bytes(parent.known_row_group_bytes, row_group_indexes.len(), parent_row_group_count);
    let known_file_bytes = if parent.known_row_group_bytes.is_some() {
        parent.known_file_bytes
    } else {
        proportional_bytes(parent.known_file_bytes, row_group_indexes.len(), parent_row_group_count)
    };
    let estimated_chunk_bytes = proportional_bytes(
        parent.estimated_chunk_bytes.or(parent.known_row_group_bytes).or(parent.known_file_bytes),
        row_group_indexes.len(),
        parent_row_group_count,
    );

    SourceChunk {
        source_node_id: parent.source_node_id,
        chunk_index: parent.chunk_index,
        file_index: parent.file_index,
        path: parent.path.clone(),
        row_group_indexes,
        reads_all_row_groups: false,
        known_file_bytes,
        known_row_group_bytes,
        estimated_chunk_bytes,
        read_columns: parent.read_columns.clone(),
        size_estimate_based: true,
    }
}

fn proportional_bytes(total_bytes: Option<u64>, child_row_groups: usize, parent_row_groups: usize) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

fn chunk_retry_facts(chunk: &SourceChunk, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if chunk.reads_all_row_groups || chunk.row_group_indexes.len() <= 1 {
        "fail_min_chunk"
    } else {
        "retry_attempts_exhausted"
    };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("source_chunk")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

fn execute_final_aggregate_merge(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: Vec<CommittedPartial>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_aggregate_merge is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let mut partials = partials;
    let mut merge_level = 1;

    while partials.len() > 1 {
        let mut merge_fan_in = partials.len();
        let mut attempt = 1;

        loop {
            match execute_final_merge_level_attempt(
                context,
                bounded,
                &partials,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    metrics.record_final_merge_level();
                    partials = commit_final_merge_level(partials, level_output, reservations);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_merge_error(&error) => {
                    record_final_merge_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    return Err(final_merge_retry_facts(merge_level, merge_fan_in, metrics).attach_to_error(error));
                },
                Err(error) => return Err(error),
            }
        }
    }

    Ok(partials.pop().expect("final merge loop leaves exactly one partial"))
}

#[allow(clippy::too_many_arguments)]
fn execute_final_merge_level_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<MergeLevelItem>> {
    metrics.record_final_merge_attempt();
    let mut output = Vec::with_capacity(partials.len().div_ceil(merge_fan_in));

    for (group_index, group) in partials.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(MergeLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_final_merge_group_attempt(context, bounded, group, attempt, max_attempts, reservations, metrics) {
            Ok(partial) => output.push(MergeLevelItem::Merged(partial)),
            Err(error) => {
                release_merge_level_items(output, reservations);
                return Err(error);
            },
        }
    }

    Ok(output)
}

fn execute_final_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes).sum::<u64>();
    let merge_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let merge_reservation = reservations.try_reserve(merge_request, metrics)?;

    let timer = Instant::now();
    let result = merge_final_aggregate_partials(context, bounded, partials);
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(error);
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

fn merge_final_aggregate_partials(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
) -> Result<GpuDataFrame> {
    let partial_count = partials.len();
    let first_frame = partials[0].frame.frame().clone();
    let other_frames = partials[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let concatenated = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.concat_partials",
            "failed to concatenate bounded native aggregate partials",
            error,
        )
        .with_fact("partial_aggregate_count", partial_count)
    })?;

    execute_final_aggregate(context, bounded, GpuDataFrame::new(concatenated))
}

enum MergeLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

fn commit_final_merge_level(
    partials: Vec<CommittedPartial>,
    level_output: Vec<MergeLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut partials = partials.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_partials = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            MergeLevelItem::Merged(partial) => next_partials.push(partial),
            MergeLevelItem::Carry { partial_index } => {
                let partial = partials[partial_index].take().expect("carry index references a source partial");
                next_partials.push(partial);
            },
        }
    }

    for partial in partials.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_partials
}

fn release_merge_level_items(items: Vec<MergeLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let MergeLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

fn is_retryable_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    if merge_fan_in <= 2 {
        return None;
    }

    Some(merge_fan_in.div_ceil(2).max(2))
}

fn final_merge_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if merge_fan_in <= 2 { "fail_min_merge_fan_in" } else { "retry_attempts_exhausted" };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_merge")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
        .with_merge_level(merge_level)
        .with_merge_fan_in(merge_fan_in)
}

fn execute_final_sort_limit_with_retry(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some(sort) = bounded.sort else {
        return Ok(input);
    };
    let MemoryControlPolicy::Bounded { max_retry_attempts, .. } = context.memory_control_policy() else {
        unreachable!("execute_final_sort_limit_with_retry is called only for bounded policies");
    };
    let max_attempts = max_retry_attempts.max(1);
    let timer = Instant::now();
    let result = execute_full_final_sort_limit_attempt(
        context,
        bounded,
        sort,
        bounded.limit,
        &input,
        1,
        max_attempts,
        reservations,
        metrics,
    );

    match result {
        Ok(output) => {
            metrics.final_sort_time += timer.elapsed();
            reservations.release(input.reservation);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, metrics);
            if max_attempts > 1
                && let Some(limit) = bounded.limit
            {
                match execute_hierarchical_topk_final_sort(
                    context,
                    bounded,
                    sort,
                    limit,
                    input,
                    max_attempts,
                    reservations,
                    metrics,
                ) {
                    Ok(output) => {
                        metrics.final_sort_time += timer.elapsed();
                        Ok(output)
                    },
                    Err(error) => {
                        metrics.final_sort_time += timer.elapsed();
                        Err(error)
                    },
                }
            } else {
                metrics.final_sort_time += timer.elapsed();
                reservations.release(input.reservation);
                Err(final_sort_retry_facts("retry_attempts_exhausted", metrics).attach_to_error(error))
            }
        },
        Err(error) => {
            metrics.final_sort_time += timer.elapsed();
            reservations.release(input.reservation);
            Err(error)
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_full_final_sort_limit_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    limit: Option<&LimitNode>,
    input: &CommittedPartial,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let sort_request =
        MemoryReservationRequest::new(ReservationKind::FinalSort, input.reservation.bytes, reservations.budget())
            .with_memory_boundary("final_sort")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_sort")
            .with_attempt(attempt, max_attempts);
    let sort_reservation = reservations.try_reserve(sort_request, metrics)?;

    let result = execute_sort_on_frame(context, sort, input.frame.frame().clone(), metrics)
        .and_then(|frame| execute_optional_limit(context, limit, frame, metrics));
    reservations.release(sort_reservation);
    let frame = result?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

fn execute_hierarchical_topk_final_sort(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    limit: &LimitNode,
    input: CommittedPartial,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let keep_rows = topk_candidate_rows(limit.spec)?;
    if keep_rows == 0 {
        let output = match execute_limit_on_frame(context, limit, input.frame.frame().clone(), metrics) {
            Ok(output) => output,
            Err(error) => {
                reservations.release(input.reservation);
                return Err(error);
            },
        };
        reservations.release(input.reservation);
        return reserve_final_sort_output(bounded, output, 2, max_attempts, reservations, metrics);
    }

    let input_rows = input.frame.num_rows();
    if input_rows == 0 {
        return Ok(input);
    }
    let bytes_per_row = estimated_bytes_per_row(input.reservation.bytes, input_rows);
    let initial_rows_per_chunk = final_sort_rows_per_chunk(
        input_rows,
        bytes_per_row,
        reservations.budget().saturating_sub(input.reservation.bytes),
    );
    let mut candidates = match execute_initial_topk_candidates(
        context,
        bounded,
        sort,
        &input,
        keep_rows,
        bytes_per_row,
        initial_rows_per_chunk,
        max_attempts,
        reservations,
        metrics,
    ) {
        Ok(candidates) => candidates,
        Err(error) => {
            record_final_sort_retry_error(&error, metrics);
            reservations.release(input.reservation);
            return Err(final_sort_retry_facts("hierarchical_topk_initial_failed", metrics).attach_to_error(error));
        },
    };

    reservations.release(input.reservation);

    let mut merge_level = 1;
    while candidates.len() > 1 {
        let mut merge_fan_in = candidates.len();
        let mut attempt = 2;

        loop {
            match execute_topk_merge_level_attempt(
                context,
                bounded,
                sort,
                &candidates,
                keep_rows,
                merge_fan_in,
                attempt,
                max_attempts,
                reservations,
                metrics,
            ) {
                Ok(level_output) => {
                    candidates = commit_topk_merge_level(candidates, level_output, reservations);
                    merge_level += 1;
                    break;
                },
                Err(error) if is_retryable_final_sort_error(&error) => {
                    record_final_sort_retry_error(&error, metrics);
                    if attempt < max_attempts
                        && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                    {
                        merge_fan_in = reduced_fan_in;
                        attempt += 1;
                        continue;
                    }

                    release_committed_partials(candidates, reservations);
                    return Err(final_sort_topk_retry_facts(merge_level, merge_fan_in, metrics).attach_to_error(error));
                },
                Err(error) => {
                    release_committed_partials(candidates, reservations);
                    return Err(error);
                },
            }
        }
    }

    let candidate = candidates.pop().expect("hierarchical top-k leaves exactly one candidate");
    if limit.spec.skip == 0 && candidate.frame.num_rows() <= limit.spec.fetch {
        return Ok(candidate);
    }

    let output = execute_limit(context, limit, candidate.frame, metrics)?;
    reservations.release(candidate.reservation);
    reserve_final_sort_output(bounded, output, max_attempts, max_attempts, reservations, metrics)
}

fn execute_initial_topk_candidates(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    input: &CommittedPartial,
    keep_rows: usize,
    bytes_per_row: u64,
    rows_per_chunk: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<CommittedPartial>> {
    let input_rows = input.frame.num_rows();
    let mut candidates = Vec::with_capacity(input_rows.div_ceil(rows_per_chunk));
    let mut offset = 0;
    while offset < input_rows {
        let length = rows_per_chunk.min(input_rows - offset);
        match execute_topk_slice_attempt(
            context,
            bounded,
            sort,
            input,
            offset,
            length,
            keep_rows,
            bytes_per_row,
            2,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(candidate) => candidates.push(candidate),
            Err(error) => {
                release_committed_partials(candidates, reservations);
                return Err(error);
            },
        }
        offset += length;
    }
    Ok(candidates)
}

#[allow(clippy::too_many_arguments)]
fn execute_topk_slice_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    input: &CommittedPartial,
    offset: usize,
    length: usize,
    keep_rows: usize,
    bytes_per_row: u64,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let requested_bytes = estimate_rows_bytes(bytes_per_row, length);
    let request = MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, reservations.budget())
        .with_memory_boundary("final_sort_topk_chunk")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let work_reservation = reservations.try_reserve(request, metrics)?;

    let result = input
        .frame
        .frame()
        .clone()
        .with_execution(&execution_overrides(context))
        .slice(offset, length)
        .map(GpuDataFrame::new)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_sort.topk_slice",
                "failed to slice bounded native final sort input",
                error,
            )
        })
        .and_then(|frame| execute_topk_candidate(context, sort, frame, keep_rows, metrics));
    reservations.release(work_reservation);
    let frame = result?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

fn execute_topk_merge_level_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    candidates: &[CommittedPartial],
    keep_rows: usize,
    merge_fan_in: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<TopKLevelItem>> {
    let mut output = Vec::with_capacity(candidates.len().div_ceil(merge_fan_in));

    for (group_index, group) in candidates.chunks(merge_fan_in).enumerate() {
        if group.len() == 1 {
            output.push(TopKLevelItem::Carry { partial_index: group_index * merge_fan_in });
            continue;
        }

        match execute_topk_merge_group_attempt(
            context,
            bounded,
            sort,
            group,
            keep_rows,
            attempt,
            max_attempts,
            reservations,
            metrics,
        ) {
            Ok(candidate) => output.push(TopKLevelItem::Merged(candidate)),
            Err(error) => {
                release_topk_level_items(output, reservations);
                return Err(error);
            },
        }
    }

    Ok(output)
}

fn execute_topk_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    sort: &SortNode,
    candidates: &[CommittedPartial],
    keep_rows: usize,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(candidates.len() > 1);
    let requested_bytes = candidates.iter().map(|candidate| candidate.reservation.bytes).sum::<u64>();
    let request = MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, reservations.budget())
        .with_memory_boundary("final_sort_topk_merge")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let merge_reservation = reservations.try_reserve(request, metrics)?;

    let result = concat_committed_frames(context, candidates, "cudf_query_engine.final_sort.topk_concat")
        .and_then(|frame| execute_topk_candidate(context, sort, frame, keep_rows, metrics));
    reservations.release(merge_reservation);
    let frame = result?;
    reserve_final_sort_output(bounded, frame, attempt, max_attempts, reservations, metrics)
}

fn execute_topk_candidate(
    context: &ExecutionContext,
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sorted = execute_sort(context, sort, input, metrics)?;
    execute_limit_spec(context, LimitSpec::fetch(keep_rows), sorted, metrics)
}

fn concat_committed_frames(
    context: &ExecutionContext,
    frames: &[CommittedPartial],
    operation: &'static str,
) -> Result<GpuDataFrame> {
    let first_frame = frames[0].frame.frame().clone();
    let other_frames = frames[1..].iter().map(|partial| partial.frame.frame().clone()).collect::<Vec<_>>();
    let other_frame_refs = other_frames.iter().collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let output = first_frame.with_execution(&overrides).concat_many(&other_frame_refs).map_err(|error| {
        QueryError::from_cudf(operation, "failed to concatenate bounded native final sort candidates", error)
            .with_fact("candidate_count", frames.len())
    })?;
    Ok(GpuDataFrame::new(output))
}

fn reserve_final_sort_output(
    bounded: &BoundedPlan<'_>,
    frame: GpuDataFrame,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

enum TopKLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

fn commit_topk_merge_level(
    candidates: Vec<CommittedPartial>,
    level_output: Vec<TopKLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_candidates = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            TopKLevelItem::Merged(partial) => next_candidates.push(partial),
            TopKLevelItem::Carry { partial_index } => {
                let partial = candidates[partial_index].take().expect("carry index references a top-k candidate");
                next_candidates.push(partial);
            },
        }
    }

    for partial in candidates.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_candidates
}

fn release_topk_level_items(items: Vec<TopKLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

fn release_committed_partials(partials: Vec<CommittedPartial>, reservations: &mut ReservationTracker) {
    for partial in partials {
        reservations.release(partial.reservation);
    }
}

fn is_retryable_final_sort_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

fn record_final_sort_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

fn final_sort_retry_facts(retry_action: &'static str, metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_sort")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

fn final_sort_topk_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    final_sort_retry_facts(
        if merge_fan_in <= 2 { "fail_min_topk_fan_in" } else { "hierarchical_topk_retry_exhausted" },
        metrics,
    )
    .with_merge_level(merge_level)
    .with_merge_fan_in(merge_fan_in)
}

fn topk_candidate_rows(spec: LimitSpec) -> Result<usize> {
    spec.skip.checked_add(spec.fetch).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.final_sort.topk_limit",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", spec.fetch)
    })
}

fn estimated_bytes_per_row(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        return bytes.max(1);
    }
    bytes.div_ceil(rows as u64).max(1)
}

fn estimate_rows_bytes(bytes_per_row: u64, rows: usize) -> u64 {
    let bytes = u128::from(bytes_per_row) * rows as u128;
    bytes.min(u128::from(u64::MAX)) as u64
}

fn final_sort_rows_per_chunk(input_rows: usize, bytes_per_row: u64, available_bytes: u64) -> usize {
    if input_rows == 0 {
        return 1;
    }
    let rows = (available_bytes / bytes_per_row).min(input_rows as u64) as usize;
    rows.max(1)
}

fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
) -> Result<GpuDataFrame> {
    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        bounded.aggregate.1,
        "cudf_query_engine.final_merge.validate",
        "final aggregate output names must match output schema fields",
    )?;
    let expected_output_names =
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .with_execution(&overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_merge.aggregate",
                "failed to execute bounded native final aggregate merge",
                error,
            )
        })?;
    output = cast_count_output_columns(output, &bounded.aggregate.0.spec, bounded.aggregate.1)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.final_merge.select",
            "failed to select bounded native final aggregate output columns",
            error,
        )
    })?;
    Ok(GpuDataFrame::new(output))
}

fn execute_filter(
    context: &ExecutionContext,
    node: &FilterNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let predicate = lower_expr(&node.predicate)?;
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).filter_expr(predicate).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.filter.execute", "failed to execute native cudf filter", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_projection(
    context: &ExecutionContext,
    node: &ProjectionNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let output_field_count = output_schema.fields().len();
    if node.expressions.len() != output_field_count {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.projection.validate",
            "projection expression count must match output schema field count",
        )
        .with_fact("expression_count", node.expressions.len())
        .with_fact("output_field_count", output_field_count));
    }

    let frame = input.into_frame();
    let output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let exprs = output_names
        .iter()
        .zip(&node.expressions)
        .map(|(name, expr)| lower_expr(expr).map(|expr| (name.as_str(), expr)))
        .collect::<Result<Vec<_>>>()?;
    let output_columns = output_names.iter().map(String::as_str).collect::<Vec<_>>();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output =
        frame.with_execution(&overrides).with_columns(exprs).and_then(|frame| frame.select(output_columns)).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.projection.execute",
                    "failed to execute native cudf projection",
                    error,
                )
            },
        )?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .with_execution(&overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.execute",
                "failed to execute native cudf aggregate",
                error,
            )
        })?;
    output = cast_count_output_columns(output, &node.spec, output_schema)?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_keys",
                "failed to rename native cudf aggregate keys",
                error,
            )
        })?;
    }
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn cast_count_output_columns(
    mut frame: DataFrame,
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Count) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.aggregate.cast_count",
                "count aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        let target_type = arrow_type_to_cudf_type_id(field.data_type()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::UnsupportedType,
                "cudf_query_engine.aggregate.cast_count",
                "count aggregate output schema type is not supported by native execution",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
        frame = frame.cast_column(expr.alias.as_str(), target_type).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.cast_count",
                "failed to cast native count aggregate output column",
                error,
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type()))
        })?;
    }
    Ok(frame)
}

fn arrow_type_to_cudf_type_id(data_type: &ArrowDataType) -> Option<TypeId> {
    match data_type {
        ArrowDataType::Boolean => Some(TypeId::Bool8),
        ArrowDataType::Int8 => Some(TypeId::Int8),
        ArrowDataType::Int16 => Some(TypeId::Int16),
        ArrowDataType::Int32 => Some(TypeId::Int32),
        ArrowDataType::Int64 => Some(TypeId::Int64),
        ArrowDataType::UInt8 => Some(TypeId::UInt8),
        ArrowDataType::UInt16 => Some(TypeId::UInt16),
        ArrowDataType::UInt32 => Some(TypeId::UInt32),
        ArrowDataType::UInt64 => Some(TypeId::UInt64),
        ArrowDataType::Float32 => Some(TypeId::Float32),
        ArrowDataType::Float64 => Some(TypeId::Float64),
        _ => None,
    }
}

fn execute_sort(
    context: &ExecutionContext,
    node: &SortNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    execute_sort_on_frame(context, node, input.into_frame(), metrics)
}

fn execute_sort_on_frame(
    context: &ExecutionContext,
    node: &SortNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sort = lower_sort_spec(&node.spec)?;
    let options = CudfSortOptions::new().with_descending(sort.descending).with_nulls_first(sort.nulls_first);
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).sort_values(sort.columns, options).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.sort.execute", "failed to execute native cudf sort", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_limit(
    context: &ExecutionContext,
    node: &LimitNode,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, node.spec)? {
        return Ok(input);
    }

    execute_limit_on_frame(context, node, input.into_frame(), metrics)
}

fn execute_optional_limit(
    context: &ExecutionContext,
    node: Option<&LimitNode>,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    match node {
        Some(node) => execute_limit(context, node, input, metrics),
        None => Ok(input),
    }
}

fn execute_limit_spec(
    context: &ExecutionContext,
    spec: LimitSpec,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = input.num_rows();
    if limit_keeps_all_rows(row_count, spec)? {
        return Ok(input);
    }

    execute_limit_spec_on_frame(context, spec, input.into_frame(), row_count, metrics)
}

fn execute_limit_on_frame(
    context: &ExecutionContext,
    node: &LimitNode,
    frame: DataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let row_count = frame.num_rows();
    execute_limit_spec_on_frame(context, node.spec, frame, row_count, metrics)
}

fn execute_limit_spec_on_frame(
    context: &ExecutionContext,
    spec: LimitSpec,
    frame: DataFrame,
    row_count: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let (offset, length) = limit_slice(row_count, spec)?;
    if offset == 0 && length == row_count {
        return Ok(GpuDataFrame::new(frame));
    }

    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let output = frame.with_execution(&overrides).slice(offset, length).map_err(|error| {
        QueryError::from_cudf("cudf_query_engine.limit.execute", "failed to execute native cudf limit", error)
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn limit_keeps_all_rows(row_count: usize, spec: LimitSpec) -> Result<bool> {
    let (offset, length) = limit_slice(row_count, spec)?;
    Ok(offset == 0 && length == row_count)
}

fn materialize_to_host(input: GpuDataFrame, metrics: &mut QueryMetrics) -> Result<RecordBatch> {
    let timer = Instant::now();
    let output = input.into_frame().to_record_batch().map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.host_arrow.materialize",
            "failed to materialize native cudf output to host Arrow",
            error,
        )
    })?;
    metrics.materialization_time += timer.elapsed();
    metrics
        .boundary
        .record_device_to_host_materialization(output.num_rows() as u64, record_batch_memory_size(&output) as u64);
    Ok(output)
}

fn materialize_bounded_to_host(
    input: CommittedPartial,
    final_node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<RecordBatch> {
    let request =
        MemoryReservationRequest::new(ReservationKind::Materialization, input.reservation.bytes, reservations.budget())
            .with_memory_boundary("host_materialization")
            .with_node_id(final_node_id)
            .with_stage("materialization");
    let materialization_reservation = match reservations.try_reserve(request, metrics) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(input.reservation);
            return Err(error);
        },
    };

    let result = materialize_to_host(input.frame, metrics);
    reservations.release(materialization_reservation);
    reservations.release(input.reservation);
    result
}

fn execution_overrides(context: &ExecutionContext) -> ExecutionOverrides<'_> {
    ExecutionOverrides::new().with_stream(context.stream()).with_memory_resource(context.memory_resource())
}

fn observe_device_memory_bytes(frame: &GpuDataFrame, metrics: &mut QueryMetrics) -> Result<u64> {
    let info = frame.frame().info_with(InfoDetail::DEVICE_ALLOCATED).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.metrics.device_memory",
            "failed to record native device memory diagnostics",
            error,
        )
    })?;
    let bytes = info.memory_accounting.and_then(|accounting| accounting.device_allocated_bytes).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.metrics.device_memory",
            "cudf dataframe memory accounting did not include device-allocated bytes",
        )
        .with_fact("rows", frame.num_rows())
        .with_fact("columns", frame.num_columns())
    })?;
    metrics.record_device_memory_observation(bytes as u64);
    Ok(bytes as u64)
}

fn record_device_memory_observation(frame: &GpuDataFrame, metrics: &mut QueryMetrics) -> Result<()> {
    observe_device_memory_bytes(frame, metrics).map(|_| ())
}

fn format_row_groups(chunk: &SourceChunk) -> String {
    if chunk.reads_all_row_groups {
        return "all".to_string();
    }

    comma_separated_usize(&chunk.row_group_indexes)
}

fn lower_expr(expr: &Expr) -> Result<cudf::Expr> {
    match expr {
        Expr::Column(name) => Ok(col(name)),
        Expr::Literal(value) => lower_literal(value),
        Expr::Binary { left, op, right } => lower_binary(lower_expr(left)?, *op, lower_expr(right)?),
        Expr::Alias { expr, alias } => Ok(lower_expr(expr)?.alias(alias)),
    }
}

fn lower_literal(value: &LiteralValue) -> Result<cudf::Expr> {
    match value {
        LiteralValue::Boolean(value) => Ok(lit(*value)),
        LiteralValue::Int8(value) => Ok(lit(*value)),
        LiteralValue::Int16(value) => Ok(lit(*value)),
        LiteralValue::Int32(value) => Ok(lit(*value)),
        LiteralValue::Int64(value) => Ok(lit(*value)),
        LiteralValue::UInt8(value) => Ok(lit(*value)),
        LiteralValue::UInt16(value) => Ok(lit(*value)),
        LiteralValue::UInt32(value) => Ok(lit(*value)),
        LiteralValue::UInt64(value) => Ok(lit(*value)),
        LiteralValue::Float32(value) => Ok(lit(*value)),
        LiteralValue::Float64(value) => Ok(lit(*value)),
        LiteralValue::String(_) | LiteralValue::Null => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.expr.lower_literal",
            "native execution supports only non-null numeric and boolean literals",
        )
        .with_fact("literal", value)),
    }
}

fn lower_binary(left: cudf::Expr, op: BinaryOp, right: cudf::Expr) -> Result<cudf::Expr> {
    match op {
        BinaryOp::Add => Ok(left + right),
        BinaryOp::Subtract => Ok(left - right),
        BinaryOp::Multiply => Ok(left * right),
        BinaryOp::Divide => Ok(left / right),
        BinaryOp::Eq => Ok(left.eq(right)),
        BinaryOp::NotEq => Ok(left.ne(right)),
        BinaryOp::Lt => Ok(left.lt(right)),
        BinaryOp::LtEq => Ok(left.le(right)),
        BinaryOp::Gt => Ok(left.gt(right)),
        BinaryOp::GtEq => Ok(left.ge(right)),
        BinaryOp::And => Ok(left & right),
        BinaryOp::Or => Ok(left | right),
    }
}

struct LoweredAggregate {
    key_columns: Vec<String>,
    output_key_names: Vec<String>,
    aggregations: Vec<CudfAggSpec>,
    output_names: Vec<String>,
}

fn lower_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    if spec.group_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution currently requires at least one group key",
        ));
    }
    if spec.aggr_exprs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedOperator,
            "cudf_query_engine.aggregate.lower",
            "native execution requires at least one aggregate expression",
        ));
    }

    let mut key_columns = Vec::with_capacity(spec.group_exprs.len());
    let mut output_key_names = Vec::with_capacity(spec.group_exprs.len());
    for expr in &spec.group_exprs {
        let (input_column, output_name) = lower_named_column_expr(expr, "cudf_query_engine.aggregate.lower_key")?;
        key_columns.push(input_column);
        output_key_names.push(output_name);
    }

    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            let input_column = lower_column_expr(&expr.expr, "cudf_query_engine.aggregate.lower_expr")?;
            let function = lower_aggregate_function(expr.function);
            Ok(CudfAggSpec::new(input_column, function).with_alias(expr.alias.as_str()))
        })
        .collect::<Result<Vec<_>>>()?;

    let output_names = output_key_names.iter().chain(spec.aggr_exprs.iter().map(|expr| &expr.alias)).cloned().collect();

    Ok(LoweredAggregate { key_columns, output_key_names, aggregations, output_names })
}

fn lower_final_aggregate_spec(spec: &AggregateSpec) -> Result<LoweredAggregate> {
    let partial = lower_aggregate_spec(spec)?;
    let aggregations = spec
        .aggr_exprs
        .iter()
        .map(|expr| {
            CudfAggSpec::new(expr.alias.as_str(), lower_final_aggregate_function(expr.function))
                .with_alias(expr.alias.as_str())
        })
        .collect::<Vec<_>>();

    Ok(LoweredAggregate {
        key_columns: partial.output_key_names.clone(),
        output_key_names: partial.output_key_names,
        aggregations,
        output_names: partial.output_names,
    })
}

fn lower_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Count { include_nulls: false },
    }
}

fn lower_final_aggregate_function(function: AggregateFunction) -> GroupByAgg {
    match function {
        AggregateFunction::Sum => GroupByAgg::Sum,
        AggregateFunction::Min => GroupByAgg::Min,
        AggregateFunction::Max => GroupByAgg::Max,
        AggregateFunction::Count => GroupByAgg::Sum,
    }
}

struct LoweredSort {
    columns: Vec<String>,
    descending: Vec<bool>,
    nulls_first: Vec<bool>,
}

fn lower_sort_spec(spec: &SortSpec) -> Result<LoweredSort> {
    if spec.expressions.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.sort.lower",
            "native execution requires at least one sort expression",
        ));
    }

    let mut columns = Vec::with_capacity(spec.expressions.len());
    let mut descending = Vec::with_capacity(spec.expressions.len());
    let mut nulls_first = Vec::with_capacity(spec.expressions.len());
    for sort_expr in &spec.expressions {
        columns.push(lower_column_expr(&sort_expr.expr, "cudf_query_engine.sort.lower")?);
        descending.push(sort_expr.descending);
        nulls_first.push(sort_expr.nulls_first);
    }
    Ok(LoweredSort { columns, descending, nulls_first })
}

fn lower_named_column_expr(expr: &Expr, operation: &'static str) -> Result<(String, String)> {
    match expr {
        Expr::Column(name) => Ok((name.clone(), name.clone())),
        Expr::Alias { expr, alias } => Ok((lower_column_expr(expr, operation)?, alias.clone())),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

fn lower_column_expr(expr: &Expr, operation: &'static str) -> Result<String> {
    match expr {
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias { expr, .. } => lower_column_expr(expr, operation),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            operation,
            "native execution requires a column expression",
        )
        .with_fact("expression", expr)),
    }
}

fn limit_slice(row_count: usize, spec: LimitSpec) -> Result<(usize, usize)> {
    let Some(end) = spec.skip.checked_add(spec.fetch) else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.limit.validate",
            "limit skip and fetch overflow usize",
        )
        .with_fact("skip", spec.skip)
        .with_fact("fetch", spec.fetch));
    };
    let offset = spec.skip.min(row_count);
    let length = end.min(row_count).saturating_sub(offset);
    Ok((offset, length))
}

fn record_batch_memory_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|array| array.get_array_memory_size()).sum()
}

fn invalid_node_reference(id: NodeId, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "cudf_query_engine.plan.execute", message).with_fact("node", id)
}

fn unsupported_bounded_shape(message: &'static str, reason: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::UnsupportedPlanShape, "cudf_query_engine.bounded.plan", message)
        .with_fact("reason", reason)
}

fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}
