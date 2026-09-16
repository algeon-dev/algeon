use std::{sync::Arc, time::Instant};

use super::{
    BoundedExecutionAttempt, BoundedPlan, ExecutionBudgetToken, ExecutionContext, ExecutionMetrics, FilterNode,
    GpuDataFrame, GpuResidentInputMap, GpuResidentSource, GpuResultHandle, LocalParquetSource, NodeId, PlanNode,
    ProjectionNode, QueryError, QueryErrorCode, QueryOutput, QueryPlan, ResidentGpuFrame, Result, ResultSink,
    SourceChunk, SourceChunkPlanningPolicy, concatenate_source_read_outputs, cudf_query_error,
    execute_bounded_aggregate_to_frame, execute_compiled_bounded_aggregate_from_frame, execute_edge_normalize,
    execute_filter_with_budget, execute_limit, execute_projection_with_budget,
    execute_projection_with_budget_and_prepared, execute_sort, execute_source_adjacent_chunk_projection,
    execute_source_adjacent_chunk_unary, execute_window, execution_budget_token_for_boundary, execution_overrides,
    materialize_bounded_to_host, observe_device_memory_bytes, plan_local_parquet_source_chunks_with_execution_context,
    projection_regexp_replace_prepare, read_gpu_resident_source, read_literal_batch_source,
    read_local_parquet_chunks_required_only, reject_bounded_join_plan, release_source_read_outputs_reverse,
    unsupported_bounded_shape, validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
use super::{DataFrame, execute_graph_algorithm_sanitized, graph_preparation_bypass_input};
#[cfg(feature = "cugraph")]
use crate::exec::graph_algorithm::{bounded_betweenness::*, graph_empty_input_prevalidated, statistics::*};
#[cfg(not(feature = "cugraph"))]
use crate::exec::graph_algorithm_feature_disabled;
use crate::{
    exec::source::{ScheduledLocalParquetReadOutput, record_bounded_parquet_source_metrics},
    observability::SourceReadFact,
};

pub(in crate::exec::bounded) fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.generic_query_plan")?;
    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics)?;
    let committed = execute_bounded_generic_node(&mut execution, plan, plan.sink().input(), gpu_inputs)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let output_bytes = committed.frame.device_allocated_size_bytes()?;
            let resident_owner = execution.context().retain_resident_output_owner(output_bytes)?;
            let frame = committed
                .frame
                .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(plan));
            let handle = match GpuResultHandle::new_with_resident_owner(
                execution.context(),
                frame,
                Arc::clone(plan.output_schema()),
                resident_owner,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    return Err(error);
                },
            };
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                &mut execution,
                committed,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
            )?;
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_generic_node(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    execution.context().check_cancellation("nexus_query_engine.exec.bounded.generic_node")?;
    let node = plan.node(id).expect("validated bounded generic node exists");
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                execute_bounded_generic_local_parquet_source(execution, source, id)
            },
            crate::source::SourceNode::GpuResident(source) => {
                execute_bounded_generic_gpu_resident_source(execution, source, gpu_inputs)
            },
            crate::source::SourceNode::LiteralBatch(source) => {
                execute_bounded_generic_literal_batch_source(execution, source)
            },
        },
        PlanNode::Filter(filter) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(execution, plan, id)?
            {
                return Ok(frame);
            }
            let input = execute_bounded_generic_node(execution, plan, filter.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_filter", input, true, |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for bounded generic filter stages"),
                        metrics,
                    )
                })
        },
        PlanNode::Projection(projection) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(execution, plan, id)?
            {
                return Ok(frame);
            }
            let input =
                execute_bounded_generic_node(execution, plan, projection.input, gpu_inputs)?;
            execute_bounded_generic_projection(execution, id, "bounded_generic_projection", projection, input)
        },
        PlanNode::Window(window) => {
            let input = execute_bounded_generic_node(execution, plan, window.input, gpu_inputs)?;
            let input_schema = plan
                .node(window.input)
                .expect("validated window input exists")
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            execute_bounded_generic_unary(execution, id, "bounded_generic_window", input, false, |context, input, _token, metrics| {
                    execute_window(context, window, &input_schema, input, &output_schema, metrics)
                })
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(execution, plan, sort.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_sort", input, false, |context, input, _token, metrics| execute_sort(context, sort, input, metrics))
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(execution, plan, limit.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_limit", input, false, |context, input, _token, metrics| execute_limit(context, limit, input, metrics))
        },
        PlanNode::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| execute_bounded_generic_node(execution, plan, *input, gpu_inputs))
                .collect::<Result<Vec<_>>>()?;
            execute_bounded_generic_union(execution, id, inputs)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let input =
                execute_bounded_generic_node(execution, plan, edge_normalize.input, gpu_inputs)?;
            execute_bounded_edge_normalize_from_input(execution, id, edge_normalize, input)
        },
        PlanNode::GraphAlgorithm(graph) => {
            execute_bounded_generic_graph_algorithm(execution, plan, id, graph, gpu_inputs)
        },
        PlanNode::VectorAlgorithm(vector) => {
            #[cfg(feature = "cuvs")]
            {
                crate::exec::validate_vector_algorithm_execution(id, vector)?;
                let mut inputs = std::collections::BTreeMap::new();
                for input_id in &vector.inputs {
                    if !inputs.contains_key(input_id) {
                        let input = execute_bounded_generic_node(execution, plan, *input_id, gpu_inputs)?;
                        inputs.insert(*input_id, input.frame);
                    }
                }
                let frame = crate::exec::execute_vector_algorithm(execution.context(), plan, id, vector, &inputs)?;
                let device_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
                Ok(ResidentGpuFrame { frame, device_bytes })
            }
            #[cfg(not(feature = "cuvs"))]
            {
                match crate::exec::execute_vector_algorithm(id, vector) {
                    Ok(_) => unreachable!("cuVS-disabled execution cannot produce a GPU frame"),
                    Err(error) => Err(error),
                }
            }
        },
        PlanNode::Aggregate(aggregate) => {
            let input = match plan.node(aggregate.input).map(|node| &node.kind) {
                Some(PlanNode::Aggregate(_)) => match BoundedPlan::analyze_aggregate_subplan(plan, aggregate.input) {
                    Ok(bounded) => execute_bounded_aggregate_to_frame(execution, &bounded)?,
                    Err(_error) => {
                        execute_bounded_generic_node(execution, plan, aggregate.input, gpu_inputs)?
                    },
                },
                _ => execute_bounded_generic_node(execution, plan, aggregate.input, gpu_inputs)?,
            };
            let input_schema = plan
                .node(aggregate.input)
                .expect("validated aggregate input exists")
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            let input_reserved_bytes = input.device_bytes();
            execute_compiled_bounded_aggregate_from_frame(
                execution,
                id,
                aggregate,
                &input_schema,
                &output_schema,
                input.frame,
                input_reserved_bytes,
            )
        },
        PlanNode::Join(_) => {
            Err(unsupported_bounded_shape(
            "bounded generic execution supports only source, filter, projection, aggregate, window, sort, limit, union, edge-normalize, and graph-algorithm nodes",
            "bounded_generic_plan_shape_unsupported",
        )
        .with_fact("node_id", id)
        .with_fact("node_kind", bounded_generic_node_kind(&node.kind)))
        },
    }?;

    Ok(frame)
}

pub(in crate::exec::bounded) fn execute_bounded_edge_normalize_from_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    edge_normalize: &crate::plan::EdgeNormalizeNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary(
        execution,
        node_id,
        "bounded_generic_edge_normalize",
        input,
        false,
        |context, input, _token, metrics| execute_edge_normalize(context, node_id, edge_normalize, input, metrics),
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_union(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    inputs: Vec<ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    if inputs.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.union",
            "bounded union requires at least one input",
        )
        .with_fact("node_id", node_id));
    }
    if inputs.len() == 1 {
        return Ok(inputs.into_iter().next().expect("single union input checked"));
    }

    let input_count = inputs.len();
    let output = (|| {
        let mut frames = Vec::with_capacity(input_count);
        for input in inputs {
            frames.push(input.frame.into_frame());
        }
        let first = frames.remove(0);
        let rest = frames.iter().collect::<Vec<_>>();
        let overrides_lease = execution_overrides(execution.context())?;
        let overrides = overrides_lease.overrides();
        let timer = Instant::now();
        let output = first.with_execution(&overrides).concat_many(&rest).map_err(|error| {
            cudf_query_error(
                "nexus_query_engine.bounded.union.concat",
                "failed to concatenate bounded native union inputs",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("input_count", input_count)
        })?;
        execution.metrics().add_operator_time(timer.elapsed());

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics())?;
        Ok::<_, QueryError>((output_frame, output_bytes))
    })();
    match output {
        Ok((frame, device_bytes)) => Ok(ResidentGpuFrame { frame, device_bytes }),
        Err(error) => Err(error),
    }
}

#[cfg(not(feature = "cugraph"))]
fn execute_bounded_generic_graph_algorithm(
    _execution: &mut BoundedExecutionAttempt<'_>,
    _plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    _gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    Err(graph_algorithm_feature_disabled(node_id, graph))
}

#[cfg(feature = "cugraph")]
fn execute_bounded_generic_graph_algorithm(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    let input = if let Some(raw_input_id) = graph_preparation_bypass_input(plan, graph) {
        let raw_input = execute_bounded_generic_node(execution, plan, raw_input_id, gpu_inputs)?;
        let PlanNode::EdgeNormalize(edge_normalize) =
            &plan.node(graph.input).expect("preparation bypass validated graph input").kind
        else {
            unreachable!("preparation bypass requires an edge-normalize input")
        };
        execute_bounded_edge_normalize_from_input(execution, graph.input, edge_normalize, raw_input)?
    } else {
        execute_bounded_generic_node(execution, plan, graph.input, gpu_inputs)?
    };
    let mut side_inputs = std::collections::BTreeMap::new();
    for input_id in graph.inputs.iter().skip(1) {
        match execute_bounded_generic_node(execution, plan, *input_id, gpu_inputs) {
            Ok(frame) => {
                side_inputs.insert(*input_id, frame);
            },
            Err(error) => {
                release_resident_frames_reverse(std::iter::once(input).chain(side_inputs.into_values()).collect());
                return Err(error);
            },
        }
    }
    execute_bounded_graph_algorithm_from_inputs(execution, node_id, graph, input, side_inputs)
}

#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) fn execute_bounded_graph_algorithm_from_inputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    input: ResidentGpuFrame,
    side_input_frames: std::collections::BTreeMap<NodeId, ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    let ResidentGpuFrame { frame: mut input_frame, .. } = input;
    let _retained_normalization_source = input_frame.take_retained_normalization_source();
    let input_rows = input_frame.num_rows() as u64;
    let input_columns = input_frame.num_columns() as u64;
    let input_graph_residency = input_frame.graph_residency().cloned();
    let prior_null_edge_rows_dropped = execution.metrics().null_edge_rows_dropped(graph.input);
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope {
        weight: graph.spec.algorithm.consumes_weight(),
        edge_id: graph.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input_frame = match super::edge_null_filter::filter_null_edge_rows(
        execution.context(),
        node_id,
        "graph_edge_null_filter",
        input_frame.into_frame(),
        &graph.spec.edge_columns,
        null_filter_scope,
        execution.metrics(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let sanitized_input_rows = filtered_input_frame.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    let mut filtered_input = GpuDataFrame::new(filtered_input_frame);
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        filtered_input = filtered_input.with_graph_residency(graph_residency);
    }
    let mut side_inputs = std::collections::BTreeMap::new();
    for (input_id, frame) in side_input_frames {
        side_inputs.insert(input_id, frame.frame);
    }
    if let Err(error) = validate_betweenness_static_seed_policy(&graph.spec) {
        return Err(error.with_fact_if_absent("node_id", node_id));
    }
    let empty_input_prevalidated = match graph_empty_input_prevalidated(
        execution.context(),
        graph,
        filtered_input.frame(),
        &side_inputs,
        input_rows,
        sanitized_input_rows,
    ) {
        Ok(empty_input_prevalidated) => empty_input_prevalidated,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let mut runtime_spec = RuntimeGraphSpecResolution::CheckedNoRewrite;
    if !empty_input_prevalidated {
        execution.metrics().record_graph_claim_calibration(node_id, graph.spec.algorithm);
        let graph_statistics_probe =
            match graph_runtime_vertex_count_probe(execution.context(), &graph.spec, filtered_input.frame()) {
                Ok(probe) => probe,
                Err(error) => {
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            };
        if let Some(vertex_count) = graph_statistics_probe
            && let Err(error) = validate_betweenness_seed_policy_for_vertex_count(&graph.spec, vertex_count)
        {
            return Err(error.with_fact_if_absent("node_id", node_id));
        }
        runtime_spec =
            runtime_graph_spec_resolution_after_bounded_seed_validation(&graph.spec, graph_statistics_probe.is_some());
    }
    let output = match execute_graph_algorithm_sanitized(
        execution.context(),
        node_id,
        graph,
        runtime_spec,
        filtered_input,
        side_inputs,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        empty_input_prevalidated,
        execution.metrics(),
    ) {
        Ok(output) => output,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&output, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame { frame: output, device_bytes: output_bytes })
}

#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) fn release_resident_frames_reverse(mut frames: Vec<ResidentGpuFrame>) {
    while frames.pop().is_some() {}
}

#[cfg(feature = "cugraph")]
/// Runs the runtime graph vertex-count probe on the attempt's own runtime.
///
/// This runtime sizing only informs betweenness seed selection. It carries no
/// reservation or rejection authority, so an allocation failure degrades the
/// estimate instead of failing the query.
#[cfg(feature = "cugraph")]
fn graph_runtime_vertex_count_probe(
    context: &ExecutionContext,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
) -> Result<Option<u64>> {
    match runtime_graph_statistics_probe(context, input, spec) {
        Ok(vertex_count) => Ok(Some(vertex_count)),
        Err(error)
            if matches!(
                error.code(),
                QueryErrorCode::QueryMemoryLimitExceeded
                    | QueryErrorCode::DeviceOutOfMemory
                    | QueryErrorCode::DeviceAllocationFailed
                    | QueryErrorCode::CudfResourceExhausted
            ) =>
        {
            Ok(None)
        },
        Err(error) => Err(error),
    }
}

#[derive(Clone, Copy)]
enum BoundedGenericSourceAdjacentStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode },
}

fn try_execute_bounded_generic_source_adjacent_stream(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: &QueryPlan,
    root_id: NodeId,
) -> Result<Option<ResidentGpuFrame>> {
    let Some((source_id, source, stages)) = collect_bounded_generic_source_adjacent_stages(plan, root_id)? else {
        return Ok(None);
    };
    if source.has_position_delete_visible_scan_limit() {
        return Ok(None);
    }
    execute_bounded_generic_local_parquet_source_with_stages(execution, source, source_id, &stages).map(Some)
}

fn collect_bounded_generic_source_adjacent_stages(
    plan: &QueryPlan,
    root_id: NodeId,
) -> Result<Option<(NodeId, &LocalParquetSource, Vec<BoundedGenericSourceAdjacentStage<'_>>)>> {
    let mut stages = Vec::new();
    let mut current_id = root_id;
    loop {
        let node = plan.node(current_id).expect("validated source-adjacent node exists");
        match &node.kind {
            PlanNode::Filter(filter) => {
                stages.push(BoundedGenericSourceAdjacentStage::Filter { node_id: current_id, node: filter });
                current_id = filter.input;
            },
            PlanNode::Projection(projection) => {
                stages.push(BoundedGenericSourceAdjacentStage::Projection { node_id: current_id, node: projection });
                current_id = projection.input;
            },
            PlanNode::Source(source) => match source.as_ref() {
                crate::source::SourceNode::LocalParquet(source) if !stages.is_empty() => {
                    stages.reverse();
                    return Ok(Some((current_id, source, stages)));
                },
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }
    }
}

pub(in crate::exec::bounded) fn execute_bounded_generic_local_parquet_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source, crate::rapids_backend::rapids_analysis_context())?;
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        node_id,
        source,
        SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
            source,
            execution.context().memory_control_policy(),
            execution.context().source_chunk_planning_mode(),
            execution.context().source_chunk_planning_options(),
        ),
        execution.context(),
    )?;
    for chunk in &chunk_plan.chunks {
        execution.record_source_read(SourceReadFact::ChunkPlanned { row_groups: chunk.row_group_count() as u64 });
    }
    if chunk_plan.chunks.is_empty() {
        let frame =
            crate::exec::source::read_local_parquet_source(execution.context(), node_id, source, execution.metrics())?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        return Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes });
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, execution.metrics());

    let chunks =
        read_local_parquet_chunks_required_only(execution, source, &chunk_plan.chunks, "bounded_generic_source_read")?;

    concatenate_source_read_outputs(
        execution,
        node_id,
        chunks,
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn execute_bounded_generic_local_parquet_source_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source, crate::rapids_backend::rapids_analysis_context())?;
    let planning_policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
        source,
        execution.context().memory_control_policy(),
        execution.context().source_chunk_planning_mode(),
        execution.context().source_chunk_planning_options(),
    );
    let chunk_plan =
        plan_local_parquet_source_chunks_with_execution_context(node_id, source, planning_policy, execution.context())?;
    for chunk in &chunk_plan.chunks {
        execution.record_source_read(SourceReadFact::ChunkPlanned { row_groups: chunk.row_group_count() as u64 });
    }
    if chunk_plan.chunks.is_empty() {
        let frame =
            crate::exec::source::read_local_parquet_source(execution.context(), node_id, source, execution.metrics())?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        let mut resident = ResidentGpuFrame { frame, device_bytes: requested_bytes };
        for stage in stages {
            resident = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id, node } => execute_bounded_generic_unary(
                    execution,
                    *node_id,
                    "bounded_generic_source_filter",
                    resident,
                    true,
                    |context, input, token, metrics| {
                        execute_filter_with_budget(
                            context,
                            node,
                            input,
                            token.expect("budget token is created for source filter stage"),
                            metrics,
                        )
                    },
                )?,
                BoundedGenericSourceAdjacentStage::Projection { node_id, node } => execute_bounded_generic_projection(
                    execution,
                    *node_id,
                    "bounded_generic_source_projection",
                    node,
                    resident,
                )?,
            };
        }
        return Ok(resident);
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, execution.metrics());

    let chunks = read_bounded_generic_source_adjacent_chunks(execution, source, &chunk_plan.chunks, stages)?;

    concatenate_source_read_outputs(
        execution,
        node_id,
        chunks,
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn read_bounded_generic_source_adjacent_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut transformed = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let source_chunk = match read_local_parquet_chunks_required_only(
            execution,
            source,
            std::slice::from_ref(chunk),
            "bounded_generic_source_read",
        ) {
            Ok(source_chunk) => source_chunk,
            Err(error) => {
                release_source_read_outputs_reverse(transformed);
                return Err(error);
            },
        };
        let mut staged = match apply_bounded_generic_source_adjacent_stages(execution, source_chunk, stages) {
            Ok(staged) => staged,
            Err(error) => {
                release_source_read_outputs_reverse(transformed);
                return Err(error);
            },
        };
        transformed.append(&mut staged);
    }
    Ok(transformed)
}

fn apply_bounded_generic_source_adjacent_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    chunks: Vec<ScheduledLocalParquetReadOutput>,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut pending = chunks;
    let mut transformed = Vec::with_capacity(pending.len());
    while !pending.is_empty() {
        let source = pending.remove(0);
        let chunk = source.chunk.clone();
        let mut current = ResidentGpuFrame::with_device_bytes(source.frame, source.observed_bytes);
        for stage in stages {
            let result = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id: _, node } => execute_source_adjacent_chunk_unary(
                    execution,
                    "bounded_generic_source_filter",
                    "bounded_generic_source_filter_output",
                    &chunk,
                    current,
                    1,
                    1,
                    |context, input, token, metrics| execute_filter_with_budget(context, node, input, token, metrics),
                ),
                BoundedGenericSourceAdjacentStage::Projection { node_id: _, node } => {
                    execute_source_adjacent_chunk_projection(
                        execution,
                        "bounded_generic_source_projection",
                        "bounded_generic_source_projection_output",
                        &chunk,
                        node,
                        current,
                        1,
                        1,
                    )
                },
            };
            current = match result {
                Ok(current) => current,
                Err(error) => {
                    release_source_read_outputs_reverse(pending);
                    release_source_read_outputs_reverse(transformed);
                    return Err(error);
                },
            };
        }
        let observed_bytes = current.device_bytes();
        transformed.push(ScheduledLocalParquetReadOutput {
            chunk,
            frame: current.frame,
            observed_bytes,
            observed_row_group_count: None,
        });
    }
    Ok(transformed)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_gpu_resident_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &GpuResidentSource,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(execution.context(), source, gpu_inputs, execution.metrics())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_literal_batch_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &crate::source::LiteralBatchSource,
) -> Result<ResidentGpuFrame> {
    let frame = read_literal_batch_source(execution.context(), source, execution.metrics())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
    Ok(ResidentGpuFrame { frame, device_bytes: requested_bytes })
}
pub(in crate::exec::bounded) fn execute_bounded_generic_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: ResidentGpuFrame,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &ExecutionMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_consuming_input(execution, node_id, stage, input, use_budget_token, execute)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary(
            execution,
            node_id,
            stage,
            input,
            true,
            |context, input, token, metrics| {
                execute_projection_with_budget(
                    context,
                    projection,
                    input,
                    token.expect("budget token is created for bounded generic projection stages"),
                    metrics,
                )
            },
        );
    }
    execute_bounded_generic_regexp_projection(execution, node_id, stage, projection, input.frame)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary_from_frame(
            execution,
            node_id,
            stage,
            input,
            true,
            |context, input, token, metrics| {
                execute_projection_with_budget(
                    context,
                    projection,
                    input,
                    token.expect("budget token is created for bounded generic projection stages"),
                    metrics,
                )
            },
        );
    }
    execute_bounded_generic_regexp_projection(execution, node_id, stage, projection, input)
}

fn execute_bounded_generic_regexp_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    projection: &ProjectionNode,
    input_frame: GpuDataFrame,
) -> Result<ResidentGpuFrame> {
    // Prepared state stays owned by the native operation and runs under the
    // attempt's allocator-enforced budget token.
    let prepared =
        match projection_regexp_replace_prepare(execution.context(), projection, &input_frame, execution.metrics()) {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                // Defense in depth: no regexp after the gate above.
                let token = match execution_budget_token_for_boundary(
                    execution.context(),
                    execution.device_budget_bytes(),
                    stage,
                    "nexus_query_engine.bounded_generic.budget_token",
                    "failed to create bounded native generic operator budget token",
                ) {
                    Ok(token) => token,
                    Err(error) => {
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                };
                match execute_projection_with_budget(
                    execution.context(),
                    projection,
                    input_frame,
                    &token,
                    execution.metrics(),
                ) {
                    Ok(frame) => {
                        let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                return Err(error.with_fact_if_absent("node_id", node_id));
                            },
                        };
                        return Ok(ResidentGpuFrame { frame, device_bytes: output_bytes });
                    },
                    Err(error) => {
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                }
            },
            Err(error) => {
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };

    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        stage,
        "nexus_query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let result = execute_projection_with_budget_and_prepared(
        execution.context(),
        projection,
        input_frame,
        &token,
        Some(prepared),
        execution.metrics(),
    );
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    Ok(ResidentGpuFrame { frame, device_bytes: output_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: GpuDataFrame,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &ExecutionMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => return Err(error.with_fact_if_absent("node_id", node_id)),
        }
    } else {
        None
    };
    let frame = execute(execution.context(), input, token.as_ref(), execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    Ok(ResidentGpuFrame { frame, device_bytes: output_bytes })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_consuming_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    input: ResidentGpuFrame,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &ExecutionMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_frame = input.frame;
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            execution.context(),
            execution.device_budget_bytes(),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let frame = execute(execution.context(), input_frame, token.as_ref(), execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    let output_bytes = observe_device_memory_bytes(&frame, execution.metrics())
        .map_err(|error| error.with_fact_if_absent("node_id", node_id))?;
    Ok(ResidentGpuFrame { frame, device_bytes: output_bytes })
}

fn bounded_generic_node_kind(node: &PlanNode) -> &'static str {
    match node {
        PlanNode::Source(_) => "source",
        PlanNode::Filter(_) => "filter",
        PlanNode::Projection(_) => "projection",
        PlanNode::Aggregate(_) => "aggregate",
        PlanNode::Window(_) => "window",
        PlanNode::Sort(_) => "sort",
        PlanNode::Limit(_) => "limit",
        PlanNode::Join(_) => "join",
        PlanNode::Union(_) => "union",
        PlanNode::EdgeNormalize(_) => "edge_normalize",
        PlanNode::GraphAlgorithm(_) => "graph_algorithm",
        PlanNode::VectorAlgorithm(_) => "vector_algorithm",
    }
}
