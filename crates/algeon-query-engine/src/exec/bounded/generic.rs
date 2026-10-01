#[cfg(feature = "cugraph")]
use std::collections::BTreeMap;
use std::{slice::from_ref, sync::Arc};

use super::{
    BoundedExecutionAttempt, BoundedPlan, ExecutionContext, ExecutionMetrics, FilterNode, GpuResidentInputMap,
    GpuResultHandle, LocalParquetSource, NodeId, PlanNode, ProjectionNode, QueryOutput, QueryPlan, ResidentGpuFrame,
    Result, ResultSink, SourceChunk, SourceChunkPlanningPolicy, concatenate_source_read_outputs,
    execute_bounded_aggregate_to_frame, execute_bounded_edge_normalize_from_input, execute_bounded_gpu_resident_source,
    execute_bounded_literal_batch_source, execute_bounded_local_parquet_source, execute_bounded_projection,
    execute_bounded_unary, execute_bounded_union, execute_compiled_bounded_aggregate_from_frame,
    execute_filter_with_budget, execute_limit, execute_sort, execute_source_adjacent_chunk_projection,
    execute_source_adjacent_chunk_unary, execute_window, materialize_bounded_to_host, observe_device_memory_bytes,
    plan_local_parquet_source_chunks_with_execution_context, read_local_parquet_chunks_required_only,
    reject_bounded_join_plan, unsupported_bounded_shape, validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
use super::{execute_bounded_graph_algorithm_from_inputs, graph_preparation_bypass_input};
#[cfg(not(feature = "cugraph"))]
use crate::exec::graph_algorithm_feature_disabled;
use crate::{
    exec::source::{ScheduledLocalParquetReadOutput, read_local_parquet_source, record_bounded_parquet_source_metrics},
    observability::SourceReadFact,
    plan::GraphAlgorithmNode,
    rapids_backend::rapids_analysis_context,
    source::SourceNode,
};

pub(in crate::exec::bounded) fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &ExecutionMetrics,
) -> Result<QueryOutput> {
    context.check_cancellation("query_engine.exec.bounded.generic_query_plan")?;
    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics)?;
    let committed = execute_bounded_generic_node(&mut execution, plan, plan.sink().input(), gpu_inputs)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } | ResultSink::DeferredHostArrow { .. } => {
            let output_bytes = committed.frame.device_allocated_size_bytes()?;
            let resident_owner = execution.context().retain_resident_output_owner(output_bytes)?;
            let frame = committed.frame.with_producer_execution_scope(plan.execution_fingerprint());
            let handle = match GpuResultHandle::new_with_resident_owner(
                execution.context(),
                frame,
                Arc::clone(plan.output_schema()),
                resident_owner,
            ) {
                Ok(handle) => handle.with_producer_execution_plan(plan.clone()),
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
    execution.context().check_cancellation("query_engine.exec.bounded.generic_node")?;
    let node = plan.node(id).expect("validated bounded generic node exists");
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                execute_bounded_local_parquet_source(execution, source, id)
            },
            SourceNode::GpuResident(source) => {
                execute_bounded_gpu_resident_source(execution, source, gpu_inputs)
            },
            SourceNode::LiteralBatch(source) => {
                execute_bounded_literal_batch_source(execution, source)
            },
        },
        PlanNode::Filter(filter) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(execution, plan, id)?
            {
                return Ok(frame);
            }
            let input = execute_bounded_generic_node(execution, plan, filter.input, gpu_inputs)?;
            execute_bounded_unary(execution, id, "bounded_generic_filter", input, true, |context, input, token, metrics| {
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
            execute_bounded_projection(execution, id, "bounded_generic_projection", projection, input)
        },
        PlanNode::Window(window) => {
            let input = execute_bounded_generic_node(execution, plan, window.input, gpu_inputs)?;
            let input_schema = plan
                .node(window.input)
                .expect("validated window input exists")
                .output_schema();
            let output_schema = node.output_schema();
            execute_bounded_unary(execution, id, "bounded_generic_window", input, false, |context, input, _token, metrics| {
                    execute_window(context, window, input_schema, input, output_schema, metrics)
                })
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(execution, plan, sort.input, gpu_inputs)?;
            execute_bounded_unary(execution, id, "bounded_generic_sort", input, false, |context, input, _token, metrics| execute_sort(context, sort, input, metrics))
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(execution, plan, limit.input, gpu_inputs)?;
            execute_bounded_unary(execution, id, "bounded_generic_limit", input, false, |context, input, _token, metrics| execute_limit(context, limit, input, metrics))
        },
        PlanNode::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| execute_bounded_generic_node(execution, plan, *input, gpu_inputs))
                .collect::<Result<Vec<_>>>()?;
            execute_bounded_union(execution, id, inputs)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let input =
                execute_bounded_generic_node(execution, plan, edge_normalize.input, gpu_inputs)?;
            execute_bounded_edge_normalize_from_input(execution, id, edge_normalize, input)
        },
        PlanNode::GraphAlgorithm(graph) => {
            execute_bounded_generic_graph_algorithm(execution, plan, id, graph, gpu_inputs)
        },
        PlanNode::MlAlgorithm(_) => Err(unsupported_bounded_shape(
            "bounded ML execution requires an admitted compiled pipeline",
            "compiled_ml_pipeline_required",
        )
        .with_fact("node_id", id)),
        PlanNode::VectorAlgorithm(_) => Err(unsupported_bounded_shape(
            "bounded vector execution requires an admitted compiled pipeline",
            "compiled_vector_pipeline_required",
        )
        .with_fact("node_id", id)),
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
                .output_schema();
            let output_schema = node.output_schema();
            let input_device_bytes = input.device_bytes();
            execute_compiled_bounded_aggregate_from_frame(
                execution,
                id,
                aggregate,
                input_schema,
                output_schema,
                input.frame,
                input_device_bytes,
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

#[cfg(not(feature = "cugraph"))]
fn execute_bounded_generic_graph_algorithm(
    _execution: &mut BoundedExecutionAttempt<'_>,
    _plan: &QueryPlan,
    node_id: NodeId,
    graph: &GraphAlgorithmNode,
    _gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    Err(graph_algorithm_feature_disabled(node_id, graph))
}

#[cfg(feature = "cugraph")]
fn execute_bounded_generic_graph_algorithm(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &GraphAlgorithmNode,
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
    let mut side_inputs = BTreeMap::new();
    for input_id in graph.inputs.iter().skip(1) {
        match execute_bounded_generic_node(execution, plan, *input_id, gpu_inputs) {
            Ok(frame) => {
                side_inputs.insert(*input_id, frame);
            },
            Err(error) => {
                return Err(error);
            },
        }
    }
    execute_bounded_graph_algorithm_from_inputs(execution, node_id, graph, input, side_inputs)
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
                SourceNode::LocalParquet(source) if !stages.is_empty() => {
                    stages.reverse();
                    return Ok(Some((current_id, source, stages)));
                },
                _ => return Ok(None),
            },
            _ => return Ok(None),
        }
    }
}

fn execute_bounded_generic_local_parquet_source_with_stages(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    node_id: NodeId,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source, rapids_analysis_context())?;
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
        let frame = read_local_parquet_source(execution.context(), node_id, source, execution.metrics())?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics())?;
        let mut resident = ResidentGpuFrame { frame, device_bytes: requested_bytes };
        for stage in stages {
            resident = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id, node } => execute_bounded_unary(
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
                BoundedGenericSourceAdjacentStage::Projection { node_id, node } => execute_bounded_projection(
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
        "query_engine.bounded.generic.source_concat",
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
            from_ref(chunk),
            "bounded_generic_source_read",
        ) {
            Ok(source_chunk) => source_chunk,
            Err(error) => {
                return Err(error);
            },
        };
        let mut staged = match apply_bounded_generic_source_adjacent_stages(execution, source_chunk, stages) {
            Ok(staged) => staged,
            Err(error) => {
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
        PlanNode::MlAlgorithm(_) => "ml_algorithm",
    }
}
