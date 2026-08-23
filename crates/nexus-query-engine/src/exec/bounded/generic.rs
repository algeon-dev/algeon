use std::{sync::Arc, time::Instant};

use super::{
    ActiveReservation, BoundedExecutionAttempt, BoundedPlan, CommittedPartial, ExecutionBudgetToken, ExecutionContext,
    FilterNode, GpuDataFrame, GpuResidentInputMap, GpuResidentSource, GpuResultHandle, LocalParquetSource,
    MemoryReservationRequest, NodeId, PlanNode, ProjectionNode, QueryError, QueryErrorCode, QueryMetrics, QueryOutput,
    QueryPlan, ReservationKind, ResidentGpuFrame, Result, ResultSink, SourceChunk, SourceChunkPlanningPolicy,
    concatenate_source_read_outputs, execute_bounded_aggregate_to_frame, execute_compiled_bounded_aggregate_from_frame,
    execute_edge_normalize, execute_filter_with_budget, execute_limit, execute_projection_with_budget,
    execute_projection_with_budget_and_prepared, execute_sort, execute_source_adjacent_chunk_projection,
    execute_source_adjacent_chunk_unary, execute_window, execution_budget_token_for_boundary, execution_overrides,
    materialize_bounded_to_host, observe_device_memory_bytes, plan_local_parquet_source_chunks_with_execution_context,
    projection_regexp_replace_prepare, read_gpu_resident_source, read_literal_batch_source,
    read_local_parquet_chunks_required_only, read_local_parquet_chunks_required_only_with_observation,
    reject_bounded_join_plan, release_source_read_outputs_reverse, unsupported_bounded_shape,
    validate_local_parquet_source,
};
#[cfg(feature = "cugraph")]
use super::{DataFrame, execute_graph_algorithm_sanitized, graph_preparation_bypass_input};
#[cfg(feature = "cugraph")]
use crate::exec::graph_algorithm::{
    bounded_betweenness::*, diagnostics::*, graph_empty_input_prevalidated, memory::*, statistics::*,
};
#[cfg(not(feature = "cugraph"))]
use crate::exec::graph_algorithm_feature_disabled;
use crate::exec::source::{ScheduledLocalParquetReadOutput, record_bounded_parquet_source_metrics};

pub(in crate::exec::bounded) fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
    observation_scope: Option<&crate::observability::RuntimeExecutionObservationScope>,
) -> Result<QueryOutput> {
    context.check_cancellation("nexus_query_engine.exec.bounded.generic_query_plan")?;
    if plan.contains_join() {
        reject_bounded_join_plan(context, plan)?;
    }

    let Some(device_budget_bytes) = context.bounded_device_budget_bytes() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut execution = BoundedExecutionAttempt::new(context, device_budget_bytes, metrics, observation_scope)?;
    let mut committed = {
        let augmented_gpu_inputs = execution.reservations().augment_gpu_inputs(gpu_inputs)?;
        execute_bounded_generic_node(&mut execution, plan, plan.sink().input(), &augmented_gpu_inputs)?
    };

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let live_reservation = committed.reservation.transfer_grant_live_reservation();
            let frame = committed
                .frame
                .with_producer_execution_scope(crate::admission::canonical_execution_scope_fingerprint(plan));
            let handle = match GpuResultHandle::new_with_live_reservation(
                execution.context(),
                frame,
                Arc::clone(plan.output_schema()),
                live_reservation,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    execution.release(committed.reservation);
                    return Err(error);
                },
            };
            execution.release(committed.reservation);
            debug_assert_eq!(execution.reservations().reserved_bytes(), 0);
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
            debug_assert_eq!(execution.reservations().reserved_bytes(), 0);
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
                execute_bounded_generic_gpu_resident_source(execution, source, id, gpu_inputs)
            },
            crate::source::SourceNode::LiteralBatch(source) => {
                execute_bounded_generic_literal_batch_source(execution, source, id)
            },
        },
        PlanNode::Filter(filter) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(execution, plan, id)?
            {
                return Ok(frame);
            }
            let input = execute_bounded_generic_node(execution, plan, filter.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_filter", "bounded_generic_filter_output", input, true, |context, input, token, metrics| {
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
            execute_bounded_generic_projection(execution, id, "bounded_generic_projection", "bounded_generic_projection_output", projection, input)
        },
        PlanNode::Window(window) => {
            let input = execute_bounded_generic_node(execution, plan, window.input, gpu_inputs)?;
            let input_schema = plan
                .node(window.input)
                .expect("validated window input exists")
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            execute_bounded_generic_unary(execution, id, "bounded_generic_window", "bounded_generic_window_output", input, false, |context, input, _token, metrics| {
                    execute_window(context, window, &input_schema, input, &output_schema, metrics)
                })
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(execution, plan, sort.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_sort", "bounded_generic_sort_output", input, false, |context, input, _token, metrics| execute_sort(context, sort, input, metrics))
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(execution, plan, limit.input, gpu_inputs)?;
            execute_bounded_generic_unary(execution, id, "bounded_generic_limit", "bounded_generic_limit_output", input, false, |context, input, _token, metrics| execute_limit(context, limit, input, metrics))
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
        PlanNode::VectorAlgorithm(vector) => Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.bounded.generic_vector_algorithm",
            "bounded vector algorithms must be rejected by capability-first admission before generic execution",
        )
        .with_fact("node_id", id)
        .with_fact("algorithm", vector.spec.algorithm.as_str())
        .with_fact("reason", "bounded_vector_admission_bypassed")),
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
            let input_reserved_bytes = input.reservation.bytes();
            let input_reservation = input.reservation;
            let result = execute_compiled_bounded_aggregate_from_frame(execution, id, aggregate, &input_schema, &output_schema, input.frame, input_reserved_bytes);
            execution.release(input_reservation);
            result
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
    mut input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    if !input.frame.has_live_reservation() {
        let source_reservation = input.reservation.transfer_grant_live_reservation();
        input.frame = input.frame.with_transferred_live_reservation(source_reservation);
    }
    execute_bounded_generic_unary(
        execution,
        node_id,
        "bounded_generic_edge_normalize",
        "bounded_generic_edge_normalize_output",
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

    let output_reservation_bytes =
        inputs.iter().fold(0u64, |total, input| total.saturating_add(input.reservation.bytes()));
    let mut output_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            output_reservation_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("bounded_union_concat_output")
        .with_node_id(node_id)
        .with_stage("bounded_union_concat")
        .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_resident_frames_reverse(execution, inputs);
            return Err(error);
        },
    };

    let input_count = inputs.len();
    let mut input_reservations = Vec::with_capacity(input_count);
    let output = (|| {
        let mut frames = Vec::with_capacity(input_count);
        for input in inputs {
            frames.push(input.frame.into_frame());
            input_reservations.push(input.reservation);
        }
        let first = frames.remove(0);
        let rest = frames.iter().collect::<Vec<_>>();
        let overrides_lease = execution_overrides(execution.context())?;
        let overrides = overrides_lease.overrides();
        let timer = Instant::now();
        let output = first.with_execution(&overrides).concat_many(&rest).map_err(|error| {
            QueryError::from_cudf(
                "nexus_query_engine.bounded.union.concat",
                "failed to concatenate bounded native union inputs",
                error,
            )
            .with_fact("node_id", node_id)
            .with_fact("input_count", input_count)
        })?;
        execution.metrics_mut().operator_time += timer.elapsed();

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, execution.metrics_mut())?;
        let output_request = MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            output_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary("bounded_union_concat_output")
        .with_node_id(node_id)
        .with_stage("bounded_union_concat")
        .with_attempt(1, 1);
        if output_bytes > output_reservation.bytes() {
            let delta = output_bytes.saturating_sub(output_reservation.bytes());
            execution.try_grow(&mut output_reservation, output_request.clone().with_requested_bytes(delta))?;
        } else {
            execution.shrink_to_observed_with_request(&mut output_reservation, output_bytes, output_request.clone());
        }

        Ok::<_, QueryError>(output_frame)
    })();
    release_reservations_reverse(execution, input_reservations);
    match output {
        Ok(frame) => Ok(ResidentGpuFrame { frame, reservation: output_reservation }),
        Err(error) => {
            execution.release(output_reservation);
            Err(error)
        },
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
                release_resident_frames_reverse(
                    execution,
                    std::iter::once(input).chain(side_inputs.into_values()).collect(),
                );
                return Err(error);
            },
        }
    }
    execute_bounded_graph_algorithm_from_inputs(execution, plan, node_id, graph, input, side_inputs)
}

#[cfg(feature = "cugraph")]
pub(in crate::exec::bounded) fn execute_bounded_graph_algorithm_from_inputs(
    execution: &mut BoundedExecutionAttempt<'_>,
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    input: ResidentGpuFrame,
    side_input_frames: std::collections::BTreeMap<NodeId, ResidentGpuFrame>,
) -> Result<ResidentGpuFrame> {
    let ResidentGpuFrame { frame: mut input_frame, reservation: input_reservation } = input;
    let query_memory_grant = match execution.context().memory_grant() {
        Ok(grant) => grant,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    // A tracker and a handle-carried grant can describe the same normalized
    // frame, so use only their larger charge. The retained pre-normalization
    // source is a distinct live owner and receives its own bounded credit.
    let already_accounted_normalized_graph_input_bytes =
        input_reservation.charged_bytes().max(input_frame.live_reservation_bytes_for(query_memory_grant.as_ref()));
    let retained_normalization_source_bytes =
        input_frame.retained_normalization_source_bytes_for(query_memory_grant.as_ref());
    let _retained_normalization_source = input_frame.take_retained_normalization_source();
    let input_rows = input_frame.num_rows() as u64;
    let input_columns = input_frame.num_columns() as u64;
    let input_graph_residency = input_frame.graph_residency().cloned();
    // Capture the source-bound vertex evidence before null filtering. The
    // descriptor may be dropped from the transformed frame to keep graph-cache
    // identity strict, but a filter can only shrink this edge rowset and the
    // footer bound remains a conservative memory-admission upper bound.
    let source_vertex_estimate =
        crate::exec::graph_algorithm::memory::source_graph_vertex_estimate(input_graph_residency.as_ref());
    let source_raw_input_edges =
        crate::exec::graph_algorithm::memory::source_graph_raw_edge_row_count(input_graph_residency.as_ref());
    let prior_null_edge_rows_dropped = graph_input_null_edge_rows_dropped(execution.metrics_mut(), graph.input);
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
        execution.metrics_mut(),
    ) {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let sanitized_input_rows = filtered_input_frame.num_rows() as u64;
    let null_edge_rows_dropped =
        prior_null_edge_rows_dropped.saturating_add(input_rows.saturating_sub(sanitized_input_rows));
    // EdgeNormalize projects exactly the graph columns and has already
    // transferred ownership to `input_reservation`. Its physical tracker
    // charge can include native alias bytes beyond the semantic row widths,
    // but it still names just this one live normalized graph frame. A later
    // null filter may materialize a different frame, so retain the ordinary
    // semantic cap in that case.
    let exact_normalized_graph_input_charge_bytes = (graph.spec.edge_normalize_inserted && null_edge_rows_dropped == 0)
        .then_some(already_accounted_normalized_graph_input_bytes);
    let mut filtered_input = GpuDataFrame::new(filtered_input_frame);
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        filtered_input = filtered_input.with_graph_residency(graph_residency);
    }
    // Resolve pair cardinality before reading side inputs so the admission bound is enforceable.
    let bounded_pair_cardinality = match crate::graph_memory::bounded_similarity_pair_cardinality_for_plan(plan, graph)
    {
        Ok(evidence) => evidence,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let mut live_reservations = vec![input_reservation];
    let mut side_inputs = std::collections::BTreeMap::new();
    let mut side_input_accounted_bytes = std::collections::BTreeMap::new();
    for (input_id, frame) in side_input_frames {
        // A literal side input is charged by this tracker, while a GPU-resident
        // handle may already retain bytes in this query grant. Keep the two
        // sources distinct from the frame's raw allocation so similarity can
        // credit only live-accounted pair columns from its graph reservation.
        let accounted_bytes =
            frame.reservation.charged_bytes().max(frame.frame.live_reservation_bytes_for(query_memory_grant.as_ref()));
        side_input_accounted_bytes.insert(input_id, accounted_bytes);
        live_reservations.push(frame.reservation);
        side_inputs.insert(input_id, frame.frame);
    }
    if let Err(error) = validate_betweenness_static_seed_policy(&graph.spec) {
        release_reservations_reverse(execution, live_reservations);
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
            release_reservations_reverse(execution, live_reservations);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let mut runtime_spec = RuntimeGraphSpecResolution::CheckedNoRewrite;
    let mut graph_reservation = None;
    let already_accounted_similarity_pair_input_bytes = match &graph.spec.params {
        crate::plan::GraphAlgorithmParams::Similarity(params) => *side_input_accounted_bytes
            .get(&params.vertex_pairs.input)
            .expect("validated similarity pair input is materialized"),
        _ => 0,
    };
    let graph_memory_limit_bytes = if !empty_input_prevalidated
        && crate::graph_memory::graph_memory_reservation_required(graph.spec.algorithm)
    {
        let graph_statistics_probe = match graph_memory_vertex_estimate_with_reservation(
            execution,
            node_id,
            &graph.spec,
            filtered_input.frame(),
        ) {
            Ok(probe) => probe,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        if let Some(probe) = &graph_statistics_probe
            && let Err(error) =
                validate_betweenness_seed_policy_for_vertex_count(&graph.spec, probe.vertex_estimate.count)
        {
            release_reservations_reverse(execution, live_reservations);
            return Err(error.with_fact_if_absent("node_id", node_id));
        }
        runtime_spec =
            runtime_graph_spec_resolution_after_bounded_seed_validation(&graph.spec, graph_statistics_probe.is_some());
        let device_total_memory_bytes = match graph_device_total_memory_bytes(execution.context()) {
            Ok(device_total_memory_bytes) => device_total_memory_bytes,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let graph_reservable_remaining_bytes = match execution.reservations().reservable_remaining_bytes() {
            Ok(remaining) => remaining,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let retained_normalization_source_credit_cap_bytes =
            match execution.context().activated_retained_normalization_source_model_bytes(plan) {
                Ok(bytes) => bytes,
                Err(error) => {
                    release_reservations_reverse(execution, live_reservations);
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            };
        let report = match graph_memory_reservation_report(GraphMemoryReservationRequest {
            context: execution.context(),
            node_id,
            spec: &graph.spec,
            input: filtered_input.frame(),
            side_inputs: &side_inputs,
            input_edges: sanitized_input_rows,
            raw_input_edges: source_raw_input_edges,
            bounded_pair_cardinality,
            budget_bytes: Some(graph_reservable_remaining_bytes),
            device_total_memory_bytes,
            allow_nulls: false,
            vertex_estimate: graph_vertex_estimate_for_bounded_admission(
                graph_statistics_probe.as_ref().map(|probe| probe.vertex_estimate),
                source_vertex_estimate,
            ),
            statistics: graph_statistics_probe.as_ref().map(|probe| probe.statistics),
            already_accounted_normalized_graph_input_bytes,
            exact_normalized_graph_input_charge_bytes,
            retained_normalization_source_bytes,
            retained_normalization_source_credit_cap_bytes,
            already_accounted_similarity_pair_input_bytes,
        }) {
            Ok(report) => report,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let request = MemoryReservationRequest::new(
            ReservationKind::GraphAlgorithm,
            report.requested_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary(crate::graph_memory::GRAPH_MEMORY_BOUNDARY)
        .with_node_id(node_id)
        .with_stage("graph_memory_admission");
        execution.metrics_mut().record_graph_claim_calibration(report.clone(), node_id);
        execution.metrics_mut().record_graph_reservation(report.clone());
        if report.outcome == crate::graph_memory::GraphReservationOutcome::Denied {
            execution
                .metrics_mut()
                .record_graph_claim_calibration_graph_memory_admission_denied(node_id, graph.spec.algorithm);
            let reserved_bytes = execution.reservations().reserved_bytes();
            execution.metrics_mut().record_reservation_denied_timeline_event(
                &request,
                crate::metrics::MemoryReservationEventAction::Reserve,
                graph_reservable_remaining_bytes,
                reserved_bytes,
            );
            release_reservations_reverse(execution, live_reservations);
            return Err(report
                .denied_error(&request, graph_reservable_remaining_bytes)
                .with_fact_if_absent("node_id", node_id));
        }

        // Similarity's graph-memory model is calibrated against the native
        // cuGraph allocation domain, which still needs the full model price
        // while constructing `VertexPairs` and coefficients. Its pair-column
        // overlap may already be live in the query grant, so the tracker
        // reserves only `report.requested_bytes` but the cuGraph allocator
        // retains the uncredited execution ceiling. These are deliberately
        // separate contracts: charging the full price double-counts the
        // source frame, while shrinking the native allocation limit would
        // under-provision the calibrated cuGraph call.
        let graph_execution_limit_bytes =
            report.model_requested_bytes.max(crate::graph_memory::GRAPH_MEMORY_BUDGET_TRACKING_MIN_LIMIT_BYTES);

        match execution.try_reserve_work(request) {
            Ok(reservation) => {
                graph_reservation = Some(reservation);
                Some(graph_execution_limit_bytes)
            },
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error
                    .with_fact_if_absent("reason", crate::graph_memory::GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON));
            },
        }
    } else if !empty_input_prevalidated {
        // BFS currently remains outside the live graph-tier reservation, but
        // its graph memory model is still a calibration claim that must be retained
        // and compared with the attempt's own measured peak.
        let device_total_memory_bytes = match graph_device_total_memory_bytes(execution.context()) {
            Ok(device_total_memory_bytes) => device_total_memory_bytes,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let graph_reservable_remaining_bytes = match execution.reservations().reservable_remaining_bytes() {
            Ok(remaining) => remaining,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let retained_normalization_source_credit_cap_bytes =
            match execution.context().activated_retained_normalization_source_model_bytes(plan) {
                Ok(bytes) => bytes,
                Err(error) => {
                    release_reservations_reverse(execution, live_reservations);
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            };
        let report = match graph_memory_reservation_report(GraphMemoryReservationRequest {
            context: execution.context(),
            node_id,
            spec: &graph.spec,
            input: filtered_input.frame(),
            side_inputs: &side_inputs,
            input_edges: sanitized_input_rows,
            raw_input_edges: source_raw_input_edges,
            bounded_pair_cardinality,
            budget_bytes: Some(graph_reservable_remaining_bytes),
            device_total_memory_bytes,
            allow_nulls: false,
            vertex_estimate: source_vertex_estimate,
            statistics: None,
            already_accounted_normalized_graph_input_bytes,
            exact_normalized_graph_input_charge_bytes,
            retained_normalization_source_bytes,
            retained_normalization_source_credit_cap_bytes,
            already_accounted_similarity_pair_input_bytes,
        }) {
            Ok(report) => report,
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        execution.metrics_mut().record_graph_claim_calibration(report, node_id);
        None
    } else {
        None
    };
    let output = match execute_graph_algorithm_sanitized(
        execution.context(),
        node_id,
        graph,
        runtime_spec,
        filtered_input,
        side_inputs,
        graph_memory_limit_bytes,
        input_rows,
        input_columns,
        null_edge_rows_dropped,
        empty_input_prevalidated,
        execution.metrics_mut(),
    ) {
        Ok(output) => output,
        Err(error) => {
            if let Some(reservation) = graph_reservation.take() {
                execution.release(reservation);
            }
            release_reservations_reverse(execution, live_reservations);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&output, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            if let Some(reservation) = graph_reservation.take() {
                execution.release(reservation);
            }
            release_reservations_reverse(execution, live_reservations);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, execution.reservations().budget())
            .with_memory_boundary("bounded_generic_graph_algorithm_output")
            .with_node_id(node_id)
            .with_stage("bounded_generic_graph_algorithm")
            .with_attempt(1, 1);
    let output_reservation = if let Some(mut graph_reservation) = graph_reservation {
        release_reservations_reverse(execution, live_reservations);
        if let Err(error) =
            execution.reconcile_work_reservation_to_output(&mut graph_reservation, output_bytes, &output_request)
        {
            execution.release(graph_reservation);
            return Err(error);
        }
        execution.transfer(graph_reservation, output_request)
    } else {
        match execution.try_reserve(output_request) {
            Ok(reservation) => {
                release_reservations_reverse(execution, live_reservations);
                reservation
            },
            Err(error) => {
                release_reservations_reverse(execution, live_reservations);
                return Err(error);
            },
        }
    };
    Ok(ResidentGpuFrame { frame: output, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn release_resident_frames_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut frames: Vec<ResidentGpuFrame>,
) {
    while let Some(frame) = frames.pop() {
        execution.release(frame.reservation);
    }
}

#[cfg(feature = "cugraph")]
fn graph_vertex_estimate_for_bounded_admission(
    runtime_vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
    source_vertex_estimate: Option<crate::graph_memory::GraphVertexEstimate>,
) -> Option<crate::graph_memory::GraphVertexEstimate> {
    runtime_vertex_estimate.or(source_vertex_estimate)
}

#[cfg(feature = "cugraph")]
fn graph_memory_vertex_estimate_with_reservation(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
) -> Result<Option<RuntimeGraphStatisticsProbe>> {
    let remaining = execution.reservations().reservable_remaining_bytes()?;
    if remaining == 0 {
        return Ok(None);
    }
    let model_input = graph_memory_model_input_for_frame(input, spec, input.num_rows() as u64, false)?;
    let probe_bytes = crate::graph_memory::graph_runtime_statistics_probe_bytes(spec.algorithm, model_input, remaining);
    let request =
        MemoryReservationRequest::new(ReservationKind::GraphAlgorithm, probe_bytes, execution.reservations().budget())
            .with_memory_boundary(crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY)
            .with_node_id(node_id)
            .with_stage("graph_memory_vertex_estimate");
    let reservation = execution.try_reserve_work(request)?;
    let limit_bytes = reservation.bytes();
    let estimate = runtime_graph_statistics_probe_with_budget(
        execution.context(),
        input,
        spec,
        limit_bytes,
        execution.metrics_mut(),
    );
    execution.release(reservation);
    match estimate {
        Ok(estimate) => Ok(Some(estimate)),
        Err(error) if error.code() == QueryErrorCode::CudfResourceExhausted => Ok(None),
        Err(error) => Err(error),
    }
}

pub(in crate::exec::bounded) fn release_reservations_reverse(
    execution: &mut BoundedExecutionAttempt<'_>,
    mut input_reservations: Vec<ActiveReservation>,
) {
    while let Some(reservation) = input_reservations.pop() {
        execution.release(reservation);
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
    validate_local_parquet_source(source)?;
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
        execution.metrics_mut().record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        let frame = crate::exec::source::read_local_parquet_source(
            execution.context(),
            node_id,
            source,
            execution.metrics_mut(),
        )?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
        let reservation = execution.try_reserve(
            MemoryReservationRequest::new(
                ReservationKind::SourceRead,
                requested_bytes,
                execution.reservations().budget(),
            )
            .with_memory_boundary("bounded_generic_source_read")
            .with_node_id(node_id)
            .with_stage("bounded_generic_source_read")
            .with_attempt(1, 1),
        )?;
        return Ok(ResidentGpuFrame { frame, reservation });
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, execution.metrics_mut());

    let chunks =
        read_local_parquet_chunks_required_only(execution, source, &chunk_plan.chunks, "bounded_generic_source_read")?;

    concatenate_source_read_outputs(
        execution,
        node_id,
        chunks,
        "bounded_generic_source_concat_output",
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
    validate_local_parquet_source(source)?;
    let planning_policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
        source,
        execution.context().memory_control_policy(),
        execution.context().source_chunk_planning_mode(),
        execution.context().source_chunk_planning_options(),
    );
    let chunk_plan =
        plan_local_parquet_source_chunks_with_execution_context(node_id, source, planning_policy, execution.context())?;
    for chunk in &chunk_plan.chunks {
        execution.metrics_mut().record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        let frame = crate::exec::source::read_local_parquet_source(
            execution.context(),
            node_id,
            source,
            execution.metrics_mut(),
        )?;
        let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
        let reservation = execution.try_reserve(
            MemoryReservationRequest::new(
                ReservationKind::SourceRead,
                requested_bytes,
                execution.reservations().budget(),
            )
            .with_memory_boundary("bounded_generic_source_read")
            .with_node_id(node_id)
            .with_stage("bounded_generic_source_read")
            .with_attempt(1, 1),
        )?;
        let mut resident = ResidentGpuFrame { frame, reservation };
        for stage in stages {
            resident = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id, node } => execute_bounded_generic_unary(
                    execution,
                    *node_id,
                    "bounded_generic_source_filter",
                    "bounded_generic_source_filter_output",
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
                    "bounded_generic_source_projection_output",
                    node,
                    resident,
                )?,
            };
        }
        return Ok(resident);
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, execution.metrics_mut());

    let chunks = read_bounded_generic_source_adjacent_chunks(execution, source, &chunk_plan.chunks, stages, false)?;

    concatenate_source_read_outputs(
        execution,
        node_id,
        chunks,
        "bounded_generic_source_concat_output",
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn read_bounded_generic_source_adjacent_chunks(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    grow_source_to_observed: bool,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut transformed = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let source_chunk = match read_local_parquet_chunks_required_only_with_observation(
            execution,
            source,
            std::slice::from_ref(chunk),
            "bounded_generic_source_read",
            grow_source_to_observed,
        ) {
            Ok(source_chunk) => source_chunk,
            Err(error) => {
                release_source_read_outputs_reverse(execution, transformed);
                return Err(error);
            },
        };
        let mut staged = match apply_bounded_generic_source_adjacent_stages(execution, source_chunk, stages) {
            Ok(staged) => staged,
            Err(error) => {
                release_source_read_outputs_reverse(execution, transformed);
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
        let mut current = CommittedPartial { frame: source.frame, reservation: source.reservation };
        for stage in stages {
            let result = match stage {
                BoundedGenericSourceAdjacentStage::Filter { node_id, node } => execute_source_adjacent_chunk_unary(
                    execution,
                    *node_id,
                    "bounded_generic_source_filter",
                    "bounded_generic_source_filter_output",
                    &chunk,
                    current,
                    1,
                    1,
                    |context, input, token, metrics| execute_filter_with_budget(context, node, input, token, metrics),
                ),
                BoundedGenericSourceAdjacentStage::Projection { node_id, node } => {
                    execute_source_adjacent_chunk_projection(
                        execution,
                        *node_id,
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
                    release_source_read_outputs_reverse(execution, pending);
                    release_source_read_outputs_reverse(execution, transformed);
                    return Err(error);
                },
            };
        }
        let observed_bytes = current.reservation.bytes();
        transformed.push(ScheduledLocalParquetReadOutput {
            chunk,
            frame: current.frame,
            reservation: current.reservation,
            observed_bytes,
            observed_row_group_count: None,
        });
    }
    Ok(transformed)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_gpu_resident_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &GpuResidentSource,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(execution.context(), source, gpu_inputs, execution.metrics_mut())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let reservation = execution.borrow_resident(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("bounded_generic_gpu_resident_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_gpu_resident_source")
            .with_attempt(1, 1),
    );
    Ok(ResidentGpuFrame { frame, reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_literal_batch_source(
    execution: &mut BoundedExecutionAttempt<'_>,
    source: &crate::source::LiteralBatchSource,
    node_id: NodeId,
) -> Result<ResidentGpuFrame> {
    let frame = read_literal_batch_source(execution.context(), source, execution.metrics_mut())?;
    let requested_bytes = observe_device_memory_bytes(&frame, execution.metrics_mut())?;
    let reservation = execution.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, execution.reservations().budget())
            .with_memory_boundary("bounded_generic_literal_batch_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_literal_batch_source")
            .with_attempt(1, 1),
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}
pub(in crate::exec::bounded) fn execute_bounded_generic_unary(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_with_work_bytes(
        execution,
        node_id,
        stage,
        output_boundary,
        input,
        use_budget_token,
        None,
        execute,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_with_work_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    use_budget_token: bool,
    work_request_bytes: Option<u64>,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_consuming_input_with_reservation_kind(
        execution,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::OperatorOutput,
        use_budget_token,
        work_request_bytes,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary_with_work_bytes(
            execution,
            node_id,
            stage,
            output_boundary,
            input,
            true,
            None,
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
    execute_bounded_generic_regexp_projection(execution, node_id, stage, output_boundary, projection, input, None)
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary_from_frame_with_reservation_kind(
            execution,
            node_id,
            stage,
            output_boundary,
            input,
            input_reserved_bytes,
            ReservationKind::OperatorOutput,
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
    // Wrap frame in a transient resident reservation equal to input_reserved_bytes.
    let input_reservation = execution.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            input_reserved_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary(stage)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1),
    )?;
    // The "input" reservation here only scopes prepared-path headroom until transfer.
    let input = ResidentGpuFrame { frame: input, reservation: input_reservation };
    // Drop the ownership of the synthetic wrapper on error paths inside helper.
    execute_bounded_generic_regexp_projection(
        execution,
        node_id,
        stage,
        output_boundary,
        projection,
        input,
        Some(input_reserved_bytes),
    )
}

fn execute_bounded_generic_regexp_projection(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
    _seed_reserved_bytes: Option<u64>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let input_frame = input.frame;
    let input_reservation = input.reservation;

    // Start with a work reservation covering at least the input claim so prepare
    // can allocate retained state under an explicit budget token.
    let initial_work_bytes = input_reserved_bytes.max(1);
    let mut work_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(
            ReservationKind::OperatorOutput,
            initial_work_bytes,
            execution.reservations().budget(),
        )
        .with_memory_boundary(stage)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error);
        },
    };

    // Prepare without a stream reservation so retained prepared state keeps the
    // input frame domain. Mint a single execute token after reservation reconcile.
    let prepared =
        match projection_regexp_replace_prepare(execution.context(), projection, &input_frame, execution.metrics_mut())
        {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                // Defense in depth: no regexp after the gate above.
                let token = match execution_budget_token_for_boundary(
                    execution.context(),
                    execution.reservations().available_bytes_including(&work_reservation),
                    stage,
                    "nexus_query_engine.bounded_generic.budget_token",
                    "failed to create bounded native generic operator budget token",
                ) {
                    Ok(token) => token,
                    Err(error) => {
                        execution.release(work_reservation);
                        execution.release(input_reservation);
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                };
                match execute_projection_with_budget(
                    execution.context(),
                    projection,
                    input_frame,
                    &token,
                    execution.metrics_mut(),
                ) {
                    Ok(frame) => {
                        execution.release(input_reservation);
                        let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                execution.release(work_reservation);
                                return Err(error.with_fact_if_absent("node_id", node_id));
                            },
                        };
                        let output_request = MemoryReservationRequest::new(
                            ReservationKind::OperatorOutput,
                            output_bytes,
                            execution.reservations().budget(),
                        )
                        .with_memory_boundary(output_boundary)
                        .with_node_id(node_id)
                        .with_stage(stage)
                        .with_attempt(1, 1);
                        if let Err(error) = execution.reconcile_work_reservation_to_output(
                            &mut work_reservation,
                            output_bytes,
                            &output_request,
                        ) {
                            execution.release(work_reservation);
                            return Err(error);
                        }
                        let output_reservation = execution.transfer(work_reservation, output_request);
                        return Ok(ResidentGpuFrame { frame, reservation: output_reservation });
                    },
                    Err(error) => {
                        execution.release(work_reservation);
                        execution.release(input_reservation);
                        return Err(error.with_fact_if_absent("node_id", node_id));
                    },
                }
            },
            Err(error) => {
                execution.release(work_reservation);
                execution.release(input_reservation);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };

    let required = prepared.required_budget_bytes().max(input_reserved_bytes);
    if required > work_reservation.bytes() {
        let delta = required - work_reservation.bytes();
        if let Err(error) = execution.try_grow(
            &mut work_reservation,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, delta, execution.reservations().budget())
                .with_memory_boundary(stage)
                .with_node_id(node_id)
                .with_stage(stage)
                .with_attempt(1, 1),
        ) {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error);
        }
    }

    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.reservations().available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let result = execute_projection_with_budget_and_prepared(
        execution.context(),
        projection,
        input_frame,
        &token,
        Some(prepared),
        execution.metrics_mut(),
    );
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    execution.release(input_reservation);

    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, execution.reservations().budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) =
        execution.reconcile_work_reservation_to_output(&mut work_reservation, output_bytes, &output_request)
    {
        execution.release(work_reservation);
        return Err(error);
    }
    let output_reservation = execution.transfer(work_reservation, output_request);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_from_frame_with_reservation_kind(
        execution,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        ReservationKind::OperatorOutput,
        use_budget_token,
        execute,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_consuming_input_with_reservation_kind(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservation_kind: ReservationKind,
    use_budget_token: bool,
    work_request_bytes: Option<u64>,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = work_request_bytes.unwrap_or_else(|| input.reservation.bytes());
    let input_frame = input.frame;
    let input_reservation = input.reservation;
    let mut work_reservation = match execution.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, execution.reservations().budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            execution.release(input_reservation);
            return Err(error);
        },
    };
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            execution.context(),
            execution.reservations().available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                execution.release(work_reservation);
                execution.release(input_reservation);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(execution.context(), input_frame, token.as_ref(), execution.metrics_mut());
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(work_reservation);
            execution.release(input_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    execution.release(input_reservation);

    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(reservation_kind, output_bytes, execution.reservations().budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) =
        execution.reconcile_work_reservation_to_output(&mut work_reservation, output_bytes, &output_request)
    {
        execution.release(work_reservation);
        return Err(error);
    }
    let output_reservation = execution.transfer(work_reservation, output_request);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame_with_reservation_kind(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservation_kind: ReservationKind,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let mut work_reservation = execution.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, execution.reservations().budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
    )?;
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            execution.context(),
            execution.reservations().available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                execution.release(work_reservation);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(execution.context(), input, token.as_ref(), execution.metrics_mut());
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, execution.metrics_mut()) {
        Ok(bytes) => bytes,
        Err(error) => {
            execution.release(work_reservation);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(reservation_kind, output_bytes, execution.reservations().budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) =
        execution.reconcile_work_reservation_to_output(&mut work_reservation, output_bytes, &output_request)
    {
        execution.release(work_reservation);
        return Err(error);
    }
    let output_reservation = execution.transfer(work_reservation, output_request);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
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

#[cfg(all(test, feature = "cugraph"))]
mod tests {
    use super::graph_vertex_estimate_for_bounded_admission;
    use crate::graph_memory::{GraphVertexEstimate, GraphVertexEstimateSource};

    #[test]
    fn bounded_graph_admission_uses_footer_provenance_when_runtime_probe_is_unavailable() {
        let footer = GraphVertexEstimate::graph_metadata(4_894_081);

        assert_eq!(
            graph_vertex_estimate_for_bounded_admission(None, Some(footer)),
            Some(footer),
            "bounded graph admission must not fall back to 2E after a probe resource exhaustion"
        );

        let runtime = GraphVertexEstimate::runtime_exact_distinct(4_100_000);
        assert_eq!(
            graph_vertex_estimate_for_bounded_admission(Some(runtime), Some(footer)),
            Some(runtime),
            "the exact runtime probe remains stronger than the source upper bound"
        );
        assert_eq!(runtime.source, GraphVertexEstimateSource::RuntimeExactDistinct);
    }
}
