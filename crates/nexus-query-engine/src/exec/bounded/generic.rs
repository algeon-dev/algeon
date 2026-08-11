use super::*;
#[cfg(feature = "cugraph")]
use crate::exec::graph_algorithm::{
    bounded_betweenness::*, diagnostics::*, graph_empty_input_prevalidated, memory::*, statistics::*,
};
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
    let mut reservations =
        ReservationTracker::new_for_context(context, device_budget_bytes, metrics, observation_scope)?;
    let mut committed = {
        let augmented_gpu_inputs = reservations.augment_gpu_inputs(gpu_inputs)?;
        execute_bounded_generic_node(
            context,
            plan,
            plan.sink().input(),
            &augmented_gpu_inputs,
            &mut reservations,
            metrics,
        )?
    };

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let live_reservation = committed.reservation.transfer_grant_live_reservation();
            let handle = match GpuResultHandle::new_with_live_reservation(
                context,
                committed.frame,
                Arc::clone(plan.output_schema()),
                live_reservation,
            ) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation, metrics);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation, metrics);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                plan.node(plan.sink().input()).expect("validated sink input exists").output_schema(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_generic_node(
    context: &ExecutionContext,
    plan: &QueryPlan,
    id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    context.check_cancellation("nexus_query_engine.exec.bounded.generic_node")?;
    let node = plan.node(id).expect("validated bounded generic node exists");
    let frame = match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            crate::source::SourceNode::LocalParquet(source) => {
                execute_bounded_generic_local_parquet_source(context, source, id, reservations, metrics)
            },
            crate::source::SourceNode::GpuResident(source) => {
                execute_bounded_generic_gpu_resident_source(context, source, id, gpu_inputs, reservations, metrics)
            },
            crate::source::SourceNode::LiteralBatch(source) => {
                execute_bounded_generic_literal_batch_source(context, source, id, reservations, metrics)
            },
        },
        PlanNode::Filter(filter) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(context, plan, id, reservations, metrics)?
            {
                return Ok(frame);
            }
            let input = execute_bounded_generic_node(context, plan, filter.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_filter",
                "bounded_generic_filter_output",
                input,
                reservations,
                metrics,
                true,
                |context, input, token, metrics| {
                    execute_filter_with_budget(
                        context,
                        filter,
                        input,
                        token.expect("budget token is created for bounded generic filter stages"),
                        metrics,
                    )
                },
            )
        },
        PlanNode::Projection(projection) => {
            if let Some(frame) =
                try_execute_bounded_generic_source_adjacent_stream(context, plan, id, reservations, metrics)?
            {
                return Ok(frame);
            }
            let input =
                execute_bounded_generic_node(context, plan, projection.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_projection(
                context,
                id,
                "bounded_generic_projection",
                "bounded_generic_projection_output",
                projection,
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::Window(window) => {
            let input = execute_bounded_generic_node(context, plan, window.input, gpu_inputs, reservations, metrics)?;
            let input_schema = plan
                .node(window.input)
                .expect("validated window input exists")
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_window",
                "bounded_generic_window_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| {
                    execute_window(context, window, &input_schema, input, &output_schema, metrics)
                },
            )
        },
        PlanNode::Sort(sort) => {
            let input = execute_bounded_generic_node(context, plan, sort.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_sort",
                "bounded_generic_sort_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_sort(context, sort, input, metrics),
            )
        },
        PlanNode::Limit(limit) => {
            let input = execute_bounded_generic_node(context, plan, limit.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_unary(
                context,
                id,
                "bounded_generic_limit",
                "bounded_generic_limit_output",
                input,
                reservations,
                metrics,
                false,
                |context, input, _token, metrics| execute_limit(context, limit, input, metrics),
            )
        },
        PlanNode::Union(union) => {
            let inputs = union
                .inputs
                .iter()
                .map(|input| execute_bounded_generic_node(context, plan, *input, gpu_inputs, reservations, metrics))
                .collect::<Result<Vec<_>>>()?;
            execute_bounded_generic_union(context, id, inputs, reservations, metrics)
        },
        PlanNode::EdgeNormalize(edge_normalize) => {
            let input =
                execute_bounded_generic_node(context, plan, edge_normalize.input, gpu_inputs, reservations, metrics)?;
            execute_bounded_generic_edge_normalize_from_input(
                context,
                id,
                edge_normalize,
                input,
                reservations,
                metrics,
            )
        },
        PlanNode::GraphAlgorithm(graph) => {
            execute_bounded_generic_graph_algorithm(context, plan, id, graph, gpu_inputs, reservations, metrics)
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
                    Ok(bounded) => execute_bounded_aggregate_to_frame(context, &bounded, reservations, metrics)?,
                    Err(_error) => {
                        execute_bounded_generic_node(context, plan, aggregate.input, gpu_inputs, reservations, metrics)?
                    },
                },
                _ => execute_bounded_generic_node(context, plan, aggregate.input, gpu_inputs, reservations, metrics)?,
            };
            let input_schema = plan
                .node(aggregate.input)
                .expect("validated aggregate input exists")
                .output_schema()
                .clone();
            let output_schema = node.output_schema().clone();
            let input_reserved_bytes = input.reservation.bytes();
            let input_reservation = input.reservation;
            let result = execute_compiled_bounded_aggregate_from_frame(
                context,
                id,
                aggregate,
                &input_schema,
                &output_schema,
                input.frame,
                input_reserved_bytes,
                reservations,
                metrics,
            );
            reservations.release(input_reservation, metrics);
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

fn execute_bounded_generic_edge_normalize_from_input(
    context: &ExecutionContext,
    node_id: NodeId,
    edge_normalize: &crate::plan::EdgeNormalizeNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary(
        context,
        node_id,
        "bounded_generic_edge_normalize",
        "bounded_generic_edge_normalize_output",
        input,
        reservations,
        metrics,
        false,
        |context, input, _token, metrics| execute_edge_normalize(context, node_id, edge_normalize, input, metrics),
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_union(
    context: &ExecutionContext,
    node_id: NodeId,
    inputs: Vec<ResidentGpuFrame>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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
    let mut output_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_reservation_bytes, reservations.budget())
            .with_memory_boundary("bounded_union_concat_output")
            .with_node_id(node_id)
            .with_stage("bounded_union_concat")
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            release_resident_frames_reverse(reservations, inputs, metrics);
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
        let overrides_lease = execution_overrides(context)?;
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
        metrics.operator_time += timer.elapsed();

        let output_frame = GpuDataFrame::new(output);
        let output_bytes = observe_device_memory_bytes(&output_frame, metrics)?;
        let output_request =
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
                .with_memory_boundary("bounded_union_concat_output")
                .with_node_id(node_id)
                .with_stage("bounded_union_concat")
                .with_attempt(1, 1);
        if output_bytes > output_reservation.bytes() {
            let delta = output_bytes.saturating_sub(output_reservation.bytes());
            reservations.try_grow(
                &mut output_reservation,
                output_request.clone().with_requested_bytes(delta),
                metrics,
            )?;
        } else {
            reservations.shrink_to_observed_with_request(
                &mut output_reservation,
                output_bytes,
                output_request.clone(),
                metrics,
            );
        }

        Ok::<_, QueryError>(output_frame)
    })();
    release_reservations_reverse(reservations, input_reservations, metrics);
    match output {
        Ok(frame) => Ok(ResidentGpuFrame { frame, reservation: output_reservation }),
        Err(error) => {
            reservations.release(output_reservation, metrics);
            Err(error)
        },
    }
}

#[cfg(not(feature = "cugraph"))]
fn execute_bounded_generic_graph_algorithm(
    _context: &ExecutionContext,
    _plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    _gpu_inputs: &GpuResidentInputMap,
    _reservations: &mut ReservationTracker,
    _metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    Err(graph_algorithm_feature_disabled(node_id, graph))
}

#[cfg(feature = "cugraph")]
fn execute_bounded_generic_graph_algorithm(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    graph: &crate::plan::GraphAlgorithmNode,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input = if let Some(raw_input_id) = graph_preparation_bypass_input(plan, graph) {
        let raw_input = execute_bounded_generic_node(context, plan, raw_input_id, gpu_inputs, reservations, metrics)?;
        let PlanNode::EdgeNormalize(edge_normalize) =
            &plan.node(graph.input).expect("preparation bypass validated graph input").kind
        else {
            unreachable!("preparation bypass requires an edge-normalize input")
        };
        execute_bounded_generic_edge_normalize_from_input(
            context,
            graph.input,
            edge_normalize,
            raw_input,
            reservations,
            metrics,
        )?
    } else {
        execute_bounded_generic_node(context, plan, graph.input, gpu_inputs, reservations, metrics)?
    };
    let ResidentGpuFrame { frame: input_frame, reservation: input_reservation } = input;
    let query_memory_grant = match context.memory_grant() {
        Ok(grant) => grant,
        Err(error) => {
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    // The cold graph model includes the normalized semantic edge frame.
    // Retain only the bytes that this tracker or the same query grant already
    // accounts, then cap the credit at that physically live frame in
    // graph_memory.rs. Source provenance is not a retained DataFrame owner.
    // A source reservation and a handle-carried grant charge can describe the
    // same live frame. They are alternative owners, never concurrent bytes;
    // retain the larger exact charge rather than counting that frame twice.
    let already_accounted_graph_input_bytes =
        input_reservation.charged_bytes().max(input_frame.live_reservation_bytes_for(query_memory_grant.as_ref()));
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
    let prior_null_edge_rows_dropped = graph_input_null_edge_rows_dropped(metrics, graph.input);
    let null_filter_scope = super::edge_null_filter::EdgeNullFilterScope {
        weight: graph.spec.algorithm.consumes_weight(),
        edge_id: graph.spec.edge_columns.edge_id.is_some(),
    };
    let filtered_input_frame = match super::edge_null_filter::filter_null_edge_rows(
        context,
        node_id,
        "graph_edge_null_filter",
        input_frame.into_frame(),
        &graph.spec.edge_columns,
        null_filter_scope,
        metrics,
    ) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input_reservation, metrics);
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
        .then_some(already_accounted_graph_input_bytes);
    let mut filtered_input = GpuDataFrame::new(filtered_input_frame);
    if null_edge_rows_dropped == 0
        && let Some(graph_residency) = input_graph_residency
    {
        filtered_input = filtered_input.with_graph_residency(graph_residency);
    }
    // Bounded admission must prove the candidate-pair cardinality from the
    // plan before it evaluates the side-input relation.
    let strict_pair_cardinality = match crate::graph_memory::strict_similarity_pair_cardinality_for_plan(plan, graph) {
        Ok(evidence) => evidence,
        Err(error) => {
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let mut live_reservations = vec![input_reservation];
    let mut side_inputs = std::collections::BTreeMap::new();
    let mut side_input_accounted_bytes = std::collections::BTreeMap::new();
    for input_id in graph.inputs.iter().skip(1) {
        match execute_bounded_generic_node(context, plan, *input_id, gpu_inputs, reservations, metrics) {
            Ok(frame) => {
                // A literal side input is charged by this tracker, while a
                // GPU-resident handle may already retain bytes in this query
                // grant. Keep the two sources distinct from the frame's raw
                // allocation so similarity can credit only live-accounted
                // pair columns from its incremental graph reservation.
                let accounted_bytes = frame
                    .reservation
                    .charged_bytes()
                    .max(frame.frame.live_reservation_bytes_for(query_memory_grant.as_ref()));
                side_input_accounted_bytes.insert(*input_id, accounted_bytes);
                live_reservations.push(frame.reservation);
                side_inputs.insert(*input_id, frame.frame);
            },
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error);
            },
        }
    }
    if let Err(error) = validate_betweenness_static_seed_policy(&graph.spec) {
        release_reservations_reverse(reservations, live_reservations, metrics);
        return Err(error.with_fact_if_absent("node_id", node_id));
    }
    let empty_input_prevalidated = match graph_empty_input_prevalidated(
        context,
        graph,
        filtered_input.frame(),
        &side_inputs,
        input_rows,
        sanitized_input_rows,
    ) {
        Ok(empty_input_prevalidated) => empty_input_prevalidated,
        Err(error) => {
            release_reservations_reverse(reservations, live_reservations, metrics);
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
            context,
            node_id,
            &graph.spec,
            filtered_input.frame(),
            reservations,
            metrics,
        ) {
            Ok(probe) => probe,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        if let Some(probe) = &graph_statistics_probe
            && let Err(error) =
                validate_betweenness_seed_policy_for_vertex_count(&graph.spec, probe.vertex_estimate.count)
        {
            release_reservations_reverse(reservations, live_reservations, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        }
        runtime_spec =
            runtime_graph_spec_resolution_after_bounded_seed_validation(&graph.spec, graph_statistics_probe.is_some());
        let device_total_memory_bytes = match graph_device_total_memory_bytes(context) {
            Ok(device_total_memory_bytes) => device_total_memory_bytes,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let graph_reservable_remaining_bytes = match reservations.reservable_remaining_bytes() {
            Ok(remaining) => remaining,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let report = match graph_memory_reservation_report(GraphMemoryReservationRequest {
            context,
            node_id,
            spec: &graph.spec,
            input: filtered_input.frame(),
            side_inputs: &side_inputs,
            input_edges: sanitized_input_rows,
            raw_input_edges: source_raw_input_edges,
            strict_pair_cardinality,
            budget_bytes: Some(graph_reservable_remaining_bytes),
            device_total_memory_bytes,
            allow_nulls: false,
            vertex_estimate: graph_vertex_estimate_for_bounded_admission(
                graph_statistics_probe.as_ref().map(|probe| probe.vertex_estimate),
                source_vertex_estimate,
            ),
            statistics: graph_statistics_probe.as_ref().map(|probe| probe.statistics),
            already_accounted_graph_input_bytes,
            exact_normalized_graph_input_charge_bytes,
            already_accounted_similarity_pair_input_bytes,
        }) {
            Ok(report) => report,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let request = MemoryReservationRequest::new(
            ReservationKind::GraphAlgorithm,
            report.requested_bytes,
            reservations.budget(),
        )
        .with_memory_boundary(crate::graph_memory::GRAPH_MEMORY_BOUNDARY)
        .with_node_id(node_id)
        .with_stage("graph_memory_admission");
        metrics.record_graph_claim_calibration(report.clone(), node_id);
        metrics.record_graph_reservation(report.clone());
        if report.outcome == crate::graph_memory::GraphReservationOutcome::Denied {
            metrics.record_graph_claim_calibration_graph_memory_admission_denied(node_id, graph.spec.algorithm);
            metrics.record_reservation_denied_timeline_event(
                &request,
                crate::metrics::MemoryReservationEventAction::Reserve,
                graph_reservable_remaining_bytes,
                reservations.reserved_bytes(),
            );
            release_reservations_reverse(reservations, live_reservations, metrics);
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

        match reservations.try_reserve_work(request, metrics) {
            Ok(reservation) => {
                graph_reservation = Some(reservation);
                Some(graph_execution_limit_bytes)
            },
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error
                    .with_fact_if_absent("reason", crate::graph_memory::GRAPH_MEMORY_ESTIMATE_EXCEEDS_BUDGET_REASON));
            },
        }
    } else if !empty_input_prevalidated {
        // BFS currently remains outside the live graph-tier reservation, but
        // its graph memory model is still an I11 claim that must be retained
        // and compared with the attempt's own measured peak.
        let device_total_memory_bytes = match graph_device_total_memory_bytes(context) {
            Ok(device_total_memory_bytes) => device_total_memory_bytes,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let graph_reservable_remaining_bytes = match reservations.reservable_remaining_bytes() {
            Ok(remaining) => remaining,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        let report = match graph_memory_reservation_report(GraphMemoryReservationRequest {
            context,
            node_id,
            spec: &graph.spec,
            input: filtered_input.frame(),
            side_inputs: &side_inputs,
            input_edges: sanitized_input_rows,
            raw_input_edges: source_raw_input_edges,
            strict_pair_cardinality,
            budget_bytes: Some(graph_reservable_remaining_bytes),
            device_total_memory_bytes,
            allow_nulls: false,
            vertex_estimate: source_vertex_estimate,
            statistics: None,
            already_accounted_graph_input_bytes,
            exact_normalized_graph_input_charge_bytes,
            already_accounted_similarity_pair_input_bytes,
        }) {
            Ok(report) => report,
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        };
        metrics.record_graph_claim_calibration(report, node_id);
        None
    } else {
        None
    };
    let output = match execute_graph_algorithm_sanitized(
        context,
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
        metrics,
    ) {
        Ok(output) => output,
        Err(error) => {
            if let Some(reservation) = graph_reservation.take() {
                reservations.release(reservation, metrics);
            }
            release_reservations_reverse(reservations, live_reservations, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let output_bytes = match observe_device_memory_bytes(&output, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            if let Some(reservation) = graph_reservation.take() {
                reservations.release(reservation, metrics);
            }
            release_reservations_reverse(reservations, live_reservations, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_graph_algorithm_output")
            .with_node_id(node_id)
            .with_stage("bounded_generic_graph_algorithm")
            .with_attempt(1, 1);
    let output_reservation = if let Some(mut graph_reservation) = graph_reservation {
        release_reservations_reverse(reservations, live_reservations, metrics);
        if let Err(error) = reconcile_work_reservation_to_output(
            reservations,
            &mut graph_reservation,
            output_bytes,
            &output_request,
            metrics,
        ) {
            reservations.release(graph_reservation, metrics);
            return Err(error);
        }
        reservations.transfer(graph_reservation, output_request, metrics)
    } else {
        match reservations.try_reserve(output_request, metrics) {
            Ok(reservation) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                reservation
            },
            Err(error) => {
                release_reservations_reverse(reservations, live_reservations, metrics);
                return Err(error);
            },
        }
    };
    Ok(ResidentGpuFrame { frame: output, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn release_resident_frames_reverse(
    reservations: &mut ReservationTracker,
    mut frames: Vec<ResidentGpuFrame>,
    metrics: &mut QueryMetrics,
) {
    while let Some(frame) = frames.pop() {
        reservations.release(frame.reservation, metrics);
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
    context: &ExecutionContext,
    node_id: NodeId,
    spec: &crate::plan::GraphAlgorithmSpec,
    input: &DataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<RuntimeGraphStatisticsProbe>> {
    let remaining = reservations.reservable_remaining_bytes()?;
    if remaining == 0 {
        return Ok(None);
    }
    let model_input = graph_memory_model_input_for_frame(input, spec, input.num_rows() as u64, false)?;
    let probe_bytes = crate::graph_memory::graph_runtime_statistics_probe_bytes(spec.algorithm, model_input, remaining);
    let request = MemoryReservationRequest::new(ReservationKind::GraphAlgorithm, probe_bytes, reservations.budget())
        .with_memory_boundary(crate::graph_memory::GRAPH_VERTEX_ESTIMATE_BOUNDARY)
        .with_node_id(node_id)
        .with_stage("graph_memory_vertex_estimate");
    let reservation = reservations.try_reserve_work(request, metrics)?;
    let limit_bytes = reservation.bytes();
    let estimate = runtime_graph_statistics_probe_with_budget(context, input, spec, limit_bytes, metrics);
    reservations.release(reservation, metrics);
    match estimate {
        Ok(estimate) => Ok(Some(estimate)),
        Err(error) if error.code() == QueryErrorCode::CudfResourceExhausted => Ok(None),
        Err(error) => Err(error),
    }
}

pub(in crate::exec::bounded) fn release_reservations_reverse(
    reservations: &mut ReservationTracker,
    mut input_reservations: Vec<ActiveReservation>,
    metrics: &mut QueryMetrics,
) {
    while let Some(reservation) = input_reservations.pop() {
        reservations.release(reservation, metrics);
    }
}

#[derive(Clone, Copy)]
enum BoundedGenericSourceAdjacentStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode },
}

fn try_execute_bounded_generic_source_adjacent_stream(
    context: &ExecutionContext,
    plan: &QueryPlan,
    root_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Option<ResidentGpuFrame>> {
    let Some((source_id, source, stages)) = collect_bounded_generic_source_adjacent_stages(plan, root_id)? else {
        return Ok(None);
    };
    if source.has_position_delete_visible_scan_limit() {
        return Ok(None);
    }
    execute_bounded_generic_local_parquet_source_with_stages(context, source, source_id, &stages, reservations, metrics)
        .map(Some)
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
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source)?;
    let chunk_plan = plan_local_parquet_source_chunks_with_execution_context(
        node_id,
        source,
        SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
            source,
            context.memory_control_policy(),
            context.source_chunk_planning_mode(),
            context.source_chunk_planning_options(),
        ),
        context,
    )?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.generic.source",
            "bounded generic local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = read_local_parquet_chunks_required_only(
        context,
        source,
        &chunk_plan.chunks,
        reservations,
        metrics,
        "bounded_generic_source_read",
    )?;

    concatenate_source_read_outputs(
        context,
        node_id,
        chunks,
        reservations,
        metrics,
        "bounded_generic_source_concat_output",
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn execute_bounded_generic_local_parquet_source_with_stages(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    node_id: NodeId,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    validate_local_parquet_source(source)?;
    let planning_policy = SourceChunkPlanningPolicy::from_memory_policy_with_mode_and_options(
        source,
        context.memory_control_policy(),
        context.source_chunk_planning_mode(),
        context.source_chunk_planning_options(),
    );
    let chunk_plan =
        plan_local_parquet_source_chunks_with_execution_context(node_id, source, planning_policy, context)?;
    for chunk in &chunk_plan.chunks {
        metrics.record_source_chunk_planned(chunk.row_group_count() as u64);
    }
    if chunk_plan.chunks.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.bounded.generic.source",
            "bounded generic local parquet source requires at least one planned source chunk",
        )
        .with_fact("source_node_id", node_id));
    }
    record_bounded_parquet_source_metrics(source, &chunk_plan, metrics);

    let chunks = read_bounded_generic_source_adjacent_chunks(
        context,
        source,
        &chunk_plan.chunks,
        stages,
        false,
        reservations,
        metrics,
    )?;

    concatenate_source_read_outputs(
        context,
        node_id,
        chunks,
        reservations,
        metrics,
        "bounded_generic_source_concat_output",
        "nexus_query_engine.bounded.generic.source_concat",
        "failed to concatenate bounded generic local parquet source chunks",
    )
}

fn read_bounded_generic_source_adjacent_chunks(
    context: &ExecutionContext,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    grow_source_to_observed: bool,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<Vec<ScheduledLocalParquetReadOutput>> {
    let mut transformed = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let source_chunk = match read_local_parquet_chunks_required_only_with_observation(
            context,
            source,
            std::slice::from_ref(chunk),
            reservations,
            metrics,
            "bounded_generic_source_read",
            grow_source_to_observed,
        ) {
            Ok(source_chunk) => source_chunk,
            Err(error) => {
                release_source_read_outputs_reverse(reservations, transformed, metrics);
                return Err(error);
            },
        };
        let mut staged =
            match apply_bounded_generic_source_adjacent_stages(context, source_chunk, stages, reservations, metrics) {
                Ok(staged) => staged,
                Err(error) => {
                    release_source_read_outputs_reverse(reservations, transformed, metrics);
                    return Err(error);
                },
            };
        transformed.append(&mut staged);
    }
    Ok(transformed)
}

fn apply_bounded_generic_source_adjacent_stages(
    context: &ExecutionContext,
    chunks: Vec<ScheduledLocalParquetReadOutput>,
    stages: &[BoundedGenericSourceAdjacentStage<'_>],
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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
                    context,
                    *node_id,
                    "bounded_generic_source_filter",
                    "bounded_generic_source_filter_output",
                    &chunk,
                    current,
                    reservations,
                    metrics,
                    1,
                    1,
                    |context, input, token, metrics| execute_filter_with_budget(context, node, input, token, metrics),
                ),
                BoundedGenericSourceAdjacentStage::Projection { node_id, node } => {
                    execute_source_adjacent_chunk_projection(
                        context,
                        *node_id,
                        "bounded_generic_source_projection",
                        "bounded_generic_source_projection_output",
                        &chunk,
                        node,
                        current,
                        reservations,
                        metrics,
                        1,
                        1,
                    )
                },
            };
            current = match result {
                Ok(current) => current,
                Err(error) => {
                    release_source_read_outputs_reverse(reservations, pending, metrics);
                    release_source_read_outputs_reverse(reservations, transformed, metrics);
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
    context: &ExecutionContext,
    source: &GpuResidentSource,
    node_id: NodeId,
    gpu_inputs: &GpuResidentInputMap,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_gpu_resident_source(context, source, gpu_inputs, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.borrow_resident(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_gpu_resident_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_gpu_resident_source")
            .with_attempt(1, 1),
        metrics,
    );
    Ok(ResidentGpuFrame { frame, reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_literal_batch_source(
    context: &ExecutionContext,
    source: &crate::source::LiteralBatchSource,
    node_id: NodeId,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let frame = read_literal_batch_source(context, source, metrics)?;
    let requested_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::SourceRead, requested_bytes, reservations.budget())
            .with_memory_boundary("bounded_generic_literal_batch_source")
            .with_node_id(node_id)
            .with_stage("bounded_generic_literal_batch_source")
            .with_attempt(1, 1),
        metrics,
    )?;
    Ok(ResidentGpuFrame { frame, reservation })
}
pub(in crate::exec::bounded) fn execute_bounded_generic_unary(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_with_work_bytes(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        reservations,
        metrics,
        use_budget_token,
        None,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_with_work_bytes(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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
        context,
        node_id,
        stage,
        output_boundary,
        input,
        ReservationKind::OperatorOutput,
        reservations,
        metrics,
        use_budget_token,
        work_request_bytes,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary_with_work_bytes(
            context,
            node_id,
            stage,
            output_boundary,
            input,
            reservations,
            metrics,
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
    execute_bounded_generic_regexp_projection(
        context,
        node_id,
        stage,
        output_boundary,
        projection,
        input,
        None,
        reservations,
        metrics,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_projection_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    if !projection.outputs.iter().any(|output| output.expr.contains_regexp_replace()) {
        return execute_bounded_generic_unary_from_frame_with_reservation_kind(
            context,
            node_id,
            stage,
            output_boundary,
            input,
            input_reserved_bytes,
            ReservationKind::OperatorOutput,
            reservations,
            metrics,
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
    let input_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    )?;
    // The "input" reservation here only scopes prepared-path headroom until transfer.
    let input = ResidentGpuFrame { frame: input, reservation: input_reservation };
    // Drop the ownership of the synthetic wrapper on error paths inside helper.
    execute_bounded_generic_regexp_projection(
        context,
        node_id,
        stage,
        output_boundary,
        projection,
        input,
        Some(input_reserved_bytes),
        reservations,
        metrics,
    )
}

fn execute_bounded_generic_regexp_projection(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    projection: &ProjectionNode,
    input: ResidentGpuFrame,
    _seed_reserved_bytes: Option<u64>,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let input_reserved_bytes = input.reservation.bytes();
    let input_frame = input.frame;
    let input_reservation = input.reservation;

    // Start with a work reservation covering at least the input claim so prepare
    // can allocate retained state under an explicit budget token.
    let initial_work_bytes = input_reserved_bytes.max(1);
    let mut work_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, initial_work_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };

    // Prepare without a stream reservation so retained prepared state keeps the
    // input frame domain. Mint a single execute token after reservation reconcile.
    let prepared = match projection_regexp_replace_prepare(context, projection, &input_frame, metrics) {
        Ok(Some(prepared)) => prepared,
        Ok(None) => {
            // Defense in depth: no regexp after the gate above.
            let token = match execution_budget_token_for_boundary(
                context,
                reservations.available_bytes_including(&work_reservation),
                stage,
                "nexus_query_engine.bounded_generic.budget_token",
                "failed to create bounded native generic operator budget token",
            ) {
                Ok(token) => token,
                Err(error) => {
                    reservations.release(work_reservation, metrics);
                    reservations.release(input_reservation, metrics);
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            };
            match execute_projection_with_budget(context, projection, input_frame, &token, metrics) {
                Ok(frame) => {
                    reservations.release(input_reservation, metrics);
                    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            reservations.release(work_reservation, metrics);
                            return Err(error.with_fact_if_absent("node_id", node_id));
                        },
                    };
                    let output_request = MemoryReservationRequest::new(
                        ReservationKind::OperatorOutput,
                        output_bytes,
                        reservations.budget(),
                    )
                    .with_memory_boundary(output_boundary)
                    .with_node_id(node_id)
                    .with_stage(stage)
                    .with_attempt(1, 1);
                    if let Err(error) = reconcile_work_reservation_to_output(
                        reservations,
                        &mut work_reservation,
                        output_bytes,
                        &output_request,
                        metrics,
                    ) {
                        reservations.release(work_reservation, metrics);
                        return Err(error);
                    }
                    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
                    return Ok(ResidentGpuFrame { frame, reservation: output_reservation });
                },
                Err(error) => {
                    reservations.release(work_reservation, metrics);
                    reservations.release(input_reservation, metrics);
                    return Err(error.with_fact_if_absent("node_id", node_id));
                },
            }
        },
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let required = prepared.required_budget_bytes().max(input_reserved_bytes);
    if required > work_reservation.bytes() {
        let delta = required - work_reservation.bytes();
        if let Err(error) = reservations.try_grow(
            &mut work_reservation,
            MemoryReservationRequest::new(ReservationKind::OperatorOutput, delta, reservations.budget())
                .with_memory_boundary(stage)
                .with_node_id(node_id)
                .with_stage(stage)
                .with_attempt(1, 1),
            metrics,
        ) {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error);
        }
    }

    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&work_reservation),
        stage,
        "nexus_query_engine.bounded_generic.budget_token",
        "failed to create bounded native generic operator budget token after prepare reconcile",
    ) {
        Ok(token) => token,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    let result =
        execute_projection_with_budget_and_prepared(context, projection, input_frame, &token, Some(prepared), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    reservations.release(input_reservation, metrics);

    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request =
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, output_bytes, reservations.budget())
            .with_memory_boundary(output_boundary)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    execute_bounded_generic_unary_from_frame_with_reservation_kind(
        context,
        node_id,
        stage,
        output_boundary,
        input,
        input_reserved_bytes,
        ReservationKind::OperatorOutput,
        reservations,
        metrics,
        use_budget_token,
        execute,
    )
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_consuming_input_with_reservation_kind(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: ResidentGpuFrame,
    reservation_kind: ReservationKind,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
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
    let mut work_reservation = match reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            reservations.release(input_reservation, metrics);
            return Err(error);
        },
    };
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation, metrics);
                reservations.release(input_reservation, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(context, input_frame, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            reservations.release(input_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };

    reservations.release(input_reservation, metrics);

    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
    Ok(ResidentGpuFrame { frame, reservation: output_reservation })
}

pub(in crate::exec::bounded) fn execute_bounded_generic_unary_from_frame_with_reservation_kind(
    context: &ExecutionContext,
    node_id: NodeId,
    stage: &'static str,
    output_boundary: &'static str,
    input: GpuDataFrame,
    input_reserved_bytes: u64,
    reservation_kind: ReservationKind,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
    use_budget_token: bool,
    execute: impl FnOnce(
        &ExecutionContext,
        GpuDataFrame,
        Option<&ExecutionBudgetToken>,
        &mut QueryMetrics,
    ) -> Result<GpuDataFrame>,
) -> Result<ResidentGpuFrame> {
    let mut work_reservation = reservations.try_reserve(
        MemoryReservationRequest::new(reservation_kind, input_reserved_bytes, reservations.budget())
            .with_memory_boundary(stage)
            .with_node_id(node_id)
            .with_stage(stage)
            .with_attempt(1, 1),
        metrics,
    )?;
    let token = if use_budget_token {
        match execution_budget_token_for_boundary(
            context,
            reservations.available_bytes_including(&work_reservation),
            stage,
            "nexus_query_engine.bounded_generic.budget_token",
            "failed to create bounded native generic operator budget token",
        ) {
            Ok(token) => Some(token),
            Err(error) => {
                reservations.release(work_reservation, metrics);
                return Err(error.with_fact_if_absent("node_id", node_id));
            },
        }
    } else {
        None
    };
    let result = execute(context, input, token.as_ref(), metrics);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(work_reservation, metrics);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let output_request = MemoryReservationRequest::new(reservation_kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage)
        .with_attempt(1, 1);
    if let Err(error) = reconcile_work_reservation_to_output(
        reservations,
        &mut work_reservation,
        output_bytes,
        &output_request,
        metrics,
    ) {
        reservations.release(work_reservation, metrics);
        return Err(error);
    }
    let output_reservation = reservations.transfer(work_reservation, output_request, metrics);
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
