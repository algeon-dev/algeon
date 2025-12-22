use super::{
    source::{read_local_parquet_chunk, record_bounded_parquet_source_metrics},
    *,
};

pub(super) fn execute_bounded_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    if plan.contains_join() {
        if plan.has_bounded_executable_root_join() {
            return execute_bounded_root_join_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_join_pipeline() {
            return execute_bounded_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        reject_bounded_join_plan(context, plan)?;
    }

    if !plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Aggregate(_))) {
        return execute_bounded_generic_query_plan(context, plan, gpu_inputs, metrics);
    }

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
    let frame = execute_post_aggregate_projection(context, &bounded, frame, &mut reservations, metrics)?;
    let frame = execute_final_sort_limit_with_retry(context, &bounded, frame, &mut reservations, metrics)?;

    record_device_memory_observation(&frame.frame, metrics)?;
    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, frame.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(frame.reservation);
                    return Err(error);
                },
            };
            reservations.release(frame.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output =
                materialize_bounded_to_host(context, frame, plan.sink().input(), schema, &mut reservations, metrics)?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_generic_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    if plan.contains_join() {
        if plan.has_bounded_executable_root_join() {
            return execute_bounded_root_join_query_plan(context, plan, gpu_inputs, metrics);
        }
        if plan.has_bounded_executable_join_pipeline() {
            return execute_bounded_join_pipeline_query_plan(context, plan, gpu_inputs, metrics);
        }
        reject_bounded_join_plan(context, plan)?;
    }

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let frame = execute_node(context, plan, plan.sink().input(), gpu_inputs, &mut scratch, metrics)?;
    let frame_bytes = observe_device_memory_bytes(&frame, metrics)?;
    let reservation = reservations.try_reserve(
        MemoryReservationRequest::new(ReservationKind::OperatorOutput, frame_bytes, reservations.budget())
            .with_memory_boundary("operator_output")
            .with_node_id(plan.sink().input())
            .with_stage("bounded_generic")
            .with_attempt(1, 1),
        metrics,
    )?;
    let committed = CommittedPartial { frame, reservation };

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn reject_bounded_join_plan(context: &ExecutionContext, plan: &QueryPlan) -> Result<()> {
    if plan.nodes().iter().any(|node| matches!(node.kind, PlanNode::Join(_))) {
        let contract = BoundedJoinReservationContract;
        let mut error = unsupported_bounded_shape(
            "bounded native join execution currently supports only root reusable hash joins with ordered reservations",
            "bounded_join_plan_shape_unsupported",
        );
        for (key, value) in contract.rejection_facts(context.boundary_policy()) {
            error = error.with_fact(key, value);
        }
        return Err(error);
    }

    Ok(())
}

fn execute_bounded_root_join_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_root_join_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let join_node_id = plan.sink().input();
    let node = plan
        .node(join_node_id)
        .ok_or_else(|| invalid_node_reference(join_node_id, "bounded root join plan references a missing join node"))?;
    let PlanNode::Join(join) = &node.kind else {
        return Err(unsupported_bounded_shape(
            "bounded root join execution requires the sink input to be a native join",
            "missing_root_join",
        ));
    };

    let left = execute_node(context, plan, join.inputs[0], gpu_inputs, &mut scratch, metrics)?;
    let right = execute_node(context, plan, join.inputs[1], gpu_inputs, &mut scratch, metrics)?;
    let committed = execute_bounded_root_hash_join(context, node.id, join, left, right, &mut reservations, metrics)?;

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

fn execute_bounded_join_pipeline_query_plan(
    context: &ExecutionContext,
    plan: &QueryPlan,
    gpu_inputs: &GpuResidentInputMap,
    metrics: &mut QueryMetrics,
) -> Result<QueryOutput> {
    let pipeline = BoundedJoinPipelinePlan::analyze(plan)?;
    pipeline.validate_capabilities()?;

    let MemoryControlPolicy::Bounded { device_budget_bytes, .. } = context.memory_control_policy() else {
        unreachable!("execute_bounded_join_pipeline_query_plan is called only for bounded policies");
    };
    let mut reservations = ReservationTracker::new(device_budget_bytes);
    let mut scratch = ExecutionScratch::new(context);
    let left = execute_node(context, plan, pipeline.join.inputs[0], gpu_inputs, &mut scratch, metrics)?;
    let right = execute_node(context, plan, pipeline.join.inputs[1], gpu_inputs, &mut scratch, metrics)?;
    let mut committed = execute_bounded_root_hash_join(
        context,
        pipeline.join_id,
        pipeline.join,
        left,
        right,
        &mut reservations,
        metrics,
    )?;

    for stage in &pipeline.stages {
        committed = execute_bounded_join_pipeline_stage(context, stage, committed, &mut reservations, metrics)?;
    }

    match plan.sink() {
        ResultSink::GpuResident { .. } => {
            let handle = match GpuResultHandle::new(context, committed.frame, Arc::clone(plan.output_schema())) {
                Ok(handle) => handle,
                Err(error) => {
                    reservations.release(committed.reservation);
                    return Err(error);
                },
            };
            reservations.release(committed.reservation);
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::GpuResident(handle))
        },
        ResultSink::HostArrow { schema, .. } => {
            let output = materialize_bounded_to_host(
                context,
                committed,
                plan.sink().input(),
                schema,
                &mut reservations,
                metrics,
            )?;
            debug_assert_eq!(reservations.reserved_bytes(), 0);
            Ok(QueryOutput::HostArrow(output))
        },
    }
}

struct BoundedJoinPipelinePlan<'a> {
    join_id: NodeId,
    join: &'a JoinNode,
    stages: Vec<BoundedJoinPipelineStage<'a>>,
}

enum BoundedJoinPipelineStage<'a> {
    Filter { node_id: NodeId, node: &'a FilterNode },
    Projection { node_id: NodeId, node: &'a ProjectionNode, output_schema: &'a arrow::datatypes::SchemaRef },
    Aggregate { node_id: NodeId, node: &'a AggregateNode, output_schema: &'a arrow::datatypes::SchemaRef },
    Sort { node_id: NodeId, node: &'a SortNode },
    Limit { node_id: NodeId, node: &'a LimitNode },
}

impl<'a> BoundedJoinPipelinePlan<'a> {
    fn analyze(plan: &'a QueryPlan) -> Result<Self> {
        let mut current = plan.sink().input();
        let mut stages = Vec::new();
        let mut saw_aggregate = false;

        loop {
            let node = plan.node(current).ok_or_else(|| {
                invalid_node_reference(current, "bounded join pipeline references a missing plan node")
            })?;
            match &node.kind {
                PlanNode::Join(join) => {
                    if !join.is_bounded_root_hash_join_compatible() {
                        return Err(unsupported_bounded_shape(
                            "bounded join pipeline execution currently supports reusable hash inner, left, and full equi joins",
                            "bounded_join_plan_shape_unsupported",
                        )
                        .with_fact("join_type", join.spec.join_type)
                        .with_fact("build_side", join.spec.build_side)
                        .with_fact("key_count", join.spec.keys.len())
                        .with_fact("residual_predicate", join.spec.residual_predicate.is_some())
                        .with_fact("output_projection_len", join.spec.output_projection.len()));
                    }
                    stages.reverse();
                    return Ok(Self { join_id: current, join, stages });
                },
                PlanNode::Filter(filter) => {
                    stages.push(BoundedJoinPipelineStage::Filter { node_id: current, node: filter });
                    current = filter.input;
                },
                PlanNode::Projection(projection) => {
                    stages.push(BoundedJoinPipelineStage::Projection {
                        node_id: current,
                        node: projection,
                        output_schema: &node.facts.output_schema,
                    });
                    current = projection.input;
                },
                PlanNode::Aggregate(aggregate) if !saw_aggregate => {
                    saw_aggregate = true;
                    stages.push(BoundedJoinPipelineStage::Aggregate {
                        node_id: current,
                        node: aggregate,
                        output_schema: &node.facts.output_schema,
                    });
                    current = aggregate.input;
                },
                PlanNode::Sort(sort) => {
                    stages.push(BoundedJoinPipelineStage::Sort { node_id: current, node: sort });
                    current = sort.input;
                },
                PlanNode::Limit(limit) => {
                    let child = plan.node(limit.input).ok_or_else(|| {
                        invalid_node_reference(limit.input, "bounded join pipeline limit references a missing input")
                    })?;
                    if !matches!(child.kind, PlanNode::Sort(_)) {
                        return Err(unsupported_bounded_shape(
                            "bounded join pipeline execution does not support limit without a preceding global sort",
                            "limit_without_sort",
                        ));
                    }
                    stages.push(BoundedJoinPipelineStage::Limit { node_id: current, node: limit });
                    current = limit.input;
                },
                PlanNode::Source(_) | PlanNode::Aggregate(_) => {
                    return Err(unsupported_bounded_shape(
                        "bounded join pipeline execution requires a reusable hash join below supported unary operators",
                        "missing_root_join",
                    ));
                },
            }
        }
    }

    fn validate_capabilities(&self) -> Result<()> {
        for stage in &self.stages {
            match stage {
                BoundedJoinPipelineStage::Filter { node, .. } => {
                    lower_expr(&node.predicate)?;
                },
                BoundedJoinPipelineStage::Projection { node, output_schema, .. } => {
                    validate_projection_capability(node, output_schema)?;
                },
                BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => {
                    validate_aggregate_capability(node, output_schema)?;
                },
                BoundedJoinPipelineStage::Sort { node, .. } => {
                    lower_sort_spec(&node.spec)?;
                },
                BoundedJoinPipelineStage::Limit { node, .. } => {
                    topk_candidate_rows(node.spec)?;
                },
            }
        }
        Ok(())
    }
}

fn execute_bounded_join_pipeline_stage(
    context: &ExecutionContext,
    stage: &BoundedJoinPipelineStage<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let node_id = stage.node_id();
    let stage_name = stage.memory_boundary();
    let kind = stage.reservation_kind();
    let work_request = MemoryReservationRequest::new(kind, input.reservation.bytes(), reservations.budget())
        .with_memory_boundary(stage_name)
        .with_node_id(node_id)
        .with_stage(stage_name);
    let work_reservation = reservations.try_reserve(work_request, metrics)?;
    let result = match stage {
        BoundedJoinPipelineStage::Filter { node, .. } => execute_filter(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Projection { node, output_schema, .. } => {
            execute_projection(context, node, input.frame, output_schema, metrics)
        },
        BoundedJoinPipelineStage::Aggregate { node, output_schema, .. } => {
            execute_aggregate(context, node, input.frame, output_schema, metrics)
        },
        BoundedJoinPipelineStage::Sort { node, .. } => execute_sort(context, node, input.frame, metrics),
        BoundedJoinPipelineStage::Limit { node, .. } => execute_limit(context, node, input.frame, metrics),
    };
    reservations.release(work_reservation);

    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: stage_name,
                stage: stage_name,
                node_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: stage_name,
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation);
    let output_boundary = stage.output_memory_boundary();
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: output_boundary,
            stage: stage_name,
            node_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: stage_name,
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request = MemoryReservationRequest::new(kind, output_bytes, reservations.budget())
        .with_memory_boundary(output_boundary)
        .with_node_id(node_id)
        .with_stage(stage_name);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

impl BoundedJoinPipelineStage<'_> {
    fn node_id(&self) -> NodeId {
        match self {
            Self::Filter { node_id, .. }
            | Self::Projection { node_id, .. }
            | Self::Aggregate { node_id, .. }
            | Self::Sort { node_id, .. }
            | Self::Limit { node_id, .. } => *node_id,
        }
    }

    fn reservation_kind(&self) -> ReservationKind {
        match self {
            Self::Sort { .. } | Self::Limit { .. } => ReservationKind::FinalSort,
            Self::Filter { .. } | Self::Projection { .. } | Self::Aggregate { .. } => ReservationKind::OperatorOutput,
        }
    }

    fn memory_boundary(&self) -> &'static str {
        match self {
            Self::Filter { .. } => "join_pipeline_filter",
            Self::Projection { .. } => "join_pipeline_projection",
            Self::Aggregate { .. } => "join_pipeline_aggregate",
            Self::Sort { .. } => "join_pipeline_sort",
            Self::Limit { .. } => "join_pipeline_limit",
        }
    }

    fn output_memory_boundary(&self) -> &'static str {
        match self {
            Self::Filter { .. } => "join_pipeline_filter_output",
            Self::Projection { .. } => "join_pipeline_projection_output",
            Self::Aggregate { .. } => "join_pipeline_aggregate_output",
            Self::Sort { .. } => "join_pipeline_sort_output",
            Self::Limit { .. } => "join_pipeline_limit_output",
        }
    }
}

fn execute_bounded_root_hash_join(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: GpuDataFrame,
    right: GpuDataFrame,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    validate_executable_join_subset(node)?;
    if !node.is_bounded_root_hash_join_compatible() {
        return Err(unsupported_bounded_shape(
            "bounded root join execution currently supports reusable hash inner, left, and full equi joins",
            "bounded_join_plan_shape_unsupported",
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("build_side", node.spec.build_side)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("residual_predicate", node.spec.residual_predicate.is_some())
        .with_fact("output_projection_len", node.spec.output_projection.len()));
    }

    let left = left.into_frame();
    let right = right.into_frame();
    let original_left_rows = left.num_rows() as u64;
    let original_right_rows = right.num_rows() as u64;
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    let left = select_join_input(left, &left_columns, "left")?;
    let right = select_join_input(right, &right_columns, "right")?;
    let left_bytes = observe_device_memory_bytes(&GpuDataFrame::new(left.clone()), metrics)?;
    let right_bytes = observe_device_memory_bytes(&GpuDataFrame::new(right.clone()), metrics)?;

    let (build, build_keys, probe, probe_keys, build_side, left_indices_from_probe) = match node.spec.build_side {
        JoinSide::Left => (&left, &left_keys, &right, &right_keys, JoinSide::Left, false),
        JoinSide::Right => (&right, &right_keys, &left, &left_keys, JoinSide::Right, true),
    };
    let probe_side = probe_side(build_side);
    let build_bytes = side_metric(build_side, left_bytes, right_bytes);
    let probe_bytes = side_metric(probe_side, left_bytes, right_bytes);

    let build_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::BuildSide,
            "join_build_side",
            build_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let overrides = execution_overrides(context);
    let hash_options =
        cudf::DataFrameHashJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let hash_table_reservation_bytes = build
        .with_execution(&overrides)
        .hash_join_pre_build_reservation_size_bytes(build_keys, hash_options.clone())
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.hash_table_reservation",
                "failed to size bounded native reusable hash join build",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", build_side)
        })?;
    let hash_table_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::HashTable,
            "join_hash_table",
            hash_table_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let timer = Instant::now();
    let hash_join = build.with_execution(&overrides).build_hash_join(build_keys, hash_options).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.build_hash_join",
            "failed to build bounded native reusable hash join",
            error,
        )
        .with_fact("join_type", node.spec.join_type)
        .with_fact("key_count", node.spec.keys.len())
        .with_fact("build_side", build_side)
    })?;
    let hash_join = GpuHashJoinBuildHandle::new(context, hash_join, node.spec.null_equality).map_err(|error| {
        error
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", build_side)
    })?;
    let hash_table_bytes = hash_join.device_allocated_size_bytes();
    metrics.record_hash_join_build();
    metrics.record_hash_join_hash_table_bytes(hash_table_bytes);

    let probe_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::ProbeSide,
            "join_probe_side",
            probe_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output_size = match node.spec.join_type {
        JoinType::Inner => hash_join.hash_join().inner_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Left => hash_join.hash_join().left_probe_size_in(probe, probe_keys, &overrides),
        JoinType::Full => hash_join.hash_join().full_probe_size_in(probe, probe_keys, &overrides),
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_reservation_bytes =
        hash_join.hash_join().probe_gather_map_reservation_size_bytes(output_size).map(|bytes| bytes as u64).map_err(
            |error| {
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.gather_map_reservation",
                    "failed to size bounded native reusable hash join gather maps",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("build_side", build_side)
            },
        )?;
    let gather_map_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::GatherMap,
            "join_gather_map",
            gather_map_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let indices = match node.spec.join_type {
        JoinType::Inner => {
            hash_join.hash_join().inner_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Left => {
            hash_join.hash_join().left_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::Full => {
            hash_join.hash_join().full_probe_with_output_size_in(probe, probe_keys, output_size, &overrides)
        },
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.bounded.join.gather_map_bytes",
            "failed to observe bounded native join gather-map memory",
            error,
        )
    })?;
    let (left_indices, right_indices) = if left_indices_from_probe {
        (indices.probe().clone(), indices.build().clone())
    } else {
        (indices.build().clone(), indices.probe().clone())
    };
    let left_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Left);
    let right_projection = projected_output_columns(&node.spec.output_projection, JoinSide::Right);
    let left_output_bytes =
        join_side_output_reservation_bytes(&left, &left_indices, &left_projection, &overrides, JoinSide::Left)?;
    let right_output_bytes =
        join_side_output_reservation_bytes(&right, &right_indices, &right_projection, &overrides, JoinSide::Right)?;
    let output_reservation_bytes = left_output_bytes.saturating_add(right_output_bytes);
    let output_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_output",
            output_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let output = materialize_join_output_projection(
        &left,
        &right,
        &left_indices,
        &right_indices,
        &overrides,
        &node.spec.output_projection,
    )?;
    metrics.operator_time += timer.elapsed();

    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, original_left_rows, original_right_rows),
            build_bytes,
            probe_rows: side_metric(probe_side, original_left_rows, original_right_rows),
            probe_bytes,
            hash_table_bytes,
            gather_map_bytes,
            output_rows,
            output_bytes,
        },
    ));

    // `left_indices`/`right_indices` are `Series::clone`s of `indices.probe()` /
    // `indices.build()`; the underlying columns are Arc-shared, so device memory
    // is only released once every Arc holder drops. All three must be released
    // before `reservations.release(gather_map_reservation)` for the budget to
    // reflect actual residency.
    drop(left_indices);
    drop(right_indices);
    drop(indices);
    reservations.release(gather_map_reservation);
    drop(hash_join);
    reservations.release(hash_table_reservation);
    drop(left);
    drop(right);
    reservations.release(probe_reservation);
    reservations.release(build_reservation);

    Ok(CommittedPartial { frame: GpuDataFrame::new(output), reservation: output_reservation })
}

fn bounded_join_request(
    kind: ReservationKind,
    memory_boundary: &'static str,
    requested_bytes: u64,
    budget_bytes: u64,
    node_id: NodeId,
) -> MemoryReservationRequest {
    MemoryReservationRequest::new(kind, requested_bytes, budget_bytes)
        .with_memory_boundary(memory_boundary)
        .with_node_id(node_id)
        .with_stage("bounded_join")
        .with_attempt(1, 1)
}

fn join_side_output_reservation_bytes(
    frame: &DataFrame,
    indices: &cudf::DataFrameJoinIndices,
    columns: &[String],
    overrides: &ExecutionOverrides<'_>,
    side: JoinSide,
) -> Result<u64> {
    if columns.is_empty() {
        return Ok(0);
    }
    frame
        .with_execution(overrides)
        .materialize_join_indices_reservation_size_bytes(indices, columns)
        .map(|bytes| bytes as u64)
        .map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.output_reservation",
                "failed to size bounded native join output materialization",
                error,
            )
            .with_fact("side", side)
        })
}

pub(super) fn execute_bounded_chunk_with_retry(
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
pub(super) fn execute_bounded_chunk_retry_attempt(
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

pub(super) fn execute_bounded_chunk_attempt(
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
    let mut frame = match read_local_parquet_chunk(context, bounded.source, chunk, metrics) {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
        },
    };
    let source_observed_bytes = match observe_device_memory_bytes(&frame, metrics) {
        Ok(bytes) => bytes,
        Err(error) => {
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "source_chunk_read", attempt, max_attempts, metrics));
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
        for (projection, output_schema) in &bounded.pre_filter_projections {
            frame = execute_projection(context, projection, frame, output_schema, metrics)?;
        }
        if let Some(filter) = bounded.filter {
            frame = execute_filter(context, filter, frame, metrics)?;
        }
        for (projection, output_schema) in &bounded.post_filter_projections {
            frame = execute_projection(context, projection, frame, output_schema, metrics)?;
        }
        execute_partial_aggregate(
            context,
            bounded.aggregate.0,
            bounded.aggregate_input_schema,
            bounded.aggregate.1,
            frame,
            metrics,
        )
    })();
    metrics.chunk_operator_time += chunk_operator_timer.elapsed();
    let frame = match chunk_output {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(operator_reservation);
            reservations.release(source_reservation);
            return Err(attach_chunk_resource_facts(error, chunk, "chunk_operator", attempt, max_attempts, metrics));
        },
    };

    reservations.release(operator_reservation);
    reservations.release(source_reservation);

    let partial_bytes = observe_device_memory_bytes(&frame, metrics)
        .map_err(|error| attach_chunk_resource_facts(error, chunk, "partial_state", attempt, max_attempts, metrics))?;
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

type ProjectionStage<'a> = (&'a ProjectionNode, &'a arrow::datatypes::SchemaRef);

#[derive(Clone)]
pub(super) struct BoundedPlan<'a> {
    pub(super) source_id: NodeId,
    pub(super) source: &'a LocalParquetSource,
    pub(super) pre_filter_projections: Vec<ProjectionStage<'a>>,
    pub(super) filter: Option<&'a FilterNode>,
    pub(super) post_filter_projections: Vec<ProjectionStage<'a>>,
    pub(super) aggregate_id: NodeId,
    pub(super) aggregate_input_schema: &'a arrow::datatypes::SchemaRef,
    pub(super) aggregate: (&'a AggregateNode, &'a arrow::datatypes::SchemaRef),
    pub(super) post_aggregate_projection: Option<ProjectionStage<'a>>,
    pub(super) sort: Option<&'a SortNode>,
    pub(super) limit: Option<&'a LimitNode>,
}

fn take_projection_stack<'a>(
    plan: &'a QueryPlan,
    current: &mut NodeId,
    missing_message: &'static str,
) -> Result<Vec<ProjectionStage<'a>>> {
    let mut projections = Vec::new();
    loop {
        let Some(node) = plan.node(*current) else {
            return Err(invalid_node_reference(*current, missing_message));
        };
        let PlanNode::Projection(projection) = &node.kind else {
            break;
        };
        projections.push((projection, &node.facts.output_schema));
        *current = projection.input;
    }
    projections.reverse();
    Ok(projections)
}

impl<'a> BoundedPlan<'a> {
    pub(super) fn analyze(plan: &'a QueryPlan) -> Result<Self> {
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

        let post_aggregate_projection = match plan.node(current) {
            Some(node) => match &node.kind {
                PlanNode::Projection(projection) => {
                    current = projection.input;
                    Some((projection, &node.facts.output_schema))
                },
                _ => None,
            },
            None => {
                return Err(invalid_node_reference(current, "bounded query plan references a missing sort input node"));
            },
        };

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
        let aggregate_input_node = plan.node(aggregate.input).ok_or_else(|| {
            invalid_node_reference(aggregate.input, "bounded query plan references a missing aggregate input node")
        })?;
        current = aggregate.input;

        let post_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing aggregate input node")?;

        let filter = match plan.node(current).map(|node| &node.kind) {
            Some(PlanNode::Filter(node)) => {
                current = node.input;
                Some(node)
            },
            _ => None,
        };

        let pre_filter_projections =
            take_projection_stack(plan, &mut current, "bounded query plan references a missing source input node")?;

        let source_node = plan
            .node(current)
            .ok_or_else(|| invalid_node_reference(current, "bounded query plan references a missing source node"))?;
        let PlanNode::Source(source_node) = &source_node.kind else {
            return Err(unsupported_bounded_shape(
                "bounded execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };
        let crate::source::SourceNode::LocalParquet(source) = source_node.as_ref() else {
            return Err(unsupported_bounded_shape(
                "bounded aggregate execution currently supports only local parquet sources",
                "non_local_parquet_source",
            ));
        };

        Ok(Self {
            source_id: current,
            source,
            pre_filter_projections,
            filter,
            post_filter_projections,
            aggregate_id,
            aggregate_input_schema: &aggregate_input_node.facts.output_schema,
            aggregate: (aggregate, &aggregate_node.facts.output_schema),
            post_aggregate_projection,
            sort,
            limit,
        })
    }

    pub(super) fn validate_capabilities(&self) -> Result<()> {
        if let Some(filter) = self.filter {
            lower_expr(&filter.predicate)?;
        }

        for (projection, output_schema) in &self.pre_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        for (projection, output_schema) in &self.post_filter_projections {
            validate_projection_capability(projection, output_schema)?;
        }

        validate_aggregate_capability(self.aggregate.0, self.aggregate.1)?;
        validate_final_aggregate_capability(self.aggregate.0, self.aggregate.1)?;

        if let Some((projection, output_schema)) = self.post_aggregate_projection {
            validate_projection_capability(projection, output_schema)?;
        }

        if let Some(sort) = self.sort {
            lower_sort_spec(&sort.spec)?;
        }
        if let Some(limit) = self.limit {
            topk_candidate_rows(limit.spec)?;
        }

        Ok(())
    }
}

pub(super) fn execute_final_aggregate_merge(
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

    let partial = partials.pop().expect("final merge loop leaves exactly one partial");
    execute_final_aggregate_projection(context, bounded, partial, reservations, metrics)
}

pub(super) fn execute_post_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let Some((projection, output_schema)) = bounded.post_aggregate_projection else {
        return Ok(input);
    };

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("post_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let result = execute_projection(context, projection, input.frame, output_schema, metrics);
    reservations.release(work_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "post_aggregate_projection",
                stage: "post_aggregate_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "post_aggregate_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "post_aggregate_projection_output",
            stage: "post_aggregate_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "post_aggregate_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("post_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("post_aggregate_projection");
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn execute_final_aggregate_projection(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: CommittedPartial,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    if !aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
        return Ok(input);
    }

    let request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, input.reservation.bytes(), reservations.budget())
            .with_memory_boundary("final_aggregate_projection")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection");
    let work_reservation = reservations.try_reserve(request, metrics)?;
    let expected_output_names =
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    let result = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        finalize_avg_output_columns(context, input.frame.into_frame(), &bounded.aggregate.0.spec, bounded.aggregate.1)
    } else {
        Ok(input.frame.into_frame())
    }
    .and_then(|frame| {
        frame.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.final_projection.select",
                "failed to select bounded native final aggregate output columns",
                error,
            )
        })
    })
    .map(GpuDataFrame::new);
    reservations.release(work_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            reservations.release(input.reservation);
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_projection",
                stage: "final_projection",
                node_id: bounded.aggregate_id,
                attempt: 1,
                max_attempts: 1,
                resume_boundary: "final_projection",
                metrics,
            }
            .attach_to(error));
        },
    };

    reservations.release(input.reservation);
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_projection_output",
            stage: "final_projection",
            node_id: bounded.aggregate_id,
            attempt: 1,
            max_attempts: 1,
            resume_boundary: "final_projection",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_projection_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_projection");
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_final_merge_level_attempt(
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

pub(super) fn execute_final_merge_group_attempt(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    debug_assert!(partials.len() > 1);
    let requested_bytes = partials.iter().map(|partial| partial.reservation.bytes()).sum::<u64>();
    let merge_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, requested_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let merge_reservation = reservations.try_reserve(merge_request, metrics)?;

    let timer = Instant::now();
    let result = merge_final_aggregate_partials(context, bounded, partials, metrics);
    let merge_elapsed = timer.elapsed();
    reservations.release(merge_reservation);
    let frame = match result {
        Ok(frame) => frame,
        Err(error) => {
            metrics.final_merge_time += merge_elapsed;
            return Err(RuntimeResourceFacts {
                memory_boundary: "final_aggregate_merge",
                stage: "final_merge",
                node_id: bounded.aggregate_id,
                attempt,
                max_attempts,
                resume_boundary: "final_merge",
                metrics,
            }
            .attach_to(error));
        },
    };
    metrics.final_merge_time += merge_elapsed;

    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_aggregate_merge_output",
            stage: "final_merge",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_merge",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request =
        MemoryReservationRequest::new(ReservationKind::FinalMerge, output_bytes, reservations.budget())
            .with_memory_boundary("final_aggregate_merge_output")
            .with_node_id(bounded.aggregate_id)
            .with_stage("final_merge")
            .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) fn merge_final_aggregate_partials(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    partials: &[CommittedPartial],
    metrics: &mut QueryMetrics,
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

    execute_final_aggregate(context, bounded, GpuDataFrame::new(concatenated), metrics)
}

pub(super) enum MergeLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_final_merge_level(
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

pub(super) fn release_merge_level_items(items: Vec<MergeLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let MergeLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn is_retryable_final_merge_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_final_merge_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn reduce_merge_fan_in(merge_fan_in: usize) -> Option<usize> {
    if merge_fan_in <= 2 {
        return None;
    }

    Some(merge_fan_in.div_ceil(2).max(2))
}

pub(super) fn final_merge_retry_facts(merge_level: usize, merge_fan_in: usize, metrics: &QueryMetrics) -> RetryFacts {
    let retry_action = if merge_fan_in <= 2 { "fail_min_merge_fan_in" } else { "retry_attempts_exhausted" };

    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_merge")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
        .with_merge_level(merge_level)
        .with_merge_fan_in(merge_fan_in)
}

pub(super) fn execute_final_sort_limit_with_retry(
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
    let mut final_sort = FinalSortExecutor { context, bounded, sort, max_attempts, reservations, metrics };
    let result = final_sort.full_sort_limit_attempt(bounded.limit, &input, 1);

    match result {
        Ok(output) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation);
            Ok(output)
        },
        Err(error) if is_retryable_final_sort_error(&error) => {
            record_final_sort_retry_error(&error, final_sort.metrics);
            if max_attempts > 1
                && let Some(limit) = bounded.limit
            {
                match final_sort.hierarchical_topk(limit, input) {
                    Ok(output) => {
                        final_sort.metrics.final_sort_time += timer.elapsed();
                        Ok(output)
                    },
                    Err(error) => {
                        final_sort.metrics.final_sort_time += timer.elapsed();
                        Err(error)
                    },
                }
            } else {
                final_sort.metrics.final_sort_time += timer.elapsed();
                final_sort.reservations.release(input.reservation);
                Err(final_sort_retry_facts("retry_attempts_exhausted", final_sort.metrics).attach_to_error(error))
            }
        },
        Err(error) => {
            final_sort.metrics.final_sort_time += timer.elapsed();
            final_sort.reservations.release(input.reservation);
            Err(error)
        },
    }
}

struct FinalSortExecutor<'a, 'plan> {
    context: &'a ExecutionContext,
    bounded: &'a BoundedPlan<'plan>,
    sort: &'a SortNode,
    max_attempts: usize,
    reservations: &'a mut ReservationTracker,
    metrics: &'a mut QueryMetrics,
}

#[derive(Clone, Copy)]
struct TopKCandidatePlan {
    keep_rows: usize,
    bytes_per_row: u64,
    rows_per_chunk: usize,
}

#[derive(Clone, Copy)]
struct TopKSliceAttempt {
    offset: usize,
    length: usize,
    keep_rows: usize,
    bytes_per_row: u64,
    attempt: usize,
}

#[derive(Clone, Copy)]
struct TopKMergeAttempt {
    keep_rows: usize,
    merge_fan_in: usize,
    attempt: usize,
}

impl<'a, 'plan> FinalSortExecutor<'a, 'plan> {
    fn full_sort_limit_attempt(
        &mut self,
        limit: Option<&LimitNode>,
        input: &CommittedPartial,
        attempt: usize,
    ) -> Result<CommittedPartial> {
        let sort_request = MemoryReservationRequest::new(
            ReservationKind::FinalSort,
            input.reservation.bytes(),
            self.reservations.budget(),
        )
        .with_memory_boundary("final_sort")
        .with_node_id(self.bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, self.max_attempts);
        let sort_reservation = self.reservations.try_reserve(sort_request, self.metrics)?;

        let result = execute_sort_on_frame(self.context, self.sort, input.frame.frame().clone(), self.metrics)
            .and_then(|frame| execute_optional_limit(self.context, limit, frame, self.metrics));
        self.reservations.release(sort_reservation);
        let frame = result.map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        reserve_final_sort_output(self.bounded, frame, attempt, self.max_attempts, self.reservations, self.metrics)
    }

    fn hierarchical_topk(&mut self, limit: &LimitNode, input: CommittedPartial) -> Result<CommittedPartial> {
        let keep_rows = topk_candidate_rows(limit.spec)?;
        if keep_rows == 0 {
            let output = match execute_limit_on_frame(self.context, limit, input.frame.frame().clone(), self.metrics) {
                Ok(output) => output,
                Err(error) => {
                    self.reservations.release(input.reservation);
                    return Err(error);
                },
            };
            self.reservations.release(input.reservation);
            return reserve_final_sort_output(
                self.bounded,
                output,
                2,
                self.max_attempts,
                self.reservations,
                self.metrics,
            );
        }

        let input_rows = input.frame.num_rows();
        if input_rows == 0 {
            return Ok(input);
        }
        let bytes_per_row = estimated_bytes_per_row(input.reservation.bytes(), input_rows);
        let rows_per_chunk = final_sort_rows_per_chunk(
            input_rows,
            bytes_per_row,
            self.reservations.budget().saturating_sub(input.reservation.bytes()),
        );
        let mut candidates = match self
            .initial_topk_candidates(&input, TopKCandidatePlan { keep_rows, bytes_per_row, rows_per_chunk })
        {
            Ok(candidates) => candidates,
            Err(error) => {
                record_final_sort_retry_error(&error, self.metrics);
                self.reservations.release(input.reservation);
                return Err(
                    final_sort_retry_facts("hierarchical_topk_initial_failed", self.metrics).attach_to_error(error)
                );
            },
        };

        self.reservations.release(input.reservation);

        let mut merge_level = 1;
        while candidates.len() > 1 {
            let mut merge_fan_in = candidates.len();
            let mut attempt = 2;

            loop {
                match self.topk_merge_level_attempt(&candidates, TopKMergeAttempt { keep_rows, merge_fan_in, attempt })
                {
                    Ok(level_output) => {
                        candidates = commit_topk_merge_level(candidates, level_output, self.reservations);
                        merge_level += 1;
                        break;
                    },
                    Err(error) if is_retryable_final_sort_error(&error) => {
                        record_final_sort_retry_error(&error, self.metrics);
                        if attempt < self.max_attempts
                            && let Some(reduced_fan_in) = reduce_merge_fan_in(merge_fan_in)
                        {
                            merge_fan_in = reduced_fan_in;
                            attempt += 1;
                            continue;
                        }

                        release_committed_partials(candidates, self.reservations);
                        return Err(
                            final_sort_topk_retry_facts(merge_level, merge_fan_in, self.metrics).attach_to_error(error)
                        );
                    },
                    Err(error) => {
                        release_committed_partials(candidates, self.reservations);
                        return Err(error);
                    },
                }
            }
        }

        let candidate = candidates.pop().expect("hierarchical top-k leaves exactly one candidate");
        if limit.spec.skip == 0 && candidate.frame.num_rows() <= limit.spec.fetch {
            return Ok(candidate);
        }

        let output = execute_limit(self.context, limit, candidate.frame, self.metrics).map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_limit",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: self.max_attempts,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        self.reservations.release(candidate.reservation);
        reserve_final_sort_output(
            self.bounded,
            output,
            self.max_attempts,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }

    fn initial_topk_candidates(
        &mut self,
        input: &CommittedPartial,
        plan: TopKCandidatePlan,
    ) -> Result<Vec<CommittedPartial>> {
        let input_rows = input.frame.num_rows();
        let mut candidates = Vec::with_capacity(input_rows.div_ceil(plan.rows_per_chunk));
        let mut offset = 0;
        while offset < input_rows {
            let length = plan.rows_per_chunk.min(input_rows - offset);
            match self.topk_slice_attempt(
                input,
                TopKSliceAttempt {
                    offset,
                    length,
                    keep_rows: plan.keep_rows,
                    bytes_per_row: plan.bytes_per_row,
                    attempt: 2,
                },
            ) {
                Ok(candidate) => candidates.push(candidate),
                Err(error) => {
                    release_committed_partials(candidates, self.reservations);
                    return Err(error);
                },
            }
            offset += length;
        }
        Ok(candidates)
    }

    fn topk_slice_attempt(&mut self, input: &CommittedPartial, slice: TopKSliceAttempt) -> Result<CommittedPartial> {
        let requested_bytes = estimate_rows_bytes(slice.bytes_per_row, slice.length);
        let request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort_topk_chunk")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(slice.attempt, self.max_attempts);
        let work_reservation = self.reservations.try_reserve(request, self.metrics)?;

        let result = input
            .frame
            .frame()
            .clone()
            .with_execution(&execution_overrides(self.context))
            .slice(slice.offset, slice.length)
            .map(GpuDataFrame::new)
            .map_err(|error| {
                QueryError::from_cudf(
                    "cudf_query_engine.final_sort.topk_slice",
                    "failed to slice bounded native final sort input",
                    error,
                )
            })
            .and_then(|frame| execute_topk_candidate(self.context, self.sort, frame, slice.keep_rows, self.metrics));
        self.reservations.release(work_reservation);
        let frame = result.map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_topk_chunk",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: slice.attempt,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        reserve_final_sort_output(
            self.bounded,
            frame,
            slice.attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }

    fn topk_merge_level_attempt(
        &mut self,
        candidates: &[CommittedPartial],
        merge: TopKMergeAttempt,
    ) -> Result<Vec<TopKLevelItem>> {
        let mut output = Vec::with_capacity(candidates.len().div_ceil(merge.merge_fan_in));

        for (group_index, group) in candidates.chunks(merge.merge_fan_in).enumerate() {
            if group.len() == 1 {
                output.push(TopKLevelItem::Carry { partial_index: group_index * merge.merge_fan_in });
                continue;
            }

            match self.topk_merge_group_attempt(
                group,
                TopKMergeAttempt { keep_rows: merge.keep_rows, merge_fan_in: group.len(), attempt: merge.attempt },
            ) {
                Ok(candidate) => output.push(TopKLevelItem::Merged(candidate)),
                Err(error) => {
                    release_topk_level_items(output, self.reservations);
                    return Err(error);
                },
            }
        }

        Ok(output)
    }

    fn topk_merge_group_attempt(
        &mut self,
        candidates: &[CommittedPartial],
        merge: TopKMergeAttempt,
    ) -> Result<CommittedPartial> {
        debug_assert!(candidates.len() > 1);
        let requested_bytes = candidates.iter().map(|candidate| candidate.reservation.bytes()).sum::<u64>();
        let request =
            MemoryReservationRequest::new(ReservationKind::FinalSort, requested_bytes, self.reservations.budget())
                .with_memory_boundary("final_sort_topk_merge")
                .with_node_id(self.bounded.aggregate_id)
                .with_stage("final_sort")
                .with_attempt(merge.attempt, self.max_attempts);
        let merge_reservation = self.reservations.try_reserve(request, self.metrics)?;

        let result = concat_committed_frames(self.context, candidates, "cudf_query_engine.final_sort.topk_concat")
            .and_then(|frame| execute_topk_candidate(self.context, self.sort, frame, merge.keep_rows, self.metrics));
        self.reservations.release(merge_reservation);
        let frame = result.map_err(|error| {
            RuntimeResourceFacts {
                memory_boundary: "final_sort_topk_merge",
                stage: "final_sort",
                node_id: self.bounded.aggregate_id,
                attempt: merge.attempt,
                max_attempts: self.max_attempts,
                resume_boundary: "final_sort",
                metrics: self.metrics,
            }
            .attach_to(error)
        })?;
        reserve_final_sort_output(
            self.bounded,
            frame,
            merge.attempt,
            self.max_attempts,
            self.reservations,
            self.metrics,
        )
    }
}

pub(super) fn execute_topk_candidate(
    context: &ExecutionContext,
    sort: &SortNode,
    input: GpuDataFrame,
    keep_rows: usize,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let sorted = execute_sort(context, sort, input, metrics)?;
    execute_limit_spec(context, LimitSpec::fetch(keep_rows), sorted, metrics)
}

pub(super) fn concat_committed_frames(
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

pub(super) fn reserve_final_sort_output(
    bounded: &BoundedPlan<'_>,
    frame: GpuDataFrame,
    attempt: usize,
    max_attempts: usize,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<CommittedPartial> {
    let output_bytes = observe_device_memory_bytes(&frame, metrics).map_err(|error| {
        RuntimeResourceFacts {
            memory_boundary: "final_sort_output",
            stage: "final_sort",
            node_id: bounded.aggregate_id,
            attempt,
            max_attempts,
            resume_boundary: "final_sort",
            metrics,
        }
        .attach_to(error)
    })?;
    let output_request = MemoryReservationRequest::new(ReservationKind::FinalSort, output_bytes, reservations.budget())
        .with_memory_boundary("final_sort_output")
        .with_node_id(bounded.aggregate_id)
        .with_stage("final_sort")
        .with_attempt(attempt, max_attempts);
    let output_reservation = reservations.try_reserve(output_request, metrics)?;
    Ok(CommittedPartial { frame, reservation: output_reservation })
}

pub(super) enum TopKLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_topk_merge_level(
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

pub(super) fn release_topk_level_items(items: Vec<TopKLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn release_committed_partials(partials: Vec<CommittedPartial>, reservations: &mut ReservationTracker) {
    for partial in partials {
        reservations.release(partial.reservation);
    }
}

pub(super) fn is_retryable_final_sort_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_final_sort_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn final_sort_retry_facts(retry_action: &'static str, metrics: &QueryMetrics) -> RetryFacts {
    RetryFacts::new()
        .with_retry_action(retry_action)
        .with_resume_boundary("final_sort")
        .with_committed_chunks(metrics.source_chunks_committed as usize)
        .with_committed_partials(metrics.partial_aggregate_count as usize)
}

pub(super) fn final_sort_topk_retry_facts(
    merge_level: usize,
    merge_fan_in: usize,
    metrics: &QueryMetrics,
) -> RetryFacts {
    final_sort_retry_facts(
        if merge_fan_in <= 2 { "fail_min_topk_fan_in" } else { "hierarchical_topk_retry_exhausted" },
        metrics,
    )
    .with_merge_level(merge_level)
    .with_merge_fan_in(merge_fan_in)
}

pub(super) fn topk_candidate_rows(spec: LimitSpec) -> Result<usize> {
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

pub(super) fn estimated_bytes_per_row(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        return bytes.max(1);
    }
    bytes.div_ceil(rows as u64).max(1)
}

pub(super) fn estimate_rows_bytes(bytes_per_row: u64, rows: usize) -> u64 {
    let bytes = u128::from(bytes_per_row) * rows as u128;
    bytes.min(u128::from(u64::MAX)) as u64
}

pub(super) fn final_sort_rows_per_chunk(input_rows: usize, bytes_per_row: u64, available_bytes: u64) -> usize {
    if input_rows == 0 {
        return 1;
    }
    let rows = (available_bytes / bytes_per_row).min(input_rows as u64) as usize;
    rows.max(1)
}

pub(super) fn execute_final_aggregate(
    context: &ExecutionContext,
    bounded: &BoundedPlan<'_>,
    input: GpuDataFrame,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if bounded.aggregate.0.spec.group_exprs.is_empty() {
        let final_spec = final_global_aggregate_spec(&bounded.aggregate.0.spec);
        let output_schema = if aggregate_spec_has_global_final_state(&bounded.aggregate.0.spec) {
            final_global_aggregate_state_output_schema(
                &bounded.aggregate.0.spec,
                bounded.aggregate_input_schema,
                bounded.aggregate.1,
            )?
        } else {
            Arc::clone(bounded.aggregate.1)
        };
        return execute_global_aggregate(context, &final_spec, input, &output_schema, metrics);
    }

    let aggregate = lower_final_aggregate_spec(&bounded.aggregate.0.spec)?;
    if !aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        validate_aggregate_output_names(
            &aggregate,
            bounded.aggregate.1,
            "cudf_query_engine.final_merge.validate",
            "final aggregate output names must match output schema fields",
        )?;
    }
    let expected_output_names = if aggregate_spec_has_avg(&bounded.aggregate.0.spec) {
        aggregate.output_names.clone()
    } else {
        bounded.aggregate.1.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>()
    };

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
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

pub(super) fn final_global_aggregate_spec(spec: &AggregateSpec) -> AggregateSpec {
    AggregateSpec::new(
        Vec::new(),
        spec.aggr_exprs
            .iter()
            .enumerate()
            .flat_map(|(index, expr)| match expr.function {
                AggregateFunction::Count => vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column(expr.alias.clone()),
                    expr.alias.clone(),
                )],
                AggregateFunction::Avg => {
                    let sum_alias = avg_sum_state_alias(index);
                    let count_alias = avg_count_state_alias(index);
                    vec![
                        AggregateExpr::new(AggregateFunction::Sum, Expr::column(sum_alias.clone()), sum_alias),
                        AggregateExpr::new(AggregateFunction::Sum, Expr::column(count_alias.clone()), count_alias),
                    ]
                },
                AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                    let valid_count_alias = global_valid_count_state_alias(expr.alias.as_str());
                    vec![
                        AggregateExpr::new(expr.function, Expr::column(expr.alias.clone()), expr.alias.clone()),
                        AggregateExpr::new(
                            AggregateFunction::Sum,
                            Expr::column(valid_count_alias.clone()),
                            valid_count_alias,
                        ),
                    ]
                },
            })
            .collect(),
    )
}

pub(super) fn aggregate_spec_has_global_final_state(spec: &AggregateSpec) -> bool {
    aggregate_spec_has_avg(spec)
        || spec.group_exprs.is_empty()
            && spec.aggr_exprs.iter().any(|expr| {
                matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
            })
}

pub(super) struct ActiveReservation {
    reserved_bytes: Rc<Cell<u64>>,
    bytes: u64,
    released: bool,
}

impl ActiveReservation {
    pub(super) fn new(reserved_bytes: Rc<Cell<u64>>, bytes: u64) -> Self {
        Self { reserved_bytes, bytes, released: false }
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(super) fn release(&mut self) {
        if self.released {
            return;
        }
        self.reserved_bytes.set(self.reserved_bytes.get().saturating_sub(self.bytes));
        self.released = true;
    }
}

impl Drop for ActiveReservation {
    fn drop(&mut self) {
        self.release();
    }
}

#[derive(Debug)]
pub(super) struct ReservationTracker {
    budget_bytes: u64,
    reserved_bytes: Rc<Cell<u64>>,
}

impl ReservationTracker {
    pub(super) fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes, reserved_bytes: Rc::new(Cell::new(0)) }
    }

    pub(super) fn budget(&self) -> u64 {
        self.budget_bytes
    }

    pub(super) fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes.get()
    }

    pub(super) fn try_reserve(
        &mut self,
        request: MemoryReservationRequest,
        metrics: &mut QueryMetrics,
    ) -> Result<ActiveReservation> {
        let remaining = self.budget_bytes.saturating_sub(self.reserved_bytes());
        if request.requested_bytes > remaining {
            metrics.record_reservation_denied(request.requested_bytes);
            return Err(request.denied_error(remaining));
        }

        self.reserved_bytes.set(self.reserved_bytes().saturating_add(request.requested_bytes));
        metrics.record_reservation_granted(request.requested_bytes, request.requested_bytes);
        Ok(ActiveReservation::new(Rc::clone(&self.reserved_bytes), request.requested_bytes))
    }

    pub(super) fn release(&mut self, mut reservation: ActiveReservation) {
        reservation.release();
    }
}

pub(super) struct CommittedPartial {
    pub(super) frame: GpuDataFrame,
    pub(super) reservation: ActiveReservation,
}

pub(super) fn chunk_requested_bytes(policy: MemoryControlPolicy, chunk: &SourceChunk) -> u64 {
    let fallback = match policy {
        MemoryControlPolicy::Unbounded => 0,
        MemoryControlPolicy::Bounded { max_source_chunk_bytes, .. } => max_source_chunk_bytes,
    };
    chunk.known_row_group_bytes.or(chunk.estimated_chunk_bytes).or(chunk.known_file_bytes).unwrap_or(fallback)
}

pub(super) fn chunk_reservation_request(
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

pub(super) fn is_retryable_chunk_error(error: &QueryError) -> bool {
    matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted)
}

pub(super) fn record_chunk_retry_error(error: &QueryError, metrics: &mut QueryMetrics) {
    match error.code() {
        QueryErrorCode::MemoryReservationFailed => metrics.record_retry_reservation_failure(),
        QueryErrorCode::CudfResourceExhausted => metrics.record_retry_resource_exhausted(),
        _ => {},
    }
}

pub(super) fn split_retryable_chunk(
    chunk: &SourceChunk,
    min_retry_chunk_row_groups: usize,
) -> Option<(SourceChunk, SourceChunk)> {
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

pub(super) fn retry_child_chunk(parent: &SourceChunk, row_group_indexes: Vec<usize>) -> SourceChunk {
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
    let known_row_count = proportional_bytes(parent.known_row_count, row_group_indexes.len(), parent_row_group_count);

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
        partition_values: parent.partition_values.clone(),
        zero_column_count: parent.zero_column_count,
        known_row_count,
        size_estimate_based: true,
    }
}

pub(super) fn proportional_bytes(
    total_bytes: Option<u64>,
    child_row_groups: usize,
    parent_row_groups: usize,
) -> Option<u64> {
    let total_bytes = total_bytes?;
    if parent_row_groups == 0 {
        return Some(total_bytes);
    }

    let numerator = u128::from(total_bytes) * child_row_groups as u128;
    let denominator = parent_row_groups as u128;
    Some(numerator.div_ceil(denominator).min(u128::from(u64::MAX)) as u64)
}

pub(super) fn chunk_retry_facts(chunk: &SourceChunk, metrics: &QueryMetrics) -> RetryFacts {
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

#[derive(Debug, Clone, Copy)]
pub(super) struct RuntimeResourceFacts<'a> {
    pub(super) memory_boundary: &'static str,
    pub(super) stage: &'static str,
    pub(super) node_id: NodeId,
    pub(super) attempt: usize,
    pub(super) max_attempts: usize,
    pub(super) resume_boundary: &'static str,
    pub(super) metrics: &'a QueryMetrics,
}

impl RuntimeResourceFacts<'_> {
    pub(super) fn attach_to(self, error: QueryError) -> QueryError {
        if error.code() != QueryErrorCode::CudfResourceExhausted {
            return error;
        }

        RetryFacts::new()
            .with_resume_boundary(self.resume_boundary)
            .with_committed_chunks(self.metrics.source_chunks_committed as usize)
            .with_committed_partials(self.metrics.partial_aggregate_count as usize)
            .attach_to_error(
                error
                    .with_fact("memory_boundary", self.memory_boundary)
                    .with_fact("stage", self.stage)
                    .with_fact("node_id", self.node_id)
                    .with_fact("attempt", self.attempt)
                    .with_fact("max_attempts", self.max_attempts),
            )
    }
}

pub(super) fn attach_chunk_resource_facts(
    error: QueryError,
    chunk: &SourceChunk,
    memory_boundary: &'static str,
    attempt: usize,
    max_attempts: usize,
    metrics: &QueryMetrics,
) -> QueryError {
    if error.code() != QueryErrorCode::CudfResourceExhausted {
        return error;
    }

    let mut error = RuntimeResourceFacts {
        memory_boundary,
        stage: memory_boundary,
        node_id: chunk.source_node_id,
        attempt,
        max_attempts,
        resume_boundary: "source_chunk",
        metrics,
    }
    .attach_to(error)
    .with_fact("chunk_index", chunk.chunk_index)
    .with_fact("file_index", chunk.file_index)
    .with_fact("path", chunk.path.display());
    if !chunk.reads_all_row_groups {
        error = error.with_fact("row_groups", comma_separated_usize(&chunk.row_group_indexes));
    }
    error
}
