use std::time::Instant;

use super::{
    BoundedCompiledFilteredJoinBuildState, BoundedCompiledFilteredJoinProbeState, BoundedExecutionAttempt,
    BoundedJoinSideFrame, DropDuplicatesOptions, DuplicateKeepOption, ExecutionMetrics, ExecutionOverrides,
    GpuDataFrame, GpuFilteredJoinBuildHandle, JoinNode, JoinSide, JoinType, NodeId, ResidentGpuFrame, Result,
    attach_budget_token_failure_facts, bounded_invariant_violation, cudf_query_error,
    execution_budget_token_for_boundary, execution_overrides, join_index_error, join_side_output_upper_bound_bytes,
    keyed_filtered_semi_anti_join_capable, lower_null_equality, materialize_left_only_join_output_projection,
    materialize_mark_join_output_projection, null_aware_anti_probe, null_aware_anti_probe_input,
    observe_device_memory_bytes, prepare_bounded_join_side_input_from_resident_frame,
    prepare_bounded_join_side_input_from_shared_frame, projected_input_columns,
    projected_output_columns_for_retained_sizing, release_bounded_join_side_frame, validate_executable_join_subset,
};
use crate::{
    exec::{bounded::compiled::lanes::AggregateLaneDispatch, join::NullAwareAntiProbe},
    metrics::{JoinOperatorMetrics, OperatorMetrics},
};

pub(super) fn execute_compiled_filtered_join_build_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    input: GpuDataFrame,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(bounded_invariant_violation(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let setup = filtered_join_build_setup(node);
    let build =
        prepare_bounded_join_side_input_from_shared_frame(execution, input, &setup.right_columns, JoinSide::Right)?;
    build_filtered_join_artifact(execution, node_id, node, setup, build)
}

pub(super) fn execute_compiled_filtered_join_build_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    validate_executable_join_subset(node)?;
    if !keyed_filtered_semi_anti_join_capable(node) {
        return Err(bounded_invariant_violation(
            "compiled filtered join build requires a filtered semi/anti/mark join node",
            "compiled_join_strategy_unsupported",
        ));
    }
    let setup = filtered_join_build_setup(node);
    let build =
        prepare_bounded_join_side_input_from_resident_frame(execution, input, &setup.right_columns, JoinSide::Right)?;
    build_filtered_join_artifact(execution, node_id, node, setup, build)
}

struct FilteredJoinBuildSetup {
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_columns: Vec<String>,
    right_columns: Vec<String>,
}

fn filtered_join_build_setup(node: &JoinNode) -> FilteredJoinBuildSetup {
    let left_keys = node.spec.keys.iter().map(|key| key.left.name().to_string()).collect::<Vec<_>>();
    let right_keys = node.spec.keys.iter().map(|key| key.right.name().to_string()).collect::<Vec<_>>();
    let left_columns = projected_input_columns(&left_keys, &node.spec.output_projection, [], JoinSide::Left);
    let right_columns = projected_input_columns(&right_keys, &node.spec.output_projection, [], JoinSide::Right);
    FilteredJoinBuildSetup { left_keys, right_keys, left_columns, right_columns }
}

fn build_filtered_join_artifact(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    setup: FilteredJoinBuildSetup,
    build: BoundedJoinSideFrame,
) -> Result<BoundedCompiledFilteredJoinBuildState> {
    let FilteredJoinBuildSetup { left_keys, right_keys, left_columns, .. } = setup;
    let input_build_rows = build.rows;
    let input_build_bytes = build.bytes;
    let null_aware_probe = null_aware_anti_probe(node, &build.frame, &right_keys)?;
    let build = deduplicate_filtered_join_build_keys(execution, node_id, build, &right_keys)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let filtered_options =
        cudf::DataFrameFilteredJoinOptions::new().null_equality(lower_null_equality(node.spec.null_equality));
    let timer = Instant::now();
    let filtered_join =
        build.frame.with_execution(&overrides).build_filtered_join(&right_keys, filtered_options).map_err(|error| {
            cudf_query_error(
                "query_engine.bounded.join.build_filtered_join",
                "failed to build compiled bounded filtered join",
                error,
            )
            .with_fact("join_type", node.spec.join_type)
            .with_fact("key_count", node.spec.keys.len())
            .with_fact("build_side", JoinSide::Right)
        })?;
    let filtered_join = GpuFilteredJoinBuildHandle::new(execution.context(), filtered_join, node.spec.null_equality)
        .map_err(|error| {
            error
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", JoinSide::Right)
        })?;
    let hash_table_bytes = filtered_join.device_allocated_size_bytes();
    execution.metrics().record_filtered_join_build();
    execution.metrics().record_filtered_join_hash_table_bytes(hash_table_bytes);
    Ok(BoundedCompiledFilteredJoinBuildState {
        left_keys,
        left_columns,
        build,
        input_build_rows,
        input_build_bytes,
        filtered_join,
        hash_table_bytes,
        null_aware_probe,
        timer,
    })
}

fn null_aware_anti_probe_side(
    execution: &mut BoundedExecutionAttempt<'_>,
    build: &BoundedCompiledFilteredJoinBuildState,
    probe: BoundedJoinSideFrame,
    overrides: &ExecutionOverrides<'_>,
) -> Result<BoundedJoinSideFrame> {
    let Some(frame) = null_aware_anti_probe_input(&probe.frame, &build.left_keys, build.null_aware_probe, overrides)?
    else {
        return Ok(probe);
    };
    let bytes = observe_device_memory_bytes(&GpuDataFrame::new(frame.clone()), execution.metrics())?;
    let rows = frame.num_rows() as u64;
    release_bounded_join_side_frame(probe);
    Ok(BoundedJoinSideFrame { frame, rows, bytes })
}

fn deduplicate_filtered_join_build_keys(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    build: BoundedJoinSideFrame,
    right_keys: &[String],
) -> Result<BoundedJoinSideFrame> {
    let key_columns = right_keys.iter().map(String::as_str).collect::<Vec<_>>();
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "join_filtered_build_key_deduplicate",
        "query_engine.bounded.join.filtered_build_key_deduplicate.budget_token",
        "failed to create filtered join build-key deduplication budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            release_bounded_join_side_frame(build);
            return Err(error.with_fact_if_absent("node_id", node_id));
        },
    };
    let budget_overrides = token.execution_overrides();
    let output = match build.frame.with_execution(&budget_overrides).drop_duplicates(
        DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
    ) {
        Ok(output) => output,
        Err(error) => {
            release_bounded_join_side_frame(build);
            return Err(attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.join.filtered_build_key_deduplicate",
                    "failed to deduplicate compiled filtered join build keys",
                    error,
                ),
                &token,
            )
            .with_fact("key_count", right_keys.len()));
        },
    };
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            release_bounded_join_side_frame(build);
            return Err(error);
        },
    };
    let output_rows = output.num_rows() as u64;
    release_bounded_join_side_frame(build);
    Ok(BoundedJoinSideFrame { frame: output, rows: output_rows, bytes: output_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_from_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: GpuDataFrame,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe =
        prepare_bounded_join_side_input_from_shared_frame(execution, input, &build.left_columns, JoinSide::Left)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let probe = null_aware_anti_probe_side(execution, &build, probe, &overrides)?;
    let (probe_state, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node,
        &build,
        &probe,
        &overrides,
        "failed to observe compiled bounded filtered join probe-state memory",
    )?;
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_bytes })
}

pub(super) fn execute_compiled_filtered_join_probe_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<BoundedCompiledFilteredJoinProbeState> {
    let probe =
        prepare_bounded_join_side_input_from_resident_frame(execution, input, &build.left_columns, JoinSide::Left)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let probe = null_aware_anti_probe_side(execution, &build, probe, &overrides)?;
    let (probe_state, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node,
        &build,
        &probe,
        &overrides,
        "failed to observe compiled bounded filtered join probe-state memory",
    )?;
    Ok(BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_bytes })
}

fn begin_filtered_join_probe_state(
    execution: &mut BoundedExecutionAttempt<'_>,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    probe: &BoundedJoinSideFrame,
    overrides: &ExecutionOverrides<'_>,
    probe_state_bytes_message: &'static str,
) -> Result<(cudf::DataFrameFilteredJoinProbeState, u64)> {
    let probe_state = match node.spec.join_type {
        JoinType::LeftSemi | JoinType::LeftMark => {
            build.filtered_join.filtered_join().begin_left_semi_probe_in(&probe.frame, &build.left_keys, overrides)
        },
        JoinType::LeftAnti => {
            build.filtered_join.filtered_join().begin_left_anti_probe_in(&probe.frame, &build.left_keys, overrides)
        },
        JoinType::Inner | JoinType::Left | JoinType::Full | JoinType::Cross => unreachable!("validated above"),
    }
    .map_err(|error| join_index_error(error, node))?;
    let probe_state_bytes = probe_state.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        cudf_query_error("query_engine.bounded.join.filtered_probe_state_bytes", probe_state_bytes_message, error)
            .with_fact("join_type", node.spec.join_type)
            .with_fact("build_side", JoinSide::Right)
    })?;
    execution.metrics().record_filtered_join_probe_state_bytes(probe_state_bytes);
    Ok((probe_state, probe_state_bytes))
}

pub(in crate::exec::bounded) fn execute_compiled_filtered_join_probe_chunk_from_resident_frame(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    input: ResidentGpuFrame,
) -> Result<ResidentGpuFrame> {
    let probe =
        prepare_bounded_join_side_input_from_resident_frame(execution, input, &build.left_columns, JoinSide::Left)?;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let probe = null_aware_anti_probe_side(execution, build, probe, &overrides)?;
    let (probe_state, probe_state_bytes) = begin_filtered_join_probe_state(
        execution,
        node,
        build,
        &probe,
        &overrides,
        "failed to observe compiled streamed filtered join probe-state memory",
    )?;

    let indices = match probe_state.materialize_indices_in(&overrides) {
        Ok(indices) => indices,
        Err(error) => {
            return Err(join_index_error(error, node));
        },
    };
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled streamed filtered join gather-map memory",
            error,
        )
    })?;
    let output = if node.spec.join_type == JoinType::LeftMark {
        let mark_column_name = node.spec.mark_column_name.as_deref().unwrap_or("mark");
        // Mark join: estimate reservation as probe size (all rows kept) + mark column overhead.
        let output_reservation_bytes = probe.bytes.saturating_add(probe.rows.saturating_mul(1));
        let output = materialize_mark_join_output_projection(
            execution.context(),
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics())?;
        record_filtered_join_output_metrics(
            execution.metrics(),
            node_id,
            node,
            build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes)
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            execution,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
        )?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics())?;
        record_filtered_join_output_metrics(
            execution.metrics(),
            node_id,
            node,
            build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes)
    };
    Ok(output)
}

pub(super) fn execute_compiled_filtered_join_materialize(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    state: BoundedCompiledFilteredJoinProbeState,
) -> Result<ResidentGpuFrame> {
    let BoundedCompiledFilteredJoinProbeState { build, probe, probe_state, probe_state_bytes } = state;
    let overrides_lease = execution_overrides(execution.context())?;
    let overrides = overrides_lease.overrides();
    let indices = probe_state.materialize_indices_in(&overrides).map_err(|error| join_index_error(error, node))?;
    let gather_map_bytes = indices.device_allocated_size_bytes().map(|bytes| bytes as u64).map_err(|error| {
        cudf_query_error(
            "query_engine.bounded.join.filtered_gather_map_bytes",
            "failed to observe compiled bounded filtered join gather-map memory",
            error,
        )
    })?;
    let output = if node.spec.join_type == JoinType::LeftMark {
        let mark_column_name = node.spec.mark_column_name.as_deref().unwrap_or("mark");
        let output_reservation_bytes = probe.bytes.saturating_add(probe.rows.saturating_mul(1));
        let output = materialize_mark_join_output_projection(
            execution.context(),
            &probe.frame,
            &indices,
            &node.spec.output_projection,
            mark_column_name,
        )?;
        execution.metrics().add_operator_time(build.timer.elapsed());
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics())?;
        record_filtered_join_output_metrics(
            execution.metrics(),
            node_id,
            node,
            &build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes)
    } else {
        let left_projection =
            projected_output_columns_for_retained_sizing(&node.spec.output_projection, JoinSide::Left);
        let output_reservation_bytes = join_side_output_upper_bound_bytes(
            execution,
            node_id,
            &probe.frame,
            indices.len() as u64,
            &left_projection,
            JoinSide::Left,
        )?;
        let output = materialize_left_only_join_output_projection(
            &probe.frame,
            &indices,
            &overrides,
            &node.spec.output_projection,
            node.spec.join_type,
        )?;
        execution.metrics().add_operator_time(build.timer.elapsed());
        let output_rows = output.num_rows() as u64;
        let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics())?;
        record_filtered_join_output_metrics(
            execution.metrics(),
            node_id,
            node,
            &build,
            &probe,
            probe_state_bytes,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        );
        ResidentGpuFrame::with_device_bytes(GpuDataFrame::new(output), output_bytes)
    };
    Ok(output)
}

#[expect(
    clippy::too_many_arguments,
    reason = "strategy-decision facts are reported individually; grouping them would be a metrics DTO"
)]
fn record_filtered_join_output_metrics(
    metrics: &ExecutionMetrics,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    probe: &BoundedJoinSideFrame,
    probe_state_bytes: u64,
    gather_map_bytes: u64,
    output_reservation_bytes: u64,
    output_rows: u64,
    output_bytes: u64,
) {
    metrics.record_operator_metrics(OperatorMetrics::join(
        node_id,
        JoinOperatorMetrics {
            query_id: None,
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_reusable_filtered_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: false,
            null_equality: node.spec.null_equality.to_string(),
            build_side: JoinSide::Right.to_string(),
            probe_side: JoinSide::Left.to_string(),
            build_rows: build.input_build_rows,
            build_bytes: build.input_build_bytes,
            probe_rows: probe.rows,
            probe_bytes: probe.bytes,
            hash_table_bytes: build.hash_table_bytes,
            filtered_probe_state_bytes: probe_state_bytes,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
}

pub(super) fn release_compiled_filtered_join_build_state(build: BoundedCompiledFilteredJoinBuildState) {
    let BoundedCompiledFilteredJoinBuildState { build, .. } = build;
    release_bounded_join_side_frame(build);
}

pub(super) fn release_compiled_filtered_join_probe_state(probe: BoundedCompiledFilteredJoinProbeState) {
    let BoundedCompiledFilteredJoinProbeState { build, probe, .. } = probe;
    release_compiled_filtered_join_build_state(build);
    release_bounded_join_side_frame(probe);
}

/// Whether a filtered probe may run chunks in parallel with concatenation.
///
/// Membership is per probe row into private probe state; null-aware anti
/// facts are build-time. Only right-build semi/anti/mark shares the
/// chunk-independent contract.
pub(in crate::exec::bounded) fn filtered_parallel_probe_supported(node: &JoinNode) -> bool {
    matches!(node.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark)
        && node.spec.build_side == JoinSide::Right
}

struct FilteredProbeWorkState {
    pending: std::collections::VecDeque<ResidentGpuFrame>,
    outputs: Vec<ResidentGpuFrame>,
}

/// Minimal Send+Sync view of a filtered build shared across probe lanes.
///
/// The native filtered handle is read-only across probes with private probe
/// state per lane; the timer measures the coordinator build and stays out of
/// the shared view.
struct SharedFilteredBuild {
    left_keys: Vec<String>,
    left_columns: Vec<String>,
    build_frame: cudf::DataFrame,
    build_rows: u64,
    build_bytes: u64,
    filtered_join: GpuFilteredJoinBuildHandle,
    hash_table_bytes: u64,
    null_aware_probe: NullAwareAntiProbe,
}

impl SharedFilteredBuild {
    fn from_build(build: &BoundedCompiledFilteredJoinBuildState) -> Self {
        Self {
            left_keys: build.left_keys.clone(),
            left_columns: build.left_columns.clone(),
            build_frame: build.build.frame.clone(),
            build_rows: build.build.rows,
            build_bytes: build.build.bytes,
            filtered_join: build.filtered_join.clone(),
            hash_table_bytes: build.hash_table_bytes,
            null_aware_probe: build.null_aware_probe,
        }
    }

    fn as_build_ref(&self) -> BoundedCompiledFilteredJoinBuildState {
        BoundedCompiledFilteredJoinBuildState {
            left_keys: self.left_keys.clone(),
            left_columns: self.left_columns.clone(),
            build: BoundedJoinSideFrame {
                frame: self.build_frame.clone(),
                rows: self.build_rows,
                bytes: self.build_bytes,
            },
            input_build_rows: self.build_rows,
            input_build_bytes: self.build_bytes,
            filtered_join: self.filtered_join.clone(),
            hash_table_bytes: self.hash_table_bytes,
            null_aware_probe: self.null_aware_probe,
            timer: Instant::now(),
        }
    }
}

/// Probe eager filtered chunks in parallel sharing one build.
///
/// Outputs keep input chunk order; the table is built once regardless of
/// probe batch count. The `outer` dispatcher contract matches the hash
/// parallel probe: reuse the run's lanes when present, borrow available
/// lanes otherwise.
pub(in crate::exec::bounded) fn parallel_filtered_probe_frames(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    frames: Vec<ResidentGpuFrame>,
    outer: Option<&AggregateLaneDispatch>,
) -> Result<Vec<ResidentGpuFrame>> {
    if frames.len() <= 1 {
        return serial_filtered_probe_frames(execution, node_id, node, build, frames.into());
    }
    let mut state = FilteredProbeWorkState { pending: frames.into(), outputs: Vec::new() };
    let chunk_count = state.pending.len();
    let shared = std::sync::Arc::new(SharedFilteredBuild::from_build(build));
    let node_owned = node.clone();
    let mut run_on = |dispatch: &AggregateLaneDispatch, execution: &mut BoundedExecutionAttempt<'_>| {
        let width = dispatch.width();
        execution.metrics().record_join_strategy_decision(
            node_id,
            &std::collections::BTreeMap::from([
                ("parallel_probe", "filtered".to_string()),
                ("parallel_probe_chunks", chunk_count.to_string()),
                ("parallel_probe_width", width.to_string()),
            ]),
        );
        dispatch.execute_work(
            execution,
            &mut state,
            |state, _| Ok(state.pending.pop_front()),
            {
                let shared = std::sync::Arc::clone(&shared);
                let node_owned = node_owned.clone();
                move |lane_execution, chunk| {
                    let build_ref = shared.as_build_ref();
                    execute_compiled_filtered_join_probe_chunk_from_resident_frame(
                        lane_execution,
                        node_id,
                        &node_owned,
                        &build_ref,
                        chunk,
                    )
                }
            },
            |state, _, output| {
                state.outputs.push(output);
                Ok(())
            },
        )
    };
    match outer {
        Some(dispatch) => run_on(dispatch, execution)?,
        None => {
            AggregateLaneDispatch::with_available_lanes(execution, |dispatch, execution| run_on(dispatch, execution))?
        },
    }
    Ok(state.outputs)
}

fn serial_filtered_probe_frames(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    build: &BoundedCompiledFilteredJoinBuildState,
    mut frames: std::collections::VecDeque<ResidentGpuFrame>,
) -> Result<Vec<ResidentGpuFrame>> {
    let mut outputs = Vec::new();
    while let Some(chunk) = frames.pop_front() {
        outputs.push(execute_compiled_filtered_join_probe_chunk_from_resident_frame(
            execution, node_id, node, build, chunk,
        )?);
    }
    Ok(outputs)
}
