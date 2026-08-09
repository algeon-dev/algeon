use super::{node::projected_schema, *};
pub(super) fn project_and_limit_host_batch(
    batch: RecordBatch,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<RecordBatch> {
    let mut batch = if let Some(limit) = limit { batch.slice(0, limit.min(batch.num_rows())) } else { batch };
    if let Some(projection) = projection {
        batch = batch.project(projection).map_err(DataFusionError::from)?;
    }
    Ok(batch)
}

pub(super) fn apply_host_projection_limit(
    output: EagerCugraphHostOutput,
    spec: &CugraphAlgorithmSpec,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
) -> Result<EagerCugraphHostOutput> {
    if projection.is_none() && limit.is_none() {
        return Ok(output);
    }
    let schema = projected_schema(&spec.output_schema, projection)?;
    let (batches, reservation, host_evidence) = output.into_parts_with_evidence();
    let old_bytes =
        batches.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
    let old_bytes = old_bytes.ok_or_else(|| plan_error("logical BFS host output size overflowed usize"))?;
    let mut remaining = limit;
    let mut projected = Vec::with_capacity(batches.len());
    for batch in batches.iter() {
        let Some(remaining_rows) = remaining else {
            projected.push(project_and_limit_host_batch(batch.clone(), projection, None)?);
            continue;
        };
        if remaining_rows == 0 {
            break;
        }
        let limited = project_and_limit_host_batch(batch.clone(), projection, Some(remaining_rows))?;
        remaining = remaining_rows.checked_sub(limited.num_rows());
        projected.push(limited);
    }
    let new_bytes =
        projected.iter().map(get_record_batch_memory_size).try_fold(0usize, |sum, bytes| sum.checked_add(bytes));
    let new_bytes = new_bytes.ok_or_else(|| plan_error("projected logical BFS host output size overflowed usize"))?;
    if let Some(evidence) = host_evidence.as_ref() {
        evidence.check_output_growth("projected_output", reservation.size(), new_bytes)?;
    }
    reservation.try_grow(new_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied projected logical BFS output staging",
            )
            .with_fact("boundary", "projected_output")
            .with_fact("requested_bytes", new_bytes)
            .with_source(source),
        )
    })?;
    // End the superseded batch ownership before lowering its charged output bytes.
    drop(batches);
    reservation.shrink(old_bytes);
    Ok(EagerCugraphHostOutput::from_shared_parts_with_evidence(schema, Arc::new(projected), reservation, host_evidence))
}

pub(super) fn is_bfs_path_output(spec: &CugraphAlgorithmSpec) -> bool {
    matches!(
        &spec.params,
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
    )
}

#[derive(Debug, Clone, Eq)]
pub(super) enum BfsPathVertex {
    Int64(i64),
    Utf8(String),
}

impl PartialEq for BfsPathVertex {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int64(left), Self::Int64(right)) => left == right,
            (Self::Utf8(left), Self::Utf8(right)) => left == right,
            _ => false,
        }
    }
}

impl Hash for BfsPathVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Int64(value) => {
                0_u8.hash(state);
                value.hash(state);
            },
            Self::Utf8(value) => {
                1_u8.hash(state);
                value.hash(state);
            },
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct BfsPathNode {
    distance: Option<i64>,
    predecessor: Option<BfsPathVertex>,
}

pub(super) struct BoundedBfsPathTarget {
    vertex: BfsPathVertex,
    reservation: Arc<MemoryReservation>,
}

impl BoundedBfsPathTarget {
    #[cfg(test)]
    pub(super) fn for_test(vertex: BfsPathVertex, reservation: Arc<MemoryReservation>) -> Self {
        Self { vertex, reservation }
    }

    pub(super) fn map_vertex(self, codebook: &HostBfsCodebook) -> Result<Self> {
        let vertex = match self.vertex {
            BfsPathVertex::Int64(code) => BfsPathVertex::Utf8(codebook.decode_code(code)?),
            vertex => vertex,
        };
        Ok(Self { vertex, reservation: self.reservation })
    }
}

pub(super) fn bfs_path_target_vertex(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    predicate_frames: &[CugraphGpuFrame],
    task_context: &TaskContext,
    max_staging_bytes: usize,
    host_evidence: Option<&HostReservationEvidence>,
    metrics: &mut GraphExecutionMetrics<'_>,
) -> Result<Option<BoundedBfsPathTarget>> {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Ok(None);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Ok(None);
    }
    let target =
        params.predicates.target_vertices.as_ref().ok_or_else(|| {
            plan_error("BFS path output requires target_vertices_table and target_vertex_col options")
        })?;
    let mut target_index = usize::from(params.source.planned_source_vertices_input());
    if params.predicates.include_vertices.is_some() {
        target_index += 1;
    }
    if params.predicates.exclude_vertices.is_some() {
        target_index += 1;
    }
    let frame = predicate_frames
        .get(target_index)
        .ok_or_else(|| plan_error("BFS path output target predicate source was not planned"))?;
    let handle = GpuResultHandle::new(context, frame.frame.clone(), Arc::clone(&frame.schema)).map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.path_output",
            "failed to bind BFS target predicate source for path reconstruction",
            error,
        ))
    })?;
    let output = materialize_gpu_output_eagerly(
        handle,
        task_context,
        context.cancellation().clone(),
        max_staging_bytes,
        host_evidence,
        metrics,
    )?;
    let (batches, reservation) = output.into_parts();
    let row_count = batches
        .iter()
        .try_fold(0usize, |rows, batch| rows.checked_add(batch.num_rows()))
        .ok_or_else(|| host_staging_limit_error("bfs_path_target", "output", 0, usize::MAX, max_staging_bytes))?;
    if row_count != 1 {
        return Err(plan_error(format!("BFS path output requires exactly one target vertex row, found {}", row_count)));
    }
    let batch = batches.iter().find(|batch| batch.num_rows() == 1).expect("row count checked above");
    let column_index = batch.schema().index_of(&target.column).map_err(DataFusionError::from)?;
    let target_bytes = bfs_path_vertex_owned_bytes(batch.column(column_index).as_ref(), 0)?;
    reserve_bfs_path_bytes(&reservation, "target_vertex", target_bytes, max_staging_bytes)?;
    let vertex = vertex_value(batch.column(column_index).as_ref(), 0)?;
    // Release traversal batches before retaining only the target vertex charge.
    drop(batches);
    if reservation.size() > target_bytes {
        reservation.shrink(reservation.size() - target_bytes);
    }
    Ok(Some(BoundedBfsPathTarget { vertex, reservation }))
}

pub(super) fn materialize_bfs_path_output_from_host(
    output: EagerCugraphHostOutput,
    spec: &CugraphAlgorithmSpec,
    target: BoundedBfsPathTarget,
    projection: Option<&Vec<usize>>,
    limit: Option<usize>,
    max_staging_bytes: usize,
) -> Result<(RecordBatch, Arc<MemoryReservation>)> {
    let BoundedBfsPathTarget { vertex: target, reservation: target_reservation } = target;
    let output_limit_bytes = output
        .host_evidence()
        .map_or(max_staging_bytes, |evidence| evidence.output_limit_bytes().min(max_staging_bytes));
    let available_staging_bytes = output_limit_bytes.checked_sub(target_reservation.size()).ok_or_else(|| {
        host_staging_limit_error("bfs_path", "target_vertex", target_reservation.size(), 0, output_limit_bytes)
    })?;
    let (batches, reservation, _host_evidence) = output.into_parts_with_evidence();
    let row_count =
        batches.iter().try_fold(0usize, |rows, batch| rows.checked_add(batch.num_rows())).ok_or_else(|| {
            host_staging_limit_error("bfs_path", "reconstruction_workspace", 0, usize::MAX, available_staging_bytes)
        })?;
    let workspace_bytes = bfs_path_workspace_upper_bound(reservation.size(), row_count).ok_or_else(|| {
        host_staging_limit_error(
            "bfs_path",
            "reconstruction_workspace",
            reservation.size(),
            usize::MAX,
            available_staging_bytes,
        )
    })?;
    reserve_bfs_path_bytes(&reservation, "reconstruction_workspace", workspace_bytes, available_staging_bytes)?;

    let (source, path, distances) = bfs_path_components(&batches, target.clone())?;
    validate_bfs_path_vertex_types(&source, &target, &path)?;
    let final_bytes = bfs_path_output_upper_bound(&source, &target, &path).ok_or_else(|| {
        host_staging_limit_error("bfs_path", "final_output", reservation.size(), usize::MAX, available_staging_bytes)
    })?;
    reserve_bfs_path_bytes(&reservation, "final_output", final_bytes, available_staging_bytes)?;
    let path = build_bfs_path_batch(Arc::clone(&spec.output_schema), source, target, path, distances)?;
    let path = project_and_limit_host_batch(path, projection, limit)?;
    let actual_bytes = get_record_batch_memory_size(&path);
    if actual_bytes > final_bytes {
        return Err(Error::new(
            ErrorCode::CudfInvariantViolation,
            "datafusion_nexus.cugraph_sql.path_output",
            "materialized BFS path output exceeded its reserved upper bound",
        )
        .with_fact("reserved_final_bytes", final_bytes)
        .with_fact("actual_final_bytes", actual_bytes)
        .into());
    }
    let retained_bytes = reservation.size();
    if retained_bytes > actual_bytes {
        reservation.shrink(retained_bytes - actual_bytes);
    }
    Ok((path, reservation))
}

fn validate_bfs_path_vertex_types(
    source: &BfsPathVertex,
    target: &BfsPathVertex,
    path: &[BfsPathVertex],
) -> Result<()> {
    let all_int64 = matches!((source, target), (BfsPathVertex::Int64(_), BfsPathVertex::Int64(_)))
        && path.iter().all(|vertex| matches!(vertex, BfsPathVertex::Int64(_)));
    let all_utf8 = matches!((source, target), (BfsPathVertex::Utf8(_), BfsPathVertex::Utf8(_)))
        && path.iter().all(|vertex| matches!(vertex, BfsPathVertex::Utf8(_)));
    if all_int64 || all_utf8 {
        return Ok(());
    }
    Err(plan_error("BFS path output source, target, and path vertex dtypes must match"))
}

fn reserve_bfs_path_bytes(
    reservation: &MemoryReservation,
    boundary: &'static str,
    requested_bytes: usize,
    max_staging_bytes: usize,
) -> Result<()> {
    let staged_bytes = reservation.size();
    let next_bytes = staged_bytes.checked_add(requested_bytes).ok_or_else(|| {
        host_staging_limit_error("bfs_path", boundary, staged_bytes, requested_bytes, max_staging_bytes)
    })?;
    if next_bytes > max_staging_bytes {
        return Err(host_staging_limit_error("bfs_path", boundary, staged_bytes, requested_bytes, max_staging_bytes));
    }
    reservation.try_grow(requested_bytes).map_err(|source| {
        DataFusionError::from(
            Error::new(
                ErrorCode::MemoryReservationFailed,
                "datafusion_nexus.cugraph_sql.host_staging_reservation",
                "DataFusion host memory pool denied cuGraph BFS path staging",
            )
            .with_fact("boundary", boundary)
            .with_fact("requested_bytes", requested_bytes)
            .with_fact("staged_bytes", staged_bytes)
            .with_source(source),
        )
    })
}

fn bfs_path_workspace_upper_bound(raw_bytes: usize, row_count: usize) -> Option<usize> {
    raw_bytes.checked_mul(6)?.checked_add(row_count.checked_mul(256)?)?.checked_add(4096)
}

fn bfs_path_output_upper_bound(
    source: &BfsPathVertex,
    target: &BfsPathVertex,
    path: &[BfsPathVertex],
) -> Option<usize> {
    let rows = path.len();
    let fixed_width_bytes = rows.checked_mul(40)?;
    let string_bytes = match (source, target) {
        (BfsPathVertex::Int64(_), BfsPathVertex::Int64(_)) => 0,
        (BfsPathVertex::Utf8(source), BfsPathVertex::Utf8(target)) => {
            source.len().checked_mul(rows)?.checked_add(target.len().checked_mul(rows)?)?.checked_add(
                path.iter().try_fold(0usize, |bytes, vertex| match vertex {
                    BfsPathVertex::Utf8(value) => bytes.checked_add(value.len()),
                    BfsPathVertex::Int64(_) => None,
                })?,
            )?
        },
        _ => return None,
    };
    fixed_width_bytes.checked_add(string_bytes)?.checked_mul(2)?.checked_add(4096)
}

fn bfs_path_vertex_owned_bytes(array: &dyn Array, row: usize) -> Result<usize> {
    let base = std::mem::size_of::<BfsPathVertex>();
    if array.is_null(row) {
        return Ok(base);
    }
    let string_bytes = if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        array.value(row).len()
    } else if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        array.value(row).len()
    } else if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        array.value(row).len()
    } else {
        0
    };
    base.checked_add(string_bytes).ok_or_else(|| plan_error("BFS path target vertex size overflowed usize"))
}

#[cfg(test)]
pub(super) fn reconstruct_bfs_path_batch(
    spec: &CugraphAlgorithmSpec,
    batch: &RecordBatch,
    target: BfsPathVertex,
) -> Result<RecordBatch> {
    let CugraphAlgorithmParams::Bfs(_params) = &spec.params else {
        return Err(plan_error("BFS path reconstruction received non-BFS parameters"));
    };
    let (source, path, distances) = bfs_path_components(std::slice::from_ref(batch), target.clone())?;
    validate_bfs_path_vertex_types(&source, &target, &path)?;
    build_bfs_path_batch(Arc::clone(&spec.output_schema), source, target, path, distances)
}

fn bfs_path_components(
    batches: &[RecordBatch],
    target: BfsPathVertex,
) -> Result<(BfsPathVertex, Vec<BfsPathVertex>, Vec<i64>)> {
    let row_count = batches
        .iter()
        .try_fold(0usize, |rows, batch| rows.checked_add(batch.num_rows()))
        .ok_or_else(|| plan_error("BFS path reconstruction row count overflowed usize"))?;
    let mut nodes = HashMap::with_capacity(row_count);
    for batch in batches {
        let vertex_col = batch.schema().index_of("vertex").map_err(DataFusionError::from)?;
        let distance_col = batch.schema().index_of("distance").map_err(DataFusionError::from)?;
        let predecessor_col = batch.schema().index_of("predecessor").map_err(DataFusionError::from)?;
        for row in 0..batch.num_rows() {
            let vertex = vertex_value(batch.column(vertex_col).as_ref(), row)?;
            let distance = optional_i64_value(batch.column(distance_col).as_ref(), row)?;
            let predecessor = optional_vertex_value(batch.column(predecessor_col).as_ref(), row)?;
            nodes.insert(vertex, BfsPathNode { distance, predecessor });
        }
    }
    let sources = bfs_path_source_vertices(&nodes);
    let (source, path) = bfs_path_vertices(&nodes, &sources, target.clone())?;
    let source = source.unwrap_or_else(|| target.clone());
    let distances = path
        .iter()
        .enumerate()
        .map(|(index, vertex)| nodes.get(vertex).and_then(|node| node.distance).unwrap_or(index as i64))
        .collect::<Vec<_>>();
    Ok((source, path, distances))
}

pub(super) fn bfs_path_source_vertices(nodes: &HashMap<BfsPathVertex, BfsPathNode>) -> HashSet<BfsPathVertex> {
    nodes
        .iter()
        .filter_map(|(vertex, node)| (node.distance == Some(0) && node.predecessor.is_none()).then_some(vertex.clone()))
        .collect()
}

pub(super) fn bfs_path_vertices(
    nodes: &HashMap<BfsPathVertex, BfsPathNode>,
    sources: &HashSet<BfsPathVertex>,
    target: BfsPathVertex,
) -> Result<(Option<BfsPathVertex>, Vec<BfsPathVertex>)> {
    let Some(target_node) = nodes.get(&target) else {
        return Ok((None, Vec::new()));
    };
    if target_node.distance.is_none() {
        return Ok((None, Vec::new()));
    }
    let mut reversed = Vec::new();
    let mut current = target;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(current.clone()) {
            return Err(plan_error("BFS path reconstruction encountered a predecessor cycle"));
        }
        reversed.push(current.clone());
        if sources.contains(&current) {
            let source = current.clone();
            reversed.reverse();
            return Ok((Some(source), reversed));
        }
        let node = nodes.get(&current).ok_or_else(|| {
            plan_error("BFS path reconstruction found a predecessor vertex missing from the BFS output")
        })?;
        let predecessor = node.predecessor.clone().ok_or_else(|| {
            plan_error("BFS path reconstruction reached a vertex without a predecessor before any source")
        })?;
        current = predecessor;
    }
}

pub(super) fn build_bfs_path_batch(
    schema: SchemaRef,
    source: BfsPathVertex,
    target: BfsPathVertex,
    path: Vec<BfsPathVertex>,
    distances: Vec<i64>,
) -> Result<RecordBatch> {
    let path_index =
        Arc::new(Int64Array::from((0..path.len()).map(|value| value as i64).collect::<Vec<_>>())) as ArrayRef;
    let distance = Arc::new(Int64Array::from(distances)) as ArrayRef;
    let (source_column, target_column, vertex_column): (ArrayRef, ArrayRef, ArrayRef) = match (&source, &target) {
        (BfsPathVertex::Int64(source), BfsPathVertex::Int64(target)) => (
            Arc::new(Int64Array::from(vec![*source; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(vec![*target; path.len()])) as ArrayRef,
            Arc::new(Int64Array::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Int64(value) => *value,
                        BfsPathVertex::Utf8(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (BfsPathVertex::Utf8(source), BfsPathVertex::Utf8(target)) => (
            Arc::new(StringArray::from(vec![source.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(vec![target.clone(); path.len()])) as ArrayRef,
            Arc::new(StringArray::from(
                path.iter()
                    .map(|vertex| match vertex {
                        BfsPathVertex::Utf8(value) => value.clone(),
                        BfsPathVertex::Int64(_) => unreachable!("path vertex type checked by source/target"),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        _ => {
            return Err(plan_error("BFS path output source and target vertex dtypes must match"));
        },
    };
    RecordBatch::try_new(schema, vec![path_index, source_column, target_column, vertex_column, distance])
        .map_err(DataFusionError::from)
}

pub(super) fn vertex_value(array: &dyn Array, row: usize) -> Result<BfsPathVertex> {
    if array.is_null(row) {
        return Err(plan_error("BFS path vertex values must be non-null"));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        return Ok(BfsPathVertex::Int64(array.value(row)));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        return Ok(BfsPathVertex::Int64(i64::from(array.value(row))));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    if let Some(array) = array.as_any().downcast_ref::<StringViewArray>() {
        return Ok(BfsPathVertex::Utf8(array.value(row).to_string()));
    }
    Err(plan_error(format!("BFS path reconstruction does not support vertex array type {:?}", array.data_type())))
}

pub(super) fn optional_vertex_value(array: &dyn Array, row: usize) -> Result<Option<BfsPathVertex>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let value = vertex_value(array, row)?;
    if value == BfsPathVertex::Int64(-1) {
        return Ok(None);
    }
    Ok(Some(value))
}

pub(super) fn optional_i64_value(array: &dyn Array, row: usize) -> Result<Option<i64>> {
    if array.is_null(row) {
        return Ok(None);
    }
    if let Some(array) = array.as_any().downcast_ref::<Int64Array>() {
        let value = array.value(row);
        if value == i64::MAX {
            return Ok(None);
        }
        return Ok(Some(value));
    }
    if let Some(array) = array.as_any().downcast_ref::<Int32Array>() {
        let value = array.value(row);
        if value == i32::MAX {
            return Ok(None);
        }
        return Ok(Some(i64::from(value)));
    }
    Err(plan_error(format!("BFS path reconstruction does not support distance array type {:?}", array.data_type())))
}
