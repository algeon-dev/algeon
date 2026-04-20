use super::*;
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

pub(super) fn is_bfs_path_output(spec: &CugraphAlgorithmSpec) -> bool {
    matches!(
        &spec.params,
        CugraphAlgorithmParams::Bfs(params) if params.output.mode == CugraphBfsOutputMode::Path
    )
}

pub(super) fn bfs_decoded_output_schema_for_host(spec: &CugraphAlgorithmSpec) -> SchemaRef {
    let CugraphAlgorithmParams::Bfs(params) = &spec.params else {
        return Arc::clone(&spec.output_schema);
    };
    if params.output.mode != CugraphBfsOutputMode::Path {
        return Arc::clone(&spec.output_schema);
    }
    Arc::new(Schema::new(vec![
        arrow::datatypes::Field::new("vertex", DataType::Utf8, false),
        arrow::datatypes::Field::new("distance", DataType::Int64, false),
        arrow::datatypes::Field::new("predecessor", DataType::Utf8, true),
    ]))
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

pub(super) fn bfs_path_target_vertex(
    context: &ExecutionContext,
    spec: &CugraphAlgorithmSpec,
    predicate_frames: &[CugraphGpuFrame],
) -> Result<Option<BfsPathVertex>> {
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
    let batch = handle.to_host_record_batch().map_err(|error| {
        DataFusionError::from(Error::from_native(
            "datafusion_nexus.cugraph_sql.path_output",
            "failed to materialize BFS target predicate source for path reconstruction",
            error,
        ))
    })?;
    if batch.num_rows() != 1 {
        return Err(plan_error(format!(
            "BFS path output requires exactly one target vertex row, found {}",
            batch.num_rows()
        )));
    }
    let column_index = batch.schema().index_of(&target.column).map_err(DataFusionError::from)?;
    vertex_value(batch.column(column_index).as_ref(), 0).map(Some)
}

pub(super) fn reconstruct_bfs_path_batch(
    spec: &CugraphAlgorithmSpec,
    batch: &RecordBatch,
    target: BfsPathVertex,
) -> Result<RecordBatch> {
    let CugraphAlgorithmParams::Bfs(_params) = &spec.params else {
        return Err(plan_error("BFS path reconstruction received non-BFS parameters"));
    };
    let vertex_col = batch.schema().index_of("vertex").map_err(DataFusionError::from)?;
    let distance_col = batch.schema().index_of("distance").map_err(DataFusionError::from)?;
    let predecessor_col = batch.schema().index_of("predecessor").map_err(DataFusionError::from)?;
    let mut nodes = HashMap::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let vertex = vertex_value(batch.column(vertex_col).as_ref(), row)?;
        let distance = optional_i64_value(batch.column(distance_col).as_ref(), row)?;
        let predecessor = optional_vertex_value(batch.column(predecessor_col).as_ref(), row)?;
        nodes.insert(vertex, BfsPathNode { distance, predecessor });
    }
    let sources = bfs_path_source_vertices(&nodes);
    let (source, path) = bfs_path_vertices(&nodes, &sources, target.clone())?;
    let source = source.unwrap_or_else(|| target.clone());
    let distances = path
        .iter()
        .enumerate()
        .map(|(index, vertex)| nodes.get(vertex).and_then(|node| node.distance).unwrap_or(index as i64))
        .collect::<Vec<_>>();
    build_bfs_path_batch(Arc::clone(&spec.output_schema), source, target, path, distances)
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
