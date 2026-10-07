use super::*;

/// Run one vertex-keyed graph algorithm over `edge_batch` and return its host
/// output rows as `(vertex, bits)` sorted by vertex.
///
/// `bits` carries the second output column verbatim: `label` values for
/// component algorithms, `f64::to_bits`/`f32::to_bits` for betweenness scores.
/// Comparing the full vectors across runs is a bitwise stability check.
fn run_vertex_keyed_algorithm(
    context: &BackendQueryContext,
    algorithm: GraphAlgorithm,
    params: GraphAlgorithmParams,
    directed: bool,
    edge_schema: &Arc<Schema>,
    edge_batch: &RecordBatch,
) -> Vec<(i64, u64)> {
    let mut builder = QueryPlanBuilder::new();
    let edges =
        builder.add_source(SourceNode::GpuResident(gpu_resident_source("edges", Arc::clone(edge_schema), edge_batch)));
    let spec = int32_graph_spec(
        algorithm,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        GraphConstructionOptions { directed, renumber: true, policy: GraphConstructionPolicy::RawLibcugraph },
        params,
    );
    let graph = builder.add_graph_algorithm(edges, Vec::new(), spec).unwrap();
    let plan = builder.finish_with_host_sink_schema(graph, int32_stable_graph_result_schema(algorithm)).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "edges".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, edge_batch).unwrap()),
            Arc::clone(edge_schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize graph algorithm output");
    };
    let vertices = graph_vertex_values(batch, 0);
    let values = batch.column(1);
    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let bits = match values.data_type() {
            DataType::Int32 => u64::from(values.as_primitive::<Int32Type>().value(row) as u32),
            DataType::Int64 => values.as_primitive::<Int64Type>().value(row) as u64,
            DataType::Float32 => u64::from(values.as_primitive::<Float32Type>().value(row).to_bits()),
            DataType::Float64 => values.as_primitive::<Float64Type>().value(row).to_bits(),
            data_type => panic!("expected integer label or float score column, got {data_type}"),
        };
        rows.push((vertices.value(row), bits));
    }
    rows.sort_unstable();
    admitted.finish();
    rows
}

fn int32_edge_batch(pairs: &[(i32, i32)]) -> (Arc<Schema>, RecordBatch) {
    let edge_schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int32, false),
        Field::new("dst", DataType::Int32, false),
    ]));
    let (src_values, dst_values): (Vec<i32>, Vec<i32>) = pairs.iter().copied().unzip();
    let edge_batch = RecordBatch::try_new(
        Arc::clone(&edge_schema),
        vec![Arc::new(Int32Array::from(src_values)), Arc::new(Int32Array::from(dst_values))],
    )
    .unwrap();
    (edge_schema, edge_batch)
}

/// WCC labels must be the minimum vertex id of each component and identical
/// across runs.
///
/// Raw cuGraph assigns component labels in hook-race order (and WCC reports
/// them in the internal renumbered id space), so without canonicalization the
/// three runs disagree on nearly every row and no label equals a caller vertex
/// id. Vertex ids here are deliberately non-contiguous so an internal
/// renumber id cannot coincide with a caller id by accident.
#[test]
fn wcc_labels_are_min_vertex_ids_and_stable_across_runs() {
    let (edge_schema, edge_batch) = int32_edge_batch(&[(3, 7), (7, 9), (9, 3), (100, 200)]);
    let context = backend_context();
    let params = GraphAlgorithmParams::WeaklyConnectedComponents(GraphWccParams);
    let expected = vec![(3, 3), (7, 3), (9, 3), (100, 100), (200, 100)];
    for _ in 0..3 {
        let rows = run_vertex_keyed_algorithm(
            &context,
            GraphAlgorithm::WeaklyConnectedComponents,
            params.clone(),
            false,
            &edge_schema,
            &edge_batch,
        );
        let rows: Vec<(i64, i64)> = rows.into_iter().map(|(vertex, bits)| (vertex, bits as i64)).collect();
        assert_eq!(rows, expected, "WCC labels must be component minimum vertex ids");
    }
}

/// SCC labels must be the minimum vertex id of each component and identical
/// across runs, matching the WCC contract.
#[test]
fn scc_labels_are_min_vertex_ids_and_stable_across_runs() {
    // Two directed cycles joined by a one-way edge: {3,7,9} and {100,200} stay
    // separate strongly connected components.
    let (edge_schema, edge_batch) = int32_edge_batch(&[(3, 7), (7, 9), (9, 3), (100, 200), (200, 100), (9, 100)]);
    let context = backend_context();
    let params = GraphAlgorithmParams::StronglyConnectedComponents(GraphSccParams);
    let expected = vec![(3, 3), (7, 3), (9, 3), (100, 100), (200, 100)];
    for _ in 0..3 {
        let rows = run_vertex_keyed_algorithm(
            &context,
            GraphAlgorithm::StronglyConnectedComponents,
            params.clone(),
            true,
            &edge_schema,
            &edge_batch,
        );
        let rows: Vec<(i64, i64)> = rows.into_iter().map(|(vertex, bits)| (vertex, bits as i64)).collect();
        assert_eq!(rows, expected, "SCC labels must be component minimum vertex ids");
    }
}

/// Betweenness scores must be bitwise identical across runs on the same graph.
///
/// The multisource backward pass used to accumulate floating-point deltas with
/// relaxed atomics in schedule order, so ~9% of vertices drifted by f32 ulps
/// between runs while the nonzero set stayed fixed. Fixed-order accumulation
/// makes every score bit-stable; comparing raw bits (not values) is the point.
#[test]
fn betweenness_scores_are_bitwise_stable_across_runs() {
    // Hub 0 -> 8 middles -> 4 leaves each. Seeded at 0, every middle lies on
    // shortest paths and must score nonzero.
    let mut pairs = Vec::new();
    let mut leaf = 9i32;
    for middle in 1..=8i32 {
        pairs.push((0, middle));
        for _ in 0..4 {
            pairs.push((middle, leaf));
            leaf += 1;
        }
    }
    let (edge_schema, edge_batch) = int32_edge_batch(&pairs);
    let context = backend_context();
    let params = GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
        seeds: Some(GraphBetweennessSeeds::I64(vec![0])),
        ..Default::default()
    });
    let first = run_vertex_keyed_algorithm(
        &context,
        GraphAlgorithm::BetweennessCentrality,
        params.clone(),
        true,
        &edge_schema,
        &edge_batch,
    );
    assert_eq!(first.len(), 41, "every hub, middle, and leaf vertex must be reported");
    assert!(
        first.iter().filter(|(_, bits)| *bits != 0 && *bits != 0x8000_0000).count() >= 8,
        "the 8 middles must carry nonzero scores"
    );
    for _ in 0..2 {
        let rows = run_vertex_keyed_algorithm(
            &context,
            GraphAlgorithm::BetweennessCentrality,
            params.clone(),
            true,
            &edge_schema,
            &edge_batch,
        );
        assert_eq!(rows, first, "betweenness scores must be bitwise stable across runs");
    }
}

/// Betweenness scores must be bitwise identical across device budgets.
///
/// cuGraph sizes its multisource batches from the remaining query-domain
/// budget, so different grants split the same seed set into different batch
/// counts. Fixed source-order accumulation makes the total order the global
/// seed order in every case, so the scores; and their bits; do not depend on
/// the cap. Grants below ~6MiB cannot fit this query and fail loudly with
/// `QueryMemoryLimitExceeded`; that is the designed behavior and is not
/// exercised here.
#[test]
fn betweenness_scores_are_bitwise_stable_across_device_budgets() {
    // Hub 0 -> 200 middles -> 20 leaves each, seeded at 0..32. Enough seeds
    // that small grants split them into several batches while the default
    // grant processes them at once.
    let mut pairs = Vec::new();
    let mut leaf = 201i32;
    for middle in 1..=200i32 {
        pairs.push((0, middle));
        for _ in 0..20 {
            pairs.push((middle, leaf));
            leaf += 1;
        }
    }
    let (edge_schema, edge_batch) = int32_edge_batch(&pairs);
    let params = GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
        seeds: Some(GraphBetweennessSeeds::I64((0..32).collect())),
        ..Default::default()
    });
    let mut reference: Option<Vec<(i64, u64)>> = None;
    for grant in [Some(6 * 1024 * 1024), Some(32 * 1024 * 1024), None] {
        let context = backend_context_with_grant(grant);
        let rows = run_vertex_keyed_algorithm(
            &context,
            GraphAlgorithm::BetweennessCentrality,
            params.clone(),
            true,
            &edge_schema,
            &edge_batch,
        );
        assert_eq!(rows.len(), 4201, "every hub, middle, and leaf vertex must be reported");
        match &reference {
            None => reference = Some(rows),
            Some(expected) => assert_eq!(&rows, expected, "betweenness bits must not depend on the grant"),
        }
    }
}
