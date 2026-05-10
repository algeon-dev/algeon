static GRAPH_SEED_CHUNK_RETRY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn assert_vertex_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = vertex_betweenness_scores(expected);
    let actual = vertex_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (vertex, expected_value) in expected {
        let actual_value = actual.get(&vertex).unwrap_or_else(|| panic!("missing vertex {vertex}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "vertex {vertex}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn vertex_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<i64, f64> {
    let vertices = int64_array(batch, 0);
    let values = float64_array(batch, 1);
    (0..batch.num_rows()).map(|row| (vertices.value(row), values.value(row))).collect()
}

fn assert_edge_betweenness_batches_close(expected: &RecordBatch, actual: &RecordBatch) {
    let expected = edge_betweenness_scores(expected);
    let actual = edge_betweenness_scores(actual);
    assert_eq!(actual.len(), expected.len());
    for (edge, expected_value) in expected {
        let actual_value = actual.get(&edge).unwrap_or_else(|| panic!("missing edge {edge:?}"));
        assert!(
            (actual_value - expected_value).abs() <= 1.0e-6,
            "edge {edge:?}: expected {expected_value}, got {actual_value}"
        );
    }
}

fn edge_betweenness_scores(batch: &RecordBatch) -> std::collections::BTreeMap<(i64, i64), f64> {
    let sources = int64_array(batch, 0);
    let destinations = int64_array(batch, 1);
    let scores = float64_array(batch, 2);
    (0..batch.num_rows()).map(|row| ((sources.value(row), destinations.value(row)), scores.value(row))).collect()
}
