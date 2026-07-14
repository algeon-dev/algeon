//! Arrow-to-oracle comparators for the registry-driven cuGraph parity harness.

use std::collections::{BTreeMap, BTreeSet};

use arrow::{
    array::{Array, Float32Array, Float64Array, Int32Array, Int64Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use datafusion_nexus::{
    CugraphAlgorithm,
    graph_sql::parity::{ParityComparator, ParitySpec, ParityTruthSource},
};
use serde_json::Value;

use super::datasets::ParityEdge;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cell {
    Null,
    Int(i64),
    Float(f64),
}

impl Cell {
    fn integer(self, column: &str, row: usize) -> Result<i64, String> {
        match self {
            Self::Int(value) => Ok(value),
            Self::Null => Err(format!("column '{column}' row {row} is null, expected an integer")),
            Self::Float(value) => Err(format!("column '{column}' row {row} is {value}, expected an integer")),
        }
    }

    fn number(self, column: &str, row: usize) -> Result<f64, String> {
        let value = match self {
            Self::Int(value) => value as f64,
            Self::Float(value) => value,
            Self::Null => return Err(format!("column '{column}' row {row} is null, expected a number")),
        };
        if !value.is_finite() {
            return Err(format!("column '{column}' row {row} is non-finite: {value}"));
        }
        Ok(value)
    }
}

#[derive(Debug)]
struct Table {
    columns: BTreeMap<String, Vec<Cell>>,
    rows: usize,
}

impl Table {
    fn from_batches(batches: &[RecordBatch]) -> Result<Self, String> {
        let schema = batches.first().ok_or_else(|| "GPU execution returned no record batches".to_string())?.schema();
        let mut columns =
            schema.fields().iter().map(|field| (field.name().clone(), Vec::new())).collect::<BTreeMap<_, _>>();
        let mut rows = 0;
        for batch in batches {
            if batch.schema() != schema {
                return Err("GPU execution returned record batches with different schemas".to_string());
            }
            rows += batch.num_rows();
            for (index, field) in schema.fields().iter().enumerate() {
                let output = columns.get_mut(field.name()).expect("schema column was initialized");
                append_array(output, batch.column(index).as_ref(), field.data_type())?;
            }
        }
        Ok(Self { columns, rows })
    }

    fn column(&self, name: &str) -> Result<&[Cell], String> {
        self.columns
            .get(name)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("GPU result is missing column '{name}'; columns={:?}", self.columns.keys()))
    }

    fn integer_column(&self, name: &str) -> Result<Vec<i64>, String> {
        self.column(name)?.iter().enumerate().map(|(row, value)| value.integer(name, row)).collect()
    }
}

fn append_array(output: &mut Vec<Cell>, array: &dyn Array, data_type: &DataType) -> Result<(), String> {
    macro_rules! append_primitive {
        ($array_type:ty, $variant:ident) => {{
            let values = array
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or_else(|| format!("Arrow array does not match declared dtype {data_type:?}"))?;
            output
                .extend((0..values.len()).map(|row| {
                    if values.is_null(row) { Cell::Null } else { Cell::$variant(values.value(row).into()) }
                }));
        }};
    }
    match data_type {
        DataType::Int32 => append_primitive!(Int32Array, Int),
        DataType::Int64 => append_primitive!(Int64Array, Int),
        DataType::Float32 => append_primitive!(Float32Array, Float),
        DataType::Float64 => append_primitive!(Float64Array, Float),
        other => return Err(format!("cuGraph parity comparator does not support Arrow dtype {other:?}")),
    }
    Ok(())
}

fn oracle_table(document: &Value, actual_columns: impl Iterator<Item = String>) -> Result<Table, String> {
    let result = document
        .get("result")
        .and_then(Value::as_object)
        .ok_or_else(|| "oracle JSON is missing a result object".to_string())?;
    let mut columns = BTreeMap::new();
    let mut rows = None;
    for name in actual_columns {
        let values = result
            .get(&name)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("oracle result is missing array column '{name}'"))?;
        if let Some(expected) = rows {
            if values.len() != expected {
                return Err(format!("oracle column '{name}' has {} rows, expected {expected}", values.len()));
            }
        } else {
            rows = Some(values.len());
        }
        columns.insert(name.clone(), values.iter().map(json_cell).collect::<Result<Vec<_>, _>>()?);
    }
    Ok(Table { columns, rows: rows.unwrap_or(0) })
}

fn json_cell(value: &Value) -> Result<Cell, String> {
    match value {
        Value::Null => Ok(Cell::Null),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            number.as_i64().map(Cell::Int).ok_or_else(|| format!("oracle integer {number} does not fit i64"))
        },
        Value::Number(number) => number
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Cell::Float)
            .ok_or_else(|| format!("oracle number {number} is not a finite f64")),
        other => Err(format!("oracle cell must be numeric or null, got {other}")),
    }
}

pub fn compare(
    spec: &ParitySpec,
    batches: &[RecordBatch],
    oracle_document: &Value,
    reference_vertices: Option<&[i64]>,
    edges: &[ParityEdge],
) -> Result<(), String> {
    let actual = Table::from_batches(batches)?;
    if spec.comparator == ParityComparator::TopKVertices {
        let expected =
            reference_vertices.ok_or_else(|| "top-K comparator requires a reference vertex list".to_string())?;
        return compare_top_k(&actual, expected);
    }
    if matches!(spec.truth_source, ParityTruthSource::RefCsv { .. }) {
        return Err("non-top-K RefCsv comparator is not implemented".to_string());
    }
    let expected = oracle_table(oracle_document, actual.columns.keys().cloned())?;
    if spec.comparator != ParityComparator::EdgeSubgraph && actual.rows != expected.rows {
        return Err(format!("row-count mismatch: GPU={} oracle={}", actual.rows, expected.rows));
    }
    match spec.comparator {
        ParityComparator::ExactInt if spec.algorithm == CugraphAlgorithm::Bfs => compare_bfs(&actual, &expected, edges),
        ParityComparator::ExactInt => compare_exact(&actual, &expected),
        ParityComparator::Numeric if spec.algorithm == CugraphAlgorithm::Sssp => {
            compare_sssp(&actual, &expected, edges, spec.tolerance.abs, spec.tolerance.rel)
        },
        ParityComparator::Numeric if spec.algorithm == CugraphAlgorithm::ForceAtlas2 => {
            compare_force_atlas2(&actual, &expected, spec.tolerance.abs, spec.tolerance.rel)
        },
        ParityComparator::Numeric => compare_numeric(&actual, &expected, spec.tolerance.abs, spec.tolerance.rel),
        ParityComparator::PartitionEquivalence => compare_partitions(&actual, &expected),
        ParityComparator::PartitionQuality if spec.algorithm == CugraphAlgorithm::Ecg => {
            compare_ecg_partition_quality(&actual, &expected, edges, spec)
        },
        ParityComparator::EdgeSubgraph => compare_edges(&actual, &expected),
        ParityComparator::TopKVertices => unreachable!("handled above"),
        ParityComparator::PartitionQuality | ParityComparator::SeededSequence | ParityComparator::RankCorrelation => {
            Err(format!("comparator family '{}' has no registry rows yet", spec.comparator.as_str()))
        },
    }
}

fn compare_bfs(actual: &Table, expected: &Table, edges: &[ParityEdge]) -> Result<(), String> {
    compare_named_exact(actual, expected, &["vertex", "distance"])?;
    let vertices = actual.integer_column("vertex")?;
    let distances = actual.integer_column("distance")?;
    let predecessors = actual.column("predecessor")?;
    let expected_predecessors = expected.column("predecessor")?;
    let distance_by_vertex = vertices.iter().copied().zip(distances.iter().copied()).collect::<BTreeMap<_, _>>();
    let edge_set = edges.iter().map(|edge| (edge.source, edge.destination)).collect::<BTreeSet<_>>();
    for (row, ((vertex, distance), predecessor)) in vertices.iter().zip(&distances).zip(predecessors).enumerate() {
        let predecessor = predecessor_vertex(*predecessor, row)?;
        let expected_predecessor = predecessor_vertex(expected_predecessors[row], row)?;
        if *distance == 0 {
            if predecessor.is_some() {
                return Err(format!(
                    "BFS vertex {vertex} at distance {distance} must not have predecessor {predecessor:?}"
                ));
            }
            continue;
        }
        if predecessor.is_none() && expected_predecessor.is_none() {
            // The oracle marks unreachable vertices with no predecessor. Its
            // exact distance sentinel has already matched above.
            continue;
        }
        let predecessor = predecessor
            .ok_or_else(|| format!("BFS vertex {vertex} at positive distance {distance} is missing its predecessor"))?;
        let predecessor_distance = distance_by_vertex
            .get(&predecessor)
            .ok_or_else(|| format!("BFS vertex {vertex} refers to absent predecessor {predecessor}"))?;
        if predecessor_distance.checked_add(1) != Some(*distance) {
            return Err(format!(
                "BFS predecessor distance mismatch for vertex {vertex}: predecessor {predecessor} has distance {predecessor_distance}, vertex distance is {distance}"
            ));
        }
        if !edge_set.contains(&(predecessor, *vertex)) {
            return Err(format!("BFS predecessor edge {predecessor}->{vertex} is absent from the input graph"));
        }
    }
    Ok(())
}

fn compare_sssp(actual: &Table, expected: &Table, edges: &[ParityEdge], abs: f64, rel: f64) -> Result<(), String> {
    compare_named_numeric(actual, expected, &["vertex", "distance"], abs, rel)?;
    let vertices = actual.integer_column("vertex")?;
    let distances = actual
        .column("distance")?
        .iter()
        .enumerate()
        .map(|(row, value)| value.number("distance", row))
        .collect::<Result<Vec<_>, _>>()?;
    let predecessors = actual.column("predecessor")?;
    let expected_predecessors = expected.column("predecessor")?;
    let distance_by_vertex = vertices.iter().copied().zip(distances.iter().copied()).collect::<BTreeMap<_, _>>();
    let mut weights_by_edge = BTreeMap::<(i64, i64), Vec<f64>>::new();
    for edge in edges {
        weights_by_edge.entry((edge.source, edge.destination)).or_default().push(edge.weight);
    }
    for (row, ((vertex, distance), predecessor)) in vertices.iter().zip(&distances).zip(predecessors).enumerate() {
        let predecessor = predecessor_vertex(*predecessor, row)?;
        let expected_predecessor = predecessor_vertex(expected_predecessors[row], row)?;
        if *distance == 0.0 {
            if predecessor.is_some() {
                return Err(format!(
                    "SSSP vertex {vertex} at distance {distance} must not have predecessor {predecessor:?}"
                ));
            }
            continue;
        }
        if predecessor.is_none() && expected_predecessor.is_none() {
            // Both implementations identify this vertex as unreachable; the
            // numeric distance sentinel has already matched above.
            continue;
        }
        let predecessor = predecessor.ok_or_else(|| {
            format!("SSSP vertex {vertex} at finite positive distance {distance} is missing its predecessor")
        })?;
        let predecessor_distance = distance_by_vertex
            .get(&predecessor)
            .ok_or_else(|| format!("SSSP vertex {vertex} refers to absent predecessor {predecessor}"))?;
        let weights = weights_by_edge
            .get(&(predecessor, *vertex))
            .ok_or_else(|| format!("SSSP predecessor edge {predecessor}->{vertex} is absent from the input graph"))?;
        if !weights.iter().any(|weight| numbers_close(predecessor_distance + weight, *distance, abs, rel)) {
            return Err(format!(
                "SSSP predecessor distance mismatch for vertex {vertex}: predecessor {predecessor} has distance {predecessor_distance}, edge weights={weights:?}, vertex distance={distance}"
            ));
        }
    }
    Ok(())
}

fn predecessor_vertex(value: Cell, row: usize) -> Result<Option<i64>, String> {
    match value {
        Cell::Null => Ok(None),
        Cell::Int(value) if value < 0 => Ok(None),
        Cell::Int(value) => Ok(Some(value)),
        Cell::Float(value) => Err(format!("predecessor row {row} has floating-point value {value}")),
    }
}

fn compare_named_exact(actual: &Table, expected: &Table, names: &[&str]) -> Result<(), String> {
    for name in names {
        let actual_values = actual.column(name)?;
        let expected_values = expected.column(name)?;
        for (row, (actual, expected)) in actual_values.iter().zip(expected_values).enumerate() {
            if !cells_match(name, *actual, *expected) {
                return Err(format!("column '{name}' row {row} mismatch: GPU={actual:?} oracle={expected:?}"));
            }
        }
    }
    Ok(())
}

fn compare_named_numeric(actual: &Table, expected: &Table, names: &[&str], abs: f64, rel: f64) -> Result<(), String> {
    for name in names {
        compare_numeric_column(name, actual.column(name)?, expected.column(name)?, abs, rel)?;
    }
    Ok(())
}

fn compare_exact(actual: &Table, expected: &Table) -> Result<(), String> {
    for (name, actual_values) in &actual.columns {
        let expected_values = expected.column(name)?;
        for (row, (actual, expected)) in actual_values.iter().zip(expected_values).enumerate() {
            if !cells_match(name, *actual, *expected) {
                return Err(format!("column '{name}' row {row} mismatch: GPU={actual:?} oracle={expected:?}"));
            }
        }
    }
    Ok(())
}

fn compare_numeric(actual: &Table, expected: &Table, abs: f64, rel: f64) -> Result<(), String> {
    for (name, actual_values) in &actual.columns {
        let expected_values = expected.column(name)?;
        compare_numeric_column(name, actual_values, expected_values, abs, rel)?;
    }
    Ok(())
}

fn compare_numeric_column(
    name: &str,
    actual_values: &[Cell],
    expected_values: &[Cell],
    abs: f64,
    rel: f64,
) -> Result<(), String> {
    for (row, (actual, expected)) in actual_values.iter().zip(expected_values).enumerate() {
        match (*actual, *expected) {
            _ if cells_match(name, *actual, *expected) => {},
            (Cell::Int(left), Cell::Int(right)) => {
                return Err(format!("column '{name}' row {row} integer mismatch: GPU={left} oracle={right}"));
            },
            (Cell::Int(left), right) => {
                return Err(format!(
                    "column '{name}' row {row} oracle type mismatch: GPU integer={left} oracle={right:?}"
                ));
            },
            (Cell::Null, _) | (_, Cell::Null) => {
                return Err(format!("column '{name}' row {row} null mismatch: GPU={actual:?} oracle={expected:?}"));
            },
            _ => {
                let left = actual.number(name, row)?;
                let right = expected.number(name, row)?;
                let delta = (left - right).abs();
                let limit = abs.max(rel * left.abs().max(right.abs()));
                if delta > limit {
                    return Err(format!(
                        "column '{name}' row {row} mismatch: GPU={left} oracle={right} delta={delta} tolerance={limit}"
                    ));
                }
            },
        }
    }
    Ok(())
}

fn numbers_close(left: f64, right: f64, abs: f64, rel: f64) -> bool {
    let delta = (left - right).abs();
    delta <= abs.max(rel * left.abs().max(right.abs()))
}

fn compare_force_atlas2(actual: &Table, expected: &Table, abs: f64, rel: f64) -> Result<(), String> {
    let actual_vertices = actual.integer_column("vertex")?;
    let expected_vertices = expected.integer_column("vertex")?;
    if actual_vertices != expected_vertices {
        return Err(format!("layout vertex mismatch: GPU={actual_vertices:?} oracle={expected_vertices:?}"));
    }
    let points = |table: &Table| -> Result<Vec<(f64, f64)>, String> {
        let x = table.column("x")?;
        let y = table.column("y")?;
        x.iter().zip(y).enumerate().map(|(row, (x, y))| Ok((x.number("x", row)?, y.number("y", row)?))).collect()
    };
    let actual_points = points(actual)?;
    let expected_points = points(expected)?;
    if actual_points.is_empty() {
        return Ok(());
    }

    let actual_center = centroid(&actual_points);
    let expected_center = centroid(&expected_points);
    let actual_centered = center_points(&actual_points, actual_center);
    let expected_centered = center_points(&expected_points, expected_center);

    // Two-dimensional orthogonal Procrustes has closed forms for both
    // rotation and reflection. Layout coordinates have no meaningful global
    // origin, orientation, or handedness, so choose the lower-residual rigid
    // transform before comparing point displacement.
    let rotation_a = actual_centered.iter().zip(&expected_centered).map(|((x, y), (u, v))| x * u + y * v).sum::<f64>();
    let rotation_b = actual_centered.iter().zip(&expected_centered).map(|((x, y), (u, v))| x * v - y * u).sum::<f64>();
    let rotation = transform_points(&actual_centered, expected_center, rotation_a, rotation_b, false)?;

    let reflection_a =
        actual_centered.iter().zip(&expected_centered).map(|((x, y), (u, v))| x * u - y * v).sum::<f64>();
    let reflection_b =
        actual_centered.iter().zip(&expected_centered).map(|((x, y), (u, v))| y * u + x * v).sum::<f64>();
    let reflection = transform_points(&actual_centered, expected_center, reflection_a, reflection_b, true)?;
    let residual = |aligned: &[(f64, f64)]| {
        aligned.iter().zip(&expected_points).map(|((x, y), (u, v))| (x - u).hypot(y - v)).collect::<Vec<_>>()
    };
    let rotation_residual = residual(&rotation);
    let reflection_residual = residual(&reflection);
    let residuals = if squared_sum(&rotation_residual) <= squared_sum(&reflection_residual) {
        rotation_residual
    } else {
        reflection_residual
    };
    let rms_radius =
        (expected_centered.iter().map(|(x, y)| x * x + y * y).sum::<f64>() / expected_centered.len() as f64).sqrt();
    let limit = abs.max(rel * rms_radius);
    let (worst_row, worst) = residuals
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("non-empty layout has a residual");
    if worst > limit {
        return Err(format!(
            "rigid-aligned layout mismatch at vertex {}: displacement={worst} tolerance={limit} rms_radius={rms_radius}",
            actual_vertices[worst_row]
        ));
    }
    Ok(())
}

fn centroid(points: &[(f64, f64)]) -> (f64, f64) {
    let (x, y) = points.iter().fold((0.0, 0.0), |(sum_x, sum_y), (x, y)| (sum_x + x, sum_y + y));
    (x / points.len() as f64, y / points.len() as f64)
}

fn center_points(points: &[(f64, f64)], center: (f64, f64)) -> Vec<(f64, f64)> {
    points.iter().map(|(x, y)| (x - center.0, y - center.1)).collect()
}

fn transform_points(
    points: &[(f64, f64)],
    target_center: (f64, f64),
    a: f64,
    b: f64,
    reflect: bool,
) -> Result<Vec<(f64, f64)>, String> {
    let norm = a.hypot(b);
    if norm <= f64::EPSILON {
        return Err("layout rigid alignment is degenerate".to_string());
    }
    let (cos, sin) = (a / norm, b / norm);
    Ok(points
        .iter()
        .map(|(x, y)| {
            let (x, y) =
                if reflect { (cos * x + sin * y, sin * x - cos * y) } else { (cos * x - sin * y, sin * x + cos * y) };
            (x + target_center.0, y + target_center.1)
        })
        .collect())
}

fn squared_sum(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum()
}

fn cells_match(column: &str, actual: Cell, expected: Cell) -> bool {
    actual == expected
        || (column == "predecessor" && matches!((actual, expected), (Cell::Int(value), Cell::Null) if value < 0))
}

fn compare_partitions(actual: &Table, expected: &Table) -> Result<(), String> {
    let actual_groups = partition_groups(actual)?;
    let expected_groups = partition_groups(expected)?;
    if actual_groups != expected_groups {
        return Err(format!("partition mismatch: GPU={actual_groups:?} oracle={expected_groups:?}"));
    }
    Ok(())
}

/// Verify that repeated runs from the same loaded binary and seed produce the
/// same partition. Community label IDs themselves are not part of the
/// contract, so this compares sets of vertices per community.
pub fn compare_repeated_partitions(
    initial_batches: &[RecordBatch],
    replay_batches: &[RecordBatch],
) -> Result<(), String> {
    let initial = Table::from_batches(initial_batches)?;
    let replay = Table::from_batches(replay_batches)?;
    let initial_groups = partition_groups(&initial)?;
    let replay_groups = partition_groups(&replay)?;
    if initial_groups != replay_groups {
        return Err(format!("partition changed between runs: initial={initial_groups:?} replay={replay_groups:?}"));
    }
    Ok(())
}

/// ECG uses randomized Louvain internally.  Exact partition equality with a
/// Python oracle is therefore not a portable correctness condition: a CUDA or
/// cuGraph build may choose an equally valid partition despite the same seed.
///
/// The oracle remains useful as a quality baseline. Require the exact vertex
/// set, permit at most one community either side of its count, and require
/// modularity on the same Python-compatible undirected normalized edge list to
/// remain within tolerance below the oracle baseline. A higher modularity is
/// an improvement, not a parity failure.
fn compare_ecg_partition_quality(
    actual: &Table,
    expected: &Table,
    edges: &[ParityEdge],
    spec: &ParitySpec,
) -> Result<(), String> {
    const COMMUNITY_COUNT_DELTA: usize = 1;

    let actual_assignments = partition_assignments(actual)?;
    let expected_assignments = partition_assignments(expected)?;
    let actual_vertices = actual_assignments.keys().copied().collect::<BTreeSet<_>>();
    let expected_vertices = expected_assignments.keys().copied().collect::<BTreeSet<_>>();
    let normalized_edges = python_undirected_edges(edges)?;
    let graph_vertices =
        normalized_edges.iter().flat_map(|edge| [edge.source, edge.destination]).collect::<BTreeSet<_>>();
    if actual_vertices != graph_vertices {
        return Err(format!("ECG vertex coverage mismatch: GPU={actual_vertices:?} graph={graph_vertices:?}"));
    }
    if expected_vertices != graph_vertices {
        return Err(format!(
            "ECG oracle vertex coverage mismatch: oracle={expected_vertices:?} graph={graph_vertices:?}"
        ));
    }

    let actual_communities = actual_assignments.values().copied().collect::<BTreeSet<_>>().len();
    let expected_communities = expected_assignments.values().copied().collect::<BTreeSet<_>>().len();
    let minimum_communities = expected_communities.saturating_sub(COMMUNITY_COUNT_DELTA).max(1);
    let maximum_communities = expected_communities.saturating_add(COMMUNITY_COUNT_DELTA);
    if !(minimum_communities..=maximum_communities).contains(&actual_communities) {
        return Err(format!(
            "ECG community-count mismatch: GPU={actual_communities}, oracle={expected_communities}, accepted={minimum_communities}..={maximum_communities}"
        ));
    }

    let resolution = ecg_resolution(spec)?;
    let actual_modularity = partition_modularity(&actual_assignments, &normalized_edges, resolution)?;
    let expected_modularity = partition_modularity(&expected_assignments, &normalized_edges, resolution)?;
    let tolerance = spec.tolerance.abs.max(spec.tolerance.rel * expected_modularity.abs());
    let minimum_modularity = expected_modularity - tolerance;
    if actual_modularity < minimum_modularity {
        return Err(format!(
            "ECG modularity below oracle baseline: GPU={actual_modularity}, oracle={expected_modularity}, minimum={minimum_modularity}, tolerance={tolerance}"
        ));
    }
    Ok(())
}

fn partition_assignments(table: &Table) -> Result<BTreeMap<i64, i64>, String> {
    let vertices = table.integer_column("vertex")?;
    let label_name = if table.columns.contains_key("partition") { "partition" } else { "label" };
    let labels = table.integer_column(label_name)?;
    if vertices.len() != labels.len() {
        return Err(format!(
            "partition result has mismatched vertex/label lengths: vertices={} labels={}",
            vertices.len(),
            labels.len()
        ));
    }
    let mut assignments = BTreeMap::new();
    for (row, (vertex, label)) in vertices.into_iter().zip(labels).enumerate() {
        if assignments.insert(vertex, label).is_some() {
            return Err(format!("partition result has duplicate vertex {vertex} at row {row}"));
        }
    }
    Ok(assignments)
}

fn partition_groups(table: &Table) -> Result<BTreeSet<BTreeSet<i64>>, String> {
    let mut groups = BTreeMap::<i64, BTreeSet<i64>>::new();
    for (vertex, label) in partition_assignments(table)? {
        groups.entry(label).or_default().insert(vertex);
    }
    Ok(groups.into_values().collect())
}

fn ecg_resolution(spec: &ParitySpec) -> Result<f64, String> {
    let params = serde_json::from_str::<Value>(spec.params_json)
        .map_err(|error| format!("ECG parity spec has invalid params JSON: {error}"))?;
    let resolution = params
        .get("resolution")
        .and_then(Value::as_f64)
        .ok_or_else(|| "ECG parity spec is missing a numeric resolution".to_string())?;
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(format!("ECG parity spec resolution must be finite and positive, got {resolution}"));
    }
    Ok(resolution)
}

/// Mirror the undirected Python-compatible edge normalization used by ECG:
/// add reverse arcs, then retain the minimum finite weight for every directed
/// `(src, dst)` pair. The committed karate fixture is already symmetric, but
/// doing this explicitly keeps the metric coupled to execution semantics.
fn python_undirected_edges(edges: &[ParityEdge]) -> Result<Vec<ParityEdge>, String> {
    let mut normalized = BTreeMap::<(i64, i64), f64>::new();
    for edge in edges {
        if !edge.weight.is_finite() {
            return Err(format!(
                "ECG input edge {}->{} has non-finite weight {}",
                edge.source, edge.destination, edge.weight
            ));
        }
        for (source, destination) in [(edge.source, edge.destination), (edge.destination, edge.source)] {
            normalized
                .entry((source, destination))
                .and_modify(|weight| *weight = weight.min(edge.weight))
                .or_insert(edge.weight);
        }
    }
    Ok(normalized
        .into_iter()
        .map(|((source, destination), weight)| ParityEdge { source, destination, weight })
        .collect())
}

/// Match libcugraph ECG's final modularity calculation: sum internal edge
/// weights divided by total oriented weight, minus resolution times the sum of
/// squared source-community weights divided by total weight squared.
fn partition_modularity(
    assignments: &BTreeMap<i64, i64>,
    edges: &[ParityEdge],
    resolution: f64,
) -> Result<f64, String> {
    let mut total_weight = 0.0;
    let mut internal_weight = 0.0;
    let mut community_weights = BTreeMap::<i64, f64>::new();
    for edge in edges {
        let source_community = assignments
            .get(&edge.source)
            .ok_or_else(|| format!("ECG partition is missing source vertex {}", edge.source))?;
        let destination_community = assignments
            .get(&edge.destination)
            .ok_or_else(|| format!("ECG partition is missing destination vertex {}", edge.destination))?;
        total_weight += edge.weight;
        *community_weights.entry(*source_community).or_default() += edge.weight;
        if source_community == destination_community {
            internal_weight += edge.weight;
        }
    }
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return Err(format!("ECG modularity requires positive finite total edge weight, got {total_weight}"));
    }
    let sum_degree_squared = community_weights.values().map(|weight| weight * weight).sum::<f64>();
    let modularity = internal_weight / total_weight - resolution * sum_degree_squared / (total_weight * total_weight);
    if !modularity.is_finite() {
        return Err(format!("ECG modularity is non-finite: {modularity}"));
    }
    Ok(modularity)
}

fn compare_edges(actual: &Table, expected: &Table) -> Result<(), String> {
    let canonicalize = |table: &Table| -> Result<BTreeSet<(i64, i64)>, String> {
        let sources = table.integer_column("src")?;
        let destinations = table.integer_column("dst")?;
        let mut orientations = BTreeMap::<(i64, i64), BTreeSet<(i64, i64)>>::new();
        for (row, (source, destination)) in sources.into_iter().zip(destinations).enumerate() {
            let canonical = (source.min(destination), source.max(destination));
            let inserted = orientations.entry(canonical).or_default().insert((source, destination));
            if !inserted {
                return Err(format!(
                    "edge result contains duplicate directed row {source}->{destination} at row {row}"
                ));
            }
        }
        Ok(orientations.into_keys().collect())
    };
    let actual_edges = canonicalize(actual)?;
    let expected_edges = canonicalize(expected)?;
    if actual_edges != expected_edges {
        return Err(format!("edge-set mismatch: GPU={actual_edges:?} oracle={expected_edges:?}"));
    }
    Ok(())
}

fn compare_top_k(actual: &Table, expected_vertices: &[i64]) -> Result<(), String> {
    let vertices = actual.integer_column("vertex")?;
    let values = actual.column("value")?;
    let mut ranked = vertices
        .into_iter()
        .zip(values.iter().enumerate().map(|(row, value)| value.number("value", row)))
        .map(|(vertex, value)| value.map(|value| (vertex, value)))
        .collect::<Result<Vec<_>, _>>()?;
    ranked.sort_by(|(left_vertex, left), (right_vertex, right)| {
        right.total_cmp(left).then_with(|| left_vertex.cmp(right_vertex))
    });
    let actual_vertices =
        ranked.into_iter().take(expected_vertices.len()).map(|(vertex, _)| vertex).collect::<BTreeSet<_>>();
    let expected_vertices = expected_vertices.iter().copied().collect::<BTreeSet<_>>();
    if actual_vertices != expected_vertices {
        return Err(format!("top-K vertex mismatch: GPU={actual_vertices:?} reference={expected_vertices:?}"));
    }
    Ok(())
}

pub fn family_label(comparator: ParityComparator) -> &'static str {
    comparator.as_str()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{ArrayRef, Int64Array},
        datatypes::{Field, Schema},
    };
    use datafusion_nexus::graph_sql::parity::ParityTolerance;

    use super::*;

    fn partition_table(vertices: Vec<i64>, labels: Vec<i64>, label_name: &str) -> Table {
        Table {
            rows: vertices.len(),
            columns: BTreeMap::from([
                ("vertex".to_string(), vertices.into_iter().map(Cell::Int).collect()),
                (label_name.to_string(), labels.into_iter().map(Cell::Int).collect()),
            ]),
        }
    }

    fn layout_table(vertices: Vec<i64>, points: Vec<(f64, f64)>) -> Table {
        Table {
            rows: vertices.len(),
            columns: BTreeMap::from([
                ("vertex".to_string(), vertices.into_iter().map(Cell::Int).collect()),
                ("x".to_string(), points.iter().map(|(x, _)| Cell::Float(*x)).collect()),
                ("y".to_string(), points.iter().map(|(_, y)| Cell::Float(*y)).collect()),
            ]),
        }
    }

    fn ecg_quality_spec(abs_tolerance: f64) -> ParitySpec {
        ParitySpec {
            label: "ecg/test",
            algorithm: CugraphAlgorithm::Ecg,
            dataset: "test",
            params_json: r#"{"resolution":1.0}"#,
            seed: Some(0),
            truth_source: ParityTruthSource::OracleJson,
            oracle_relpath: "test.json",
            comparator: ParityComparator::PartitionQuality,
            tolerance: ParityTolerance { abs: abs_tolerance, rel: 0.0 },
        }
    }

    fn partition_batches(vertices: Vec<i64>, labels: Vec<i64>) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Int64, false),
            Field::new("partition", DataType::Int64, false),
        ]));
        vec![
            RecordBatch::try_new(
                schema,
                vec![Arc::new(Int64Array::from(vertices)) as ArrayRef, Arc::new(Int64Array::from(labels)) as ArrayRef],
            )
            .unwrap(),
        ]
    }

    #[test]
    fn partition_comparison_ignores_label_renaming() {
        let actual = partition_table(vec![0, 1, 2, 3], vec![7, 7, 9, 9], "partition");
        let expected = partition_table(vec![0, 1, 2, 3], vec![2, 2, 1, 1], "partition");
        compare_partitions(&actual, &expected).unwrap();
    }

    #[test]
    fn ecg_quality_comparison_accepts_different_partition_at_the_same_quality() {
        // The two ways to split a four-cycle both have modularity zero. This
        // proves the ECG comparator deliberately accepts an equally good
        // partition rather than treating the oracle's labels as a golden
        // cross-build result.
        let expected = partition_table(vec![0, 1, 2, 3], vec![0, 0, 1, 1], "partition");
        let actual = partition_table(vec![0, 1, 2, 3], vec![0, 1, 1, 0], "partition");
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 1, destination: 2, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
            ParityEdge { source: 3, destination: 0, weight: 1.0 },
        ];
        compare_ecg_partition_quality(&actual, &expected, &edges, &ecg_quality_spec(0.0)).unwrap();
    }

    #[test]
    fn ecg_quality_comparison_rejects_low_modularity_partition() {
        let expected = partition_table(vec![0, 1, 2, 3], vec![0, 0, 1, 1], "partition");
        let actual = partition_table(vec![0, 1, 2, 3], vec![0, 0, 0, 0], "partition");
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
        ];
        let error = compare_ecg_partition_quality(&actual, &expected, &edges, &ecg_quality_spec(0.1)).unwrap_err();
        assert!(error.contains("modularity below oracle baseline"), "unexpected error: {error}");
    }

    #[test]
    fn ecg_quality_comparison_accepts_higher_modularity_than_the_oracle() {
        let expected = partition_table(vec![0, 1, 2, 3], vec![0, 0, 0, 0], "partition");
        let actual = partition_table(vec![0, 1, 2, 3], vec![0, 0, 1, 1], "partition");
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
        ];
        compare_ecg_partition_quality(&actual, &expected, &edges, &ecg_quality_spec(0.0)).unwrap();
    }

    #[test]
    fn ecg_quality_comparison_rejects_unreasonable_community_count() {
        let expected = partition_table(vec![0, 1, 2, 3], vec![0, 0, 1, 1], "partition");
        let actual = partition_table(vec![0, 1, 2, 3], vec![0, 1, 2, 3], "partition");
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
        ];
        let error = compare_ecg_partition_quality(&actual, &expected, &edges, &ecg_quality_spec(1.0)).unwrap_err();
        assert!(error.contains("community-count mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn ecg_quality_comparison_rejects_incomplete_vertex_coverage() {
        let expected = partition_table(vec![0, 1, 2, 3], vec![0, 0, 1, 1], "partition");
        let actual = partition_table(vec![0, 1, 2], vec![0, 0, 1], "partition");
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
        ];
        let error = compare_ecg_partition_quality(&actual, &expected, &edges, &ecg_quality_spec(1.0)).unwrap_err();
        assert!(error.contains("vertex coverage mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn repeated_partition_comparison_ignores_label_renaming_but_rejects_group_changes() {
        let initial = partition_batches(vec![0, 1, 2, 3], vec![7, 7, 9, 9]);
        let relabeled = partition_batches(vec![0, 1, 2, 3], vec![2, 2, 1, 1]);
        compare_repeated_partitions(&initial, &relabeled).unwrap();

        let changed = partition_batches(vec![0, 1, 2, 3], vec![2, 1, 1, 2]);
        let error = compare_repeated_partitions(&initial, &changed).unwrap_err();
        assert!(error.contains("partition changed between runs"), "unexpected error: {error}");
    }

    #[test]
    fn batch_conversion_preserves_null_integer_cells() {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(1), None]))]).unwrap();
        let table = Table::from_batches(&[batch]).unwrap();
        assert_eq!(table.column("vertex").unwrap(), [Cell::Int(1), Cell::Null]);
    }

    #[test]
    fn force_atlas_comparison_removes_translation_rotation_and_reflection() {
        let expected_points = vec![(0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 3.0)];
        let actual_points = expected_points.iter().map(|(x, y)| (5.0 - y, -3.0 - x)).collect();
        let vertices = vec![0, 1, 2, 3];
        compare_force_atlas2(
            &layout_table(vertices.clone(), actual_points),
            &layout_table(vertices, expected_points),
            1.0e-12,
            1.0e-12,
        )
        .unwrap();
    }

    #[test]
    fn edge_comparison_ignores_undirected_orientation_and_duplicate_rows() {
        let table = |src: Vec<i64>, dst: Vec<i64>| Table {
            rows: src.len(),
            columns: BTreeMap::from([
                ("src".to_string(), src.into_iter().map(Cell::Int).collect()),
                ("dst".to_string(), dst.into_iter().map(Cell::Int).collect()),
            ]),
        };
        compare_edges(&table(vec![0, 1, 2], vec![1, 0, 3]), &table(vec![1, 3], vec![0, 2])).unwrap();
        let error = compare_edges(&table(vec![0, 0], vec![1, 1]), &table(vec![0], vec![1])).unwrap_err();
        assert!(error.contains("duplicate directed row"), "unexpected error: {error}");
    }

    #[test]
    fn numeric_comparison_rejects_non_finite_gpu_values() {
        let actual = Table { rows: 1, columns: BTreeMap::from([("value".to_string(), vec![Cell::Float(f64::NAN)])]) };
        let expected = Table { rows: 1, columns: BTreeMap::from([("value".to_string(), vec![Cell::Float(1.0)])]) };
        let error = compare_numeric(&actual, &expected, 1.0, 1.0).unwrap_err();
        assert!(error.contains("non-finite"), "unexpected error: {error}");
    }

    #[test]
    fn numeric_comparison_keeps_integer_columns_exact() {
        let actual = Table { rows: 1, columns: BTreeMap::from([("vertex".to_string(), vec![Cell::Int(1)])]) };
        let expected = Table { rows: 1, columns: BTreeMap::from([("vertex".to_string(), vec![Cell::Int(2)])]) };
        let error = compare_numeric(&actual, &expected, 100.0, 100.0).unwrap_err();
        assert!(error.contains("integer mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn numeric_comparison_accepts_boundary_and_rejects_larger_delta() {
        let table =
            |value| Table { rows: 1, columns: BTreeMap::from([("value".to_string(), vec![Cell::Float(value)])]) };
        compare_numeric(&table(101.0), &table(100.0), 0.0, 1.0 / 101.0).unwrap();
        let error = compare_numeric(&table(101.001), &table(100.0), 0.0, 1.0 / 101.0).unwrap_err();
        assert!(error.contains("mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn top_k_comparison_uses_vertex_set_under_score_ties() {
        let actual = Table {
            rows: 3,
            columns: BTreeMap::from([
                ("vertex".to_string(), vec![Cell::Int(7), Cell::Int(3), Cell::Int(5)]),
                ("value".to_string(), vec![Cell::Float(0.5), Cell::Float(0.9), Cell::Float(0.8)]),
            ]),
        };
        compare_top_k(&actual, &[3, 5]).unwrap();
    }

    #[test]
    fn bfs_comparison_accepts_alternate_valid_predecessor_tree() {
        let bfs_table = |predecessors: Vec<i64>| Table {
            rows: 4,
            columns: BTreeMap::from([
                ("vertex".to_string(), vec![0, 1, 2, 3].into_iter().map(Cell::Int).collect()),
                ("distance".to_string(), vec![0, 1, 1, 2].into_iter().map(Cell::Int).collect()),
                ("predecessor".to_string(), predecessors.into_iter().map(Cell::Int).collect()),
            ]),
        };
        let edges = [
            ParityEdge { source: 0, destination: 1, weight: 1.0 },
            ParityEdge { source: 0, destination: 2, weight: 1.0 },
            ParityEdge { source: 1, destination: 3, weight: 1.0 },
            ParityEdge { source: 2, destination: 3, weight: 1.0 },
        ];
        compare_bfs(&bfs_table(vec![-1, 0, 0, 1]), &bfs_table(vec![-1, 0, 0, 2]), &edges).unwrap();
        let error = compare_bfs(&bfs_table(vec![-1, 0, 0, 0]), &bfs_table(vec![-1, 0, 0, 2]), &edges).unwrap_err();
        assert!(error.contains("predecessor distance mismatch"), "unexpected error: {error}");
    }
}
