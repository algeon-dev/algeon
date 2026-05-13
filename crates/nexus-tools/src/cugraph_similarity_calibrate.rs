//! Explicit one-case cuGraph similarity calibration runner.
//!
//! This is intentionally a manual tool, not a Cargo test target. Each
//! invocation constructs one symmetric graph from a synthetic topology or a
//! selected local Parquet edge fixture, executes one requested metric once,
//! and emits the native runtime-observation snapshot together with the graph
//! memory rows that admission consumed.

use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use arrow::{
    array::{Array, ArrayRef, Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use nexus_query_engine::{
    exec::execute_query_plan_with_gpu_inputs_and_runtime_observation,
    graph_memory::{GraphMemoryModel, GraphMemoryModelInput, GraphPairCardinalitySource},
    observability::{RuntimeObservationConfig, RuntimeObservationSession, write_runtime_observation_jsonl},
    plan::{
        GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphConstructionOptions, GraphConstructionPolicy,
        GraphEdgeColumnBindings, GraphSimilarityParams, GraphVertexPairInput, QueryPlanBuilder,
    },
    runtime::{
        ExecutionContext, GpuBackendDeviceConfig, GpuDeviceMemoryLimit, GpuMemoryOwnership, QueryExecutionConfig,
        resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle},
    },
    source::GpuResidentSource,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};

use crate::{ToolResult, gpu_backend::ToolGpuBackend};

const EDGE_INPUT: &str = "calibration_edges";
const PAIR_INPUT: &str = "calibration_pairs";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexDtype {
    Int32,
    Int64,
}

impl VertexDtype {
    fn data_type(self) -> DataType {
        match self {
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Int32 => "int32",
            Self::Int64 => "int64",
        }
    }
}

/// Shape of the graph used by one calibration case.
///
/// The two shapes deliberately exercise different adjacency distributions:
/// a degree-two cycle for the sparse, uniform case and a star for the
/// hub-skewed case. They remain small so pair-cardinality pricing dominates
/// the measured case, while still allowing the calibration matrix to detect a
/// topology-specific native allocation before it is hidden in a broad safety
/// multiplier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CalibrationTopology {
    SparseUniform,
    HubSkewed,
}

impl CalibrationTopology {
    fn as_str(self) -> &'static str {
        match self {
            Self::SparseUniform => "sparse-uniform",
            Self::HubSkewed => "hub-skewed",
        }
    }

    fn vertex_count(self) -> i64 {
        8
    }
}

#[derive(Debug)]
struct CalibrationArgs {
    algorithm: GraphAlgorithm,
    pair_rows: usize,
    weighted: bool,
    vertex_dtype: VertexDtype,
    topology: CalibrationTopology,
    edge_parquet: Option<PathBuf>,
    edge_row_groups: usize,
    observation_out: Option<PathBuf>,
}

struct CalibrationInput {
    edge_schema: Arc<Schema>,
    edge_batch: RecordBatch,
    pair_schema: Arc<Schema>,
    pair_batch: RecordBatch,
    vertex_dtype: VertexDtype,
    topology_label: &'static str,
    description: Value,
}

pub fn run(args: Vec<String>) -> ToolResult<()> {
    let args = parse_args(args)?;
    let calibration_input = calibration_input(&args)?;
    let service_config = QueryExecutionConfig::default();
    let device = GpuBackendDeviceConfig::new(0, GpuMemoryOwnership::WholeDeviceExclusive)
        .with_memory_limit(GpuDeviceMemoryLimit::default())
        .with_max_active_queries(1)
        .with_installation_bounds(2, 1, 1);
    let backend = ToolGpuBackend::build(service_config, device)?;
    let result = (|| -> ToolResult<()> {
        let query_handle = backend.session().admit_query()?;
        let admitted_context = query_handle.acquire_context()?;
        let context = admitted_context.context();
        let runtime = context.runtime()?;
        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(nexus_query_engine::source::SourceNode::GpuResident(GpuResidentSource::new(
            EDGE_INPUT,
            Arc::clone(&calibration_input.edge_schema),
        )));
        let pairs = builder.add_source(nexus_query_engine::source::SourceNode::GpuResident(GpuResidentSource::new(
            PAIR_INPUT,
            Arc::clone(&calibration_input.pair_schema),
        )));
        let spec = GraphAlgorithmSpec::with_stable_schema(
            args.algorithm,
            GraphEdgeColumnBindings::new("src", "dst", args.weighted.then_some("weight")),
            // Every calibration input includes both directions, so raw libcugraph
            // receives the required symmetric relation without an adapter-only
            // similarity repair path.
            GraphConstructionOptions {
                directed: false,
                renumber: true,
                policy: GraphConstructionPolicy::RawLibcugraph,
            },
            GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(
                pairs, "first", "second",
            ))),
        );
        let model = calibration_model_breakdown(&spec, &calibration_input, args.weighted)?;
        let graph = builder.try_add_graph_algorithm(edges, spec)?;
        let plan = builder.finish_with_host_sink(graph);

        let inputs = calibration_inputs(
            context,
            &runtime,
            &calibration_input.edge_schema,
            &calibration_input.edge_batch,
            &calibration_input.pair_schema,
            &calibration_input.pair_batch,
        )?;

        let outcome = execute_query_plan_with_gpu_inputs_and_runtime_observation(
            context,
            &plan,
            &inputs,
            RuntimeObservationSession::new(
                "cugraph-similarity-calibrate",
                None,
                RuntimeObservationConfig::native_execution(),
            ),
        );
        let (result, metrics, observation) = outcome.into_parts();
        let output_rows = result?.as_host_arrow().map_or(0, |batch| batch.num_rows());
        let observation = observation.ok_or("calibration execution did not produce a runtime observation snapshot")?;
        let mut runtime_observation_jsonl = Vec::new();
        write_runtime_observation_jsonl(&observation, &mut runtime_observation_jsonl)?;
        let report = json!({
            "schema": "datafusion_nexus.cugraph_similarity_calibration.v1",
            "algorithm": args.algorithm.as_str(),
            "pair_rows": args.pair_rows,
            "weighted": args.weighted,
            "vertex_dtype": calibration_input.vertex_dtype.as_str(),
            "topology": calibration_input.topology_label,
            "input": calibration_input.description,
            "model": model,
            "output_rows": output_rows,
            "metrics": metrics.to_json(),
            "runtime_observation_jsonl": String::from_utf8(runtime_observation_jsonl)?,
        });
        let rendered = serde_json::to_string_pretty(&report)?;
        if let Some(path) = args.observation_out {
            fs::write(path, &rendered)?;
        } else {
            println!("{rendered}");
        }
        Ok(())
    })();
    let cleanup = backend.close_and_wait(Duration::from_secs(30)).map_err(|error| error.into());
    combine_work_and_cleanup(result, cleanup)
}

fn combine_work_and_cleanup<T>(work: ToolResult<T>, cleanup: ToolResult<()>) -> ToolResult<T> {
    match (work, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(work_error), Err(cleanup_error)) => {
            Err(format!("calibration failed: {work_error:#}; backend cleanup also failed: {cleanup_error:#}").into())
        },
    }
}

fn calibration_inputs(
    context: &ExecutionContext,
    runtime: &cudf::Runtime,
    edge_schema: &Arc<Schema>,
    edge_batch: &RecordBatch,
    pair_schema: &Arc<Schema>,
    pair_batch: &RecordBatch,
) -> ToolResult<GpuResidentInputMap> {
    let edge_frame = GpuDataFrame::new(DataFrame::from_record_batch(runtime, edge_batch)?);
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(EDGE_INPUT.to_string(), GpuResultHandle::new(context, edge_frame, Arc::clone(edge_schema))?);
    inputs.insert(
        PAIR_INPUT.to_string(),
        GpuResultHandle::new(
            context,
            GpuDataFrame::new(DataFrame::from_record_batch(runtime, pair_batch)?),
            Arc::clone(pair_schema),
        )?,
    );
    Ok(inputs)
}

fn parse_args(args: Vec<String>) -> ToolResult<CalibrationArgs> {
    let mut algorithm = None;
    let mut pair_rows = None;
    let mut weighted = false;
    let mut vertex_dtype = VertexDtype::Int32;
    let mut topology = CalibrationTopology::SparseUniform;
    let mut edge_parquet = None;
    let mut edge_row_groups = 1;
    let mut observation_out = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--algorithm" => algorithm = Some(parse_algorithm(&next_value(&mut args, "--algorithm")?)?),
            "--pair-rows" => pair_rows = Some(next_value(&mut args, "--pair-rows")?.parse::<usize>()?),
            "--weighted" => weighted = true,
            "--vertex-dtype" => {
                vertex_dtype = match next_value(&mut args, "--vertex-dtype")?.as_str() {
                    "int32" => VertexDtype::Int32,
                    "int64" => VertexDtype::Int64,
                    other => {
                        return Err(format!("unsupported --vertex-dtype {other:?}; expected int32 or int64").into());
                    },
                };
            },
            "--topology" => {
                topology = match next_value(&mut args, "--topology")?.as_str() {
                    "sparse-uniform" => CalibrationTopology::SparseUniform,
                    "hub-skewed" => CalibrationTopology::HubSkewed,
                    other => {
                        return Err(
                            format!("unsupported --topology {other:?}; expected sparse-uniform or hub-skewed").into()
                        );
                    },
                };
            },
            "--edge-parquet" => edge_parquet = Some(PathBuf::from(next_value(&mut args, "--edge-parquet")?)),
            "--edge-row-groups" => edge_row_groups = next_value(&mut args, "--edge-row-groups")?.parse::<usize>()?,
            "--observation-out" => observation_out = Some(PathBuf::from(next_value(&mut args, "--observation-out")?)),
            "-h" | "--help" => {
                return Err(
                    "usage: cugraph-similarity-calibrate --algorithm <jaccard|sorensen|overlap|cosine> --pair-rows <N> [--weighted] [--vertex-dtype int32|int64] [--topology sparse-uniform|hub-skewed] [--edge-parquet PATH --edge-row-groups N] [--observation-out PATH]"
                        .into(),
                );
            },
            other => return Err(format!("unknown cugraph-similarity-calibrate argument: {other}").into()),
        }
    }
    if edge_row_groups == 0 {
        return Err("--edge-row-groups must be greater than zero".into());
    }
    Ok(CalibrationArgs {
        algorithm: algorithm.ok_or("missing --algorithm")?,
        pair_rows: pair_rows.ok_or("missing --pair-rows")?,
        weighted,
        vertex_dtype,
        topology,
        edge_parquet,
        edge_row_groups,
        observation_out,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, option: &str) -> ToolResult<String> {
    args.next().ok_or_else(|| format!("missing value for {option}").into())
}

fn parse_algorithm(value: &str) -> ToolResult<GraphAlgorithm> {
    match value {
        "jaccard" => Ok(GraphAlgorithm::Jaccard),
        "sorensen" => Ok(GraphAlgorithm::Sorensen),
        "overlap" => Ok(GraphAlgorithm::Overlap),
        "cosine" => Ok(GraphAlgorithm::Cosine),
        other => {
            Err(format!("unsupported --algorithm {other:?}; expected jaccard, sorensen, overlap, or cosine").into())
        },
    }
}

fn calibration_input(args: &CalibrationArgs) -> ToolResult<CalibrationInput> {
    if let Some(path) = args.edge_parquet.as_deref() {
        if args.vertex_dtype != VertexDtype::Int64 {
            return Err(
                "--edge-parquet exposes the citation fixture's Int64 vertex domain; pass --vertex-dtype int64".into()
            );
        }
        return parquet_calibration_input(path, args.edge_row_groups, args.pair_rows, args.weighted);
    }

    let (edge_schema, edge_batch) = edge_batch(args.topology, args.vertex_dtype, args.weighted)?;
    let (pair_schema, pair_batch) = pair_batch(args.topology, args.vertex_dtype, args.pair_rows)?;
    Ok(CalibrationInput {
        edge_schema,
        edge_batch,
        pair_schema,
        pair_batch,
        vertex_dtype: args.vertex_dtype,
        topology_label: args.topology.as_str(),
        description: json!({
            "kind": "synthetic_topology",
            "topology": args.topology.as_str(),
        }),
    })
}

fn parquet_calibration_input(
    path: &Path,
    selected_row_groups: usize,
    pair_rows: usize,
    weighted: bool,
) -> ToolResult<CalibrationInput> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let available_row_groups = builder.metadata().num_row_groups();
    if selected_row_groups > available_row_groups {
        return Err(format!(
            "--edge-row-groups={selected_row_groups} exceeds the {available_row_groups} row groups in {}",
            path.display(),
        )
        .into());
    }
    let reader = builder.with_row_groups((0..selected_row_groups).collect()).build()?;
    let mut source = Vec::new();
    let mut destination = Vec::new();
    let mut source_weights = Vec::new();
    for batch in reader {
        let batch = batch?;
        let source_column = required_int64_parquet_column(&batch, path, "src")?;
        let destination_column = required_int64_parquet_column(&batch, path, "dst")?;
        let weight_column = weighted.then(|| required_float64_parquet_column(&batch, path, "weight")).transpose()?;
        for row in 0..batch.num_rows() {
            source.push(source_column.value(row));
            destination.push(destination_column.value(row));
            if let Some(weight_column) = weight_column {
                source_weights.push(weight_column.value(row));
            }
        }
    }
    if source.is_empty() {
        return Err(format!("{} selected no edge rows", path.display()).into());
    }

    // The citation fixture is directed. Similarity requires a symmetric raw
    // input, so the calibration input explicitly supplies each reverse edge;
    // this is caller-owned raw-libcugraph input shape, not a similarity adapter.
    let source_edge_rows = source.len();
    let mut symmetric_source = Vec::with_capacity(source_edge_rows.saturating_mul(2));
    let mut symmetric_destination = Vec::with_capacity(source_edge_rows.saturating_mul(2));
    let mut symmetric_weights = Vec::with_capacity(source_edge_rows.saturating_mul(2));
    for row in 0..source_edge_rows {
        symmetric_source.push(source[row]);
        symmetric_destination.push(destination[row]);
        symmetric_source.push(destination[row]);
        symmetric_destination.push(source[row]);
        if weighted {
            symmetric_weights.push(source_weights[row]);
            symmetric_weights.push(source_weights[row]);
        }
    }

    let mut pair_first = Vec::with_capacity(pair_rows);
    let mut pair_second = Vec::with_capacity(pair_rows);
    for row in 0..pair_rows {
        let edge_row = row % source_edge_rows;
        pair_first.push(source[edge_row]);
        pair_second.push(destination[edge_row]);
    }

    let mut edge_fields = vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)];
    let mut edge_columns: Vec<ArrayRef> =
        vec![Arc::new(Int64Array::from(symmetric_source)), Arc::new(Int64Array::from(symmetric_destination))];
    if weighted {
        edge_fields.push(Field::new("weight", DataType::Float64, false));
        edge_columns.push(Arc::new(Float64Array::from(symmetric_weights)));
    }
    let edge_schema = Arc::new(Schema::new(edge_fields));
    let edge_batch = RecordBatch::try_new(Arc::clone(&edge_schema), edge_columns)?;
    let pair_schema = Arc::new(Schema::new(vec![
        Field::new("first", DataType::Int64, false),
        Field::new("second", DataType::Int64, false),
    ]));
    let pair_batch = RecordBatch::try_new(
        Arc::clone(&pair_schema),
        vec![Arc::new(Int64Array::from(pair_first)), Arc::new(Int64Array::from(pair_second))],
    )?;
    Ok(CalibrationInput {
        edge_schema,
        edge_batch,
        pair_schema,
        pair_batch,
        vertex_dtype: VertexDtype::Int64,
        topology_label: "parquet-fixture",
        description: json!({
            "kind": "parquet_edge_fixture",
            "edge_parquet": path.display().to_string(),
            "selected_row_groups": selected_row_groups,
            "source_directed_edge_rows": source_edge_rows,
            "native_symmetric_edge_rows": source_edge_rows.saturating_mul(2),
        }),
    })
}

fn required_int64_parquet_column<'a>(batch: &'a RecordBatch, path: &Path, name: &str) -> ToolResult<&'a Int64Array> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("{} does not contain required Int64 {name:?} edge column", path.display()))?;
    let values = column.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        format!("{} column {name:?} must be Int64 for the citation fixture calibration", path.display())
    })?;
    if values.null_count() != 0 {
        return Err(format!("{} column {name:?} contains nulls", path.display()).into());
    }
    Ok(values)
}

fn required_float64_parquet_column<'a>(
    batch: &'a RecordBatch,
    path: &Path,
    name: &str,
) -> ToolResult<&'a Float64Array> {
    let column = batch
        .column_by_name(name)
        .ok_or_else(|| format!("{} does not contain required Float64 {name:?} edge column", path.display()))?;
    let values = column.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        format!("{} column {name:?} must be Float64 for weighted fixture calibration", path.display())
    })?;
    if values.null_count() != 0 {
        return Err(format!("{} column {name:?} contains nulls", path.display()).into());
    }
    Ok(values)
}

fn calibration_model_breakdown(
    spec: &GraphAlgorithmSpec,
    input: &CalibrationInput,
    weighted: bool,
) -> ToolResult<Value> {
    let vertex_width_bytes: u64 = match input.vertex_dtype {
        VertexDtype::Int32 => 4,
        VertexDtype::Int64 => 8,
    };
    let pair_rows = input.pair_batch.num_rows() as u64;
    let pair_columns = pair_rows.saturating_mul(vertex_width_bytes.saturating_mul(2));
    let native_scores = pair_rows.saturating_mul(8);
    let normalized_output = pair_rows.saturating_mul(24);
    let estimate = GraphMemoryModel.estimate(
        spec,
        GraphMemoryModelInput {
            raw_input_edges: None,
            input_edges: input.edge_batch.num_rows() as u64,
            vertex_width_bytes,
            weight_width_bytes: weighted.then_some(8),
            has_edge_id: false,
            device_total_memory_bytes: None,
            vertex_estimate: None,
            statistics: None,
            edge_normalize_inserted: false,
            construction_workspace_bounds: None,
            pair_rows: None,
            pair_vertex_width_bytes: None,
            pair_cardinality_source: None,
        }
        .with_similarity_pair_input(
            pair_rows,
            vertex_width_bytes,
            GraphPairCardinalitySource::RuntimeMaterialized,
        ),
    )?;
    let algorithm_workspace = estimate
        .execution_bytes
        .saturating_sub(normalized_output)
        .saturating_sub(pair_columns)
        .saturating_sub(pair_columns)
        .saturating_sub(native_scores);
    Ok(json!({
        "construction_bytes": estimate.construction_bytes,
        "pair_input_overlap_bytes": pair_columns,
        "native_vertex_pairs_bytes": pair_columns,
        "native_similarity_scores_bytes": native_scores,
        "normalized_sql_output_bytes": normalized_output,
        "algorithm_workspace_bytes": algorithm_workspace,
        "execution_bytes": estimate.execution_bytes,
        "total_requested_bytes": estimate.requested_bytes,
    }))
}

fn edge_batch(
    topology: CalibrationTopology,
    vertex_dtype: VertexDtype,
    weighted: bool,
) -> ToolResult<(Arc<Schema>, RecordBatch)> {
    // Keep both edge directions because raw libcugraph deliberately leaves
    // input-shape ownership with the caller. The topology is part of the
    // content identity used by the warm-cache path.
    let (src, dst) = topology_edges(topology);
    let mut fields =
        vec![Field::new("src", vertex_dtype.data_type(), false), Field::new("dst", vertex_dtype.data_type(), false)];
    let mut columns = vec![vertex_array(vertex_dtype, src), vertex_array(vertex_dtype, dst)];
    if weighted {
        fields.push(Field::new("weight", DataType::Float64, false));
        columns.push(Arc::new(Float64Array::from(topology_weights(src.len()))));
    }
    let schema = Arc::new(Schema::new(fields));
    Ok((Arc::clone(&schema), RecordBatch::try_new(schema, columns)?))
}

fn pair_batch(
    topology: CalibrationTopology,
    vertex_dtype: VertexDtype,
    pair_rows: usize,
) -> ToolResult<(Arc<Schema>, RecordBatch)> {
    let (first, second) = topology_pairs(topology, pair_rows);
    let schema = Arc::new(Schema::new(vec![
        Field::new("first", vertex_dtype.data_type(), false),
        Field::new("second", vertex_dtype.data_type(), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![vertex_array(vertex_dtype, &first), vertex_array(vertex_dtype, &second)],
    )?;
    Ok((schema, batch))
}

fn topology_edges(topology: CalibrationTopology) -> (&'static [i64], &'static [i64]) {
    match topology {
        // A degree-two cycle across eight vertices. Each contiguous source /
        // destination pair is a reciprocal edge with the same weight.
        CalibrationTopology::SparseUniform => {
            (&[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0], &[1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7])
        },
        // A degree-seven hub with seven degree-one leaves. This is purposely
        // skewed but still symmetric and simple for raw libcugraph ingress.
        CalibrationTopology::HubSkewed => {
            (&[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7], &[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0])
        },
    }
}

fn topology_weights(edge_count: usize) -> Vec<f64> {
    // Assign each reciprocal edge pair the same strictly-positive value. The
    // chosen values exercise weighted dispatch without changing topology.
    (0..edge_count).map(|index| 1.0 + ((index / 2) % 4) as f64).collect()
}

fn topology_pairs(topology: CalibrationTopology, pair_rows: usize) -> (Vec<i64>, Vec<i64>) {
    let vertex_count = topology.vertex_count();
    let mut first = Vec::with_capacity(pair_rows);
    let mut second = Vec::with_capacity(pair_rows);
    for index in 0..pair_rows {
        match topology {
            CalibrationTopology::SparseUniform => {
                first.push((index as i64) % vertex_count);
                second.push(((index as i64) + 2) % vertex_count);
            },
            CalibrationTopology::HubSkewed => {
                let leaf = ((index as i64) % (vertex_count - 1)) + 1;
                if index % 2 == 0 {
                    first.push(0);
                    second.push(leaf);
                } else {
                    first.push(leaf);
                    second.push((leaf % (vertex_count - 1)) + 1);
                }
            },
        }
    }
    (first, second)
}

fn vertex_array(dtype: VertexDtype, values: &[i64]) -> ArrayRef {
    match dtype {
        VertexDtype::Int32 => Arc::new(Int32Array::from(values.iter().map(|value| *value as i32).collect::<Vec<_>>())),
        VertexDtype::Int64 => Arc::new(Int64Array::from(values.to_vec())),
    }
}
