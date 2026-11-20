//! Int64 graph calls whose ids fit Int32 run on Int32 vertex ids. Each case
//! runs one call twice on the same graph: once with small ids, which narrow,
//! and once with every id moved up by `OFFSET`, which keeps the Int64 path.
//! Moving every id by one constant keeps their order, so renumbering assigns
//! the same internal ids and deterministic results match row for row.

use std::collections::BTreeMap;

use query_planning::plan::{GraphBfsPathOutput, GraphLouvainParams, GraphSsspParams, GraphSsspSource};

use super::*;

const OFFSET: i64 = 1 << 32;

/// One value of a result row, with float values compared by bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Value {
    Int(i64),
    Float(u64),
    Null,
}

struct SideInput {
    columns: Vec<(&'static str, Vec<i64>)>,
    floats: Vec<(&'static str, Vec<f64>)>,
}

impl SideInput {
    fn ids(columns: Vec<(&'static str, Vec<i64>)>) -> Self {
        Self { columns, floats: Vec::new() }
    }

    fn batch(&self, offset: i64) -> (Arc<Schema>, RecordBatch) {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for (name, values) in &self.columns {
            fields.push(Field::new(*name, DataType::Int64, false));
            arrays.push(Arc::new(Int64Array::from(values.iter().map(|value| value + offset).collect::<Vec<_>>())));
        }
        for (name, values) in &self.floats {
            fields.push(Field::new(*name, DataType::Float64, false));
            arrays.push(Arc::new(Float64Array::from(values.clone())));
        }
        let schema = Arc::new(Schema::new(fields));
        (Arc::clone(&schema), RecordBatch::try_new(schema, arrays).unwrap())
    }
}

struct GraphCall {
    algorithm: GraphAlgorithm,
    directed: bool,
    policy: GraphConstructionPolicy,
    edges: Vec<(i64, i64, f64)>,
    weighted: bool,
    side_inputs: Vec<SideInput>,
    params: fn(i64) -> GraphAlgorithmParams,
}

impl GraphCall {
    fn new(algorithm: GraphAlgorithm, edges: Vec<(i64, i64, f64)>, params: fn(i64) -> GraphAlgorithmParams) -> Self {
        Self {
            algorithm,
            directed: true,
            policy: GraphConstructionPolicy::RawLibcugraph,
            edges,
            weighted: false,
            side_inputs: Vec::new(),
            params,
        }
    }

    fn undirected(mut self) -> Self {
        self.directed = false;
        self.policy = GraphConstructionPolicy::PythonCugraph;
        self
    }

    fn weighted(mut self) -> Self {
        self.weighted = true;
        self
    }

    fn with_side_input(mut self, side_input: SideInput) -> Self {
        self.side_inputs.push(side_input);
        self
    }

    /// Runs the call with every edge, side-input and literal id moved up by
    /// `offset`, and returns its result with vertex ids moved back.
    fn run(&self, offset: i64) -> QueryEngineResult<Vec<Vec<Value>>> {
        let context = backend_context();
        let mut fields = vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)];
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(self.edges.iter().map(|edge| edge.0 + offset).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(self.edges.iter().map(|edge| edge.1 + offset).collect::<Vec<_>>())),
        ];
        if self.weighted {
            fields.push(Field::new("weight", DataType::Float64, false));
            columns.push(Arc::new(Float64Array::from(self.edges.iter().map(|edge| edge.2).collect::<Vec<_>>())));
        }
        let edge_schema = Arc::new(Schema::new(fields));
        let edge_batch = RecordBatch::try_new(Arc::clone(&edge_schema), columns).unwrap();
        let side_batches = self.side_inputs.iter().map(|side_input| side_input.batch(offset)).collect::<Vec<_>>();

        let mut builder = QueryPlanBuilder::new();
        let edges = builder.add_source(SourceNode::GpuResident(gpu_resident_source(
            "edges",
            Arc::clone(&edge_schema),
            &edge_batch,
        )));
        let sides = side_batches
            .iter()
            .enumerate()
            .map(|(index, (schema, batch))| {
                builder.add_source(SourceNode::GpuResident(gpu_resident_source(
                    &format!("side_{index}"),
                    Arc::clone(schema),
                    batch,
                )))
            })
            .collect();
        let spec = graph_spec_for_vertex_type(
            self.algorithm,
            GraphEdgeColumnBindings::new("src", "dst", self.weighted.then_some("weight")),
            GraphConstructionOptions {
                directed: self.directed,
                renumber: true,
                policy: self.policy,
                distinct_edge_pairs: false,
            },
            (self.params)(offset),
            &DataType::Int64,
        );
        let output_schema = Arc::clone(&spec.output_schema);
        let graph = builder.add_graph_algorithm(edges, sides, spec).unwrap();
        let plan = builder.finish_with_host_sink_schema(graph, Arc::clone(&output_schema)).unwrap();
        let admitted = context.admit_plan(&plan).unwrap();
        let runtime = Arc::new(admitted.runtime().unwrap());
        let mut gpu_inputs = GpuResidentInputMap::new();
        gpu_inputs.insert("edges".to_string(), graph_input_handle(&admitted, &runtime, edge_schema, &edge_batch));
        for (index, (schema, batch)) in side_batches.iter().enumerate() {
            gpu_inputs
                .insert(format!("side_{index}"), graph_input_handle(&admitted, &runtime, Arc::clone(schema), batch));
        }
        let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs);
        admitted.finish();
        let result = result?;
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("host sink should materialize the graph result");
        };
        assert_eq!(batch.schema().as_ref(), output_schema.as_ref(), "declared Int64 result schema");
        let vertex_columns = (0..batch.num_columns())
            .map(|column| {
                let reference =
                    graph_result_schema_for_vertex_type(self.algorithm, &(self.params)(0), &DataType::Int32);
                reference.field(column).data_type() == &DataType::Int32
            })
            .collect::<Vec<_>>();
        let mut rows = (0..batch.num_rows())
            .map(|row| {
                (0..batch.num_columns())
                    .map(|column| {
                        let array = batch.column(column);
                        if array.is_null(row) {
                            return Value::Null;
                        }
                        match array.data_type() {
                            DataType::Int64 => {
                                let value = array.as_primitive::<Int64Type>().value(row);
                                // -1 marks an absent predecessor and keeps its value.
                                Value::Int(if vertex_columns[column] && value >= offset {
                                    value - offset
                                } else {
                                    value
                                })
                            },
                            DataType::Float64 => Value::Float(array.as_primitive::<Float64Type>().value(row).to_bits()),
                            DataType::Float32 => {
                                Value::Float(u64::from(array.as_primitive::<Float32Type>().value(row).to_bits()))
                            },
                            data_type => panic!("unexpected graph result type {data_type}"),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        rows.sort();
        Ok(rows)
    }

    /// Asserts that the narrowed call returns the rows of the Int64 call.
    fn assert_matches_int64_path(&self, label: &str) -> Vec<Vec<Value>> {
        let narrowed = self.run(0).unwrap_or_else(|error| panic!("{label}: narrowed call failed: {error:#?}"));
        let int64 = self.run(OFFSET).unwrap_or_else(|error| panic!("{label}: Int64 call failed: {error:#?}"));
        assert_eq!(narrowed, int64, "{label}");
        assert!(!narrowed.is_empty(), "{label}");
        narrowed
    }
}

fn directed_edges() -> Vec<(i64, i64, f64)> {
    vec![(0, 1, 1.0), (1, 2, 2.0), (2, 0, 4.0), (2, 3, 1.5), (3, 4, 0.5), (1, 4, 3.0), (5, 6, 1.0)]
}

pub(super) fn narrowed_pagerank_matches_the_int64_path() {
    GraphCall::new(GraphAlgorithm::PageRank, directed_edges(), |_| {
        GraphAlgorithmParams::PageRank(GraphPageRankParams::default())
    })
    .assert_matches_int64_path("pagerank");
}

/// Vertices 5 and 6 are not reachable from 0, so their distance is the Int64
/// unreachable marker in both runs.
pub(super) fn narrowed_bfs_matches_the_int64_path_and_keeps_the_int64_unreachable_marker() {
    let rows = GraphCall::new(GraphAlgorithm::Bfs, directed_edges(), |offset| {
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(offset, None)))
    })
    .assert_matches_int64_path("bfs");
    let unreachable = rows.iter().filter(|row| row[1] == Value::Int(i64::MAX)).map(|row| row[0]).collect::<Vec<_>>();
    assert_eq!(unreachable, vec![Value::Int(5), Value::Int(6)]);
}

pub(super) fn narrowed_bfs_path_output_matches_the_int64_path() {
    let rows = GraphCall::new(GraphAlgorithm::Bfs, directed_edges(), |offset| {
        let mut params = GraphBfsParams::new(offset, None);
        params.predicates.target_vertices = Some(GraphPredicateInput::new(1, "vertex"));
        params.path_output = Some(GraphBfsPathOutput::new(7));
        GraphAlgorithmParams::Bfs(Box::new(params))
    })
    .with_side_input(SideInput::ids(vec![("vertex", vec![4])]))
    .assert_matches_int64_path("bfs path");
    assert_eq!(rows.len(), 3, "the path 0 -> 1 -> 4 has three vertices");
}

pub(super) fn narrowed_sssp_matches_the_int64_path() {
    GraphCall::new(GraphAlgorithm::Sssp, directed_edges(), |offset| {
        GraphAlgorithmParams::Sssp(GraphSsspParams { source_vertex: GraphSsspSource::I64(offset), cutoff: None })
    })
    .weighted()
    .assert_matches_int64_path("sssp");
}

pub(super) fn narrowed_jaccard_with_vertex_pairs_matches_the_int64_path() {
    GraphCall::new(GraphAlgorithm::Jaccard, directed_edges(), |_| {
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(1, "first", "second")))
    })
    .undirected()
    .with_side_input(SideInput::ids(vec![("first", vec![0, 1, 2, 5]), ("second", vec![1, 2, 4, 6])]))
    .assert_matches_int64_path("jaccard");
}

pub(super) fn narrowed_personalized_pagerank_matches_the_int64_path() {
    GraphCall::new(GraphAlgorithm::PersonalizedPageRank, directed_edges(), |_| {
        GraphAlgorithmParams::PersonalizedPageRank(GraphPersonalizedPageRankParams {
            pagerank: GraphPageRankParams::default(),
            personalization: GraphPageRankPersonalizationInput::new(1, "vertex", "value"),
        })
    })
    .with_side_input(SideInput { columns: vec![("vertex", vec![0, 3])], floats: vec![("value", vec![0.75, 0.25])] })
    .assert_matches_int64_path("personalized pagerank");
}

/// Louvain community ids are arbitrary, so the two runs are compared by the
/// partition of vertices they produce.
pub(super) fn narrowed_louvain_matches_the_int64_partition() {
    let call = GraphCall::new(
        GraphAlgorithm::Louvain,
        vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 3, 1.0), (2, 3, 1.0)],
        |_| GraphAlgorithmParams::Louvain(GraphLouvainParams::default()),
    )
    .undirected();
    let partition = |rows: Vec<Vec<Value>>| {
        let mut communities = BTreeMap::<Value, Vec<Value>>::new();
        for row in rows {
            communities.entry(row[1]).or_default().push(row[0]);
        }
        let mut members = communities.into_values().collect::<Vec<_>>();
        members.sort();
        members
    };
    let narrowed = partition(call.run(0).unwrap());
    let int64 = partition(call.run(OFFSET).unwrap());
    assert_eq!(narrowed, int64);
    assert_eq!(narrowed.iter().map(Vec::len).sum::<usize>(), 6);
}

/// A pair id above the Int32 range keeps the call on Int64 ids, so the pair
/// fails as an absent vertex on the Int64 graph and reaches no Int32 cast.
pub(super) fn similarity_pair_id_outside_int32_keeps_the_int64_absent_vertex_error() {
    let call = GraphCall::new(GraphAlgorithm::Jaccard, directed_edges(), |_| {
        GraphAlgorithmParams::Similarity(GraphSimilarityParams::new(GraphVertexPairInput::new(1, "first", "second")))
    })
    .undirected()
    .with_side_input(SideInput::ids(vec![("first", vec![0]), ("second", vec![OFFSET + 3])]));
    let narrowed = call.run(0).expect_err("a pair vertex absent from the graph must be rejected");
    let int64 = call.run(OFFSET).expect_err("a pair vertex absent from the graph must be rejected");
    assert_eq!(narrowed.code(), QueryErrorCode::RapidsInvalidInput, "{narrowed:#?}");
    assert_eq!((narrowed.code(), narrowed.operation()), (int64.code(), int64.operation()), "{narrowed:#?}");
}

/// A literal BFS source above the Int32 range keeps the call on Int64 ids, so
/// the absent source has the outcome it has on an Int64 graph.
pub(super) fn bfs_literal_source_outside_int32_keeps_the_int64_outcome() {
    let call = GraphCall::new(GraphAlgorithm::Bfs, directed_edges(), |offset| {
        GraphAlgorithmParams::Bfs(Box::new(GraphBfsParams::new(offset + OFFSET, None)))
    });
    let outcome = |offset| {
        call.run(offset).map_err(|error| (error.code(), error.operation().to_string(), error.message().to_string()))
    };
    assert_eq!(outcome(0), outcome(OFFSET));
}
