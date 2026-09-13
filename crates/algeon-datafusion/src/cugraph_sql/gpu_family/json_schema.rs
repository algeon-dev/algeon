use arrow::datatypes::{DataType, SchemaRef};
use query_engine::plan::GraphVertexIdTypeSupport;
use serde_json::{Map, Value, json};

use crate::cugraph_sql::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsPredicateInputs, CugraphBfsSource,
    CugraphBfsSourceVertex, CugraphGraphOptions, CugraphPageRankParams, CugraphSqlConfig, CugraphVertexSelector,
    stable_result_schema,
};

pub(super) fn schema_json(schema: &SchemaRef) -> String {
    schema_json_value(schema).to_string()
}

pub(super) fn schema_json_value(schema: &SchemaRef) -> Value {
    json!({
        "fields": schema
            .fields()
            .iter()
            .map(|field| {
                json!({
                    "name": field.name(),
                    "data_type": data_type_label(field.data_type()),
                    "nullable": field.is_nullable(),
                })
            })
            .collect::<Vec<_>>()
    })
}

pub(super) fn vertex_id_contract_json(algorithm: CugraphAlgorithm) -> Value {
    let numeric_endpoint_types = match algorithm.native_algorithm().vertex_id_type_support() {
        GraphVertexIdTypeSupport::Int32OrInt64 => json!(["Int32", "Int64"]),
        GraphVertexIdTypeSupport::Int32OnlyLegacyEdgeType => json!(["Int32"]),
    };
    let logical_string = algorithm.definition().logical_string_contract();
    json!({
        "numeric_endpoint_types": numeric_endpoint_types,
        "logical_string_endpoint_types": if logical_string.is_some() {
            json!(["Utf8", "LargeUtf8", "Utf8View"])
        } else {
            json!([])
        },
        "logical_string_output_type": logical_string.map(|_| "Utf8"),
        "native_logical_mapping_type": logical_string.map(|contract| format!("{:?}", contract.native_id_type.data_type())),
        "edge_id_supported_for_logical_string": false,
        "call_specific_authority": "gpu_validate_call",
    })
}

pub(super) fn metadata_output_schema_json(algorithm: CugraphAlgorithm) -> String {
    if algorithm == CugraphAlgorithm::Bfs {
        return json!({"fields": bfs_raw_mode_fields(false)}).to_string();
    }
    schema_json(&stable_result_schema(algorithm))
}

pub(super) fn options_schema_json(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> String {
    let graph_options = algorithm.effective_default_graph_options(&config.default_graph_options);
    let mut schema = json!({
        "graph_options": graph_options_schema_json(algorithm, &graph_options),
        "algorithm_options": algorithm_options_schema_json(algorithm)
    });
    if algorithm == CugraphAlgorithm::Bfs {
        schema.as_object_mut().expect("options schema is an object").insert(
            "result_schema_authority".to_string(),
            json!({
                "schema_field": "result_schemas_json",
                "call_specific_authority": "gpu_validate_call",
                "note": "result_schemas_json carries generic default and mode schemas; run gpu_validate_call for the concrete table-specific output schema."
            }),
        );
        schema.as_object_mut().expect("options schema is an object").insert(
            "type_aliases".to_string(),
            json!({
                "vertex_domain": "same logical vertex domain as src_col and dst_col",
                "edge_id_domain": "same integer domain as edge_id_col"
            }),
        );
    }
    if algorithm.is_similarity() {
        schema.as_object_mut().expect("options schema is an object").insert(
            "type_aliases".to_string(),
            json!({
                "vertex_domain": "same numeric or logical string domain as src_col and dst_col",
                "vertex_pairs_domain": "same Int32/Int64 dtype for numeric graphs; any Utf8/LargeUtf8/Utf8View member for logical string graphs"
            }),
        );
    }
    if algorithm == CugraphAlgorithm::ForceAtlas2 {
        schema.as_object_mut().expect("options schema is an object").insert(
            "type_aliases".to_string(),
            json!({
                "vertex_domain": "same numeric or logical string domain as src_col and dst_col",
                "Float32": "non-null Float32"
            }),
        );
    }
    schema.to_string()
}

fn graph_options_schema_json(algorithm: CugraphAlgorithm, options: &CugraphGraphOptions) -> Value {
    let mut directed = json!({
        "scope": "graph",
        "type": "Boolean",
        "default": options.directed,
        "description": "Whether graph construction treats edges as directed."
    });
    if algorithm.is_similarity() {
        directed
            .as_object_mut()
            .expect("directed schema is an object")
            .insert("allowed_values".to_string(), json!([false]));
        directed.as_object_mut().expect("directed schema is an object").insert(
            "valid_when".to_string(),
            json!("required false: similarity operates on an undirected/symmetric graph"),
        );
    }
    json!({
        "directed": directed,
        "renumber": {
            "scope": "graph",
            "type": "Boolean",
            "default": options.renumber,
            "description": "Whether graph construction may renumber external vertex identifiers internally."
        },
        "construction_policy": {
            "scope": "graph",
            "type": "Utf8",
            "allowed_values": ["python_cugraph", "raw_libcugraph"],
            "default": options.construction_policy.as_str(),
            "description": "Edge-list construction semantics used before calling libcugraph."
        }
    })
}

fn algorithm_options_schema_json(algorithm: CugraphAlgorithm) -> Value {
    match CugraphAlgorithmParams::from_native_defaults(algorithm, None) {
        CugraphAlgorithmParams::PageRank(params) => Value::Object(pagerank_iteration_options(&params)),
        CugraphAlgorithmParams::PersonalizedPageRank(params) => {
            let mut object = pagerank_iteration_options(&params.pagerank);
            object.insert(
                "personalization_table".to_string(),
                json!({
                    "scope": "side_input",
                    "type": "Utf8",
                    "required": true,
                    "description": "Table containing personalized PageRank vertex weights."
                }),
            );
            object.insert(
                "personalization_vertex_col".to_string(),
                json!({
                    "scope": "side_input",
                    "type": "Utf8",
                    "required": true,
                    "example": params.personalization.vertex_col,
                    "description": "Column in personalization_table naming the vertices the random walk restarts at; vertices absent from the table receive no restart mass."
                }),
            );
            object.insert(
                "personalization_value_col".to_string(),
                json!({
                    "scope": "side_input",
                    "type": "Utf8",
                    "required": true,
                    "example": params.personalization.value_col,
                    "description": "Column in personalization_table holding each personalization vertex's relative restart weight; larger values pull more rank toward that vertex."
                }),
            );
            Value::Object(object)
        },
        CugraphAlgorithmParams::Bfs(_) => bfs_options_schema_json(),
        CugraphAlgorithmParams::Louvain(params) => {
            json!({
                "max_level": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_level,
                    "min": 1,
                    "description": "Maximum number of hierarchy levels (coarsening rounds) to run. Lowering it truncates the hierarchy earlier."
                },
                "threshold": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.threshold,
                    "min": 0.0,
                    "description": "Minimum modularity gain: a vertex move is accepted, and a level continues, only while the gain exceeds this threshold. Larger values stop earlier with a coarser result."
                },
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0,
                    "description": "Resolution parameter (gamma) in the modularity formula. Higher values produce more, smaller communities; lower values produce fewer, larger ones."
                }
            })
        },
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0,
                    "description": "Power-iteration convergence tolerance: iteration stops once the L1 norm of the score change between consecutive iterations drops below epsilon. Smaller values tighten convergence and may need more iterations."
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1,
                    "description": "Upper bound on power iterations. The call fails with a non-convergence error if the bound is reached before epsilon is met."
                }
            })
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            json!({
                "alpha": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.alpha,
                    "exclusive_min": 0.0,
                    "max": 1.0,
                    "description": "Attenuation factor applied per path length; larger values let longer paths contribute more. cuGraph requires alpha below the inverse of the graph's largest eigenvalue for the iteration to converge."
                },
                "beta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.beta,
                    "exclusive_min": 0.0,
                    "description": "Constant added to every vertex's score in each iteration. One scalar applies to all vertices; per-vertex betas are not exposed."
                },
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0,
                    "description": "Convergence tolerance: iteration stops once the L1 sum of score changes between consecutive iterations is below the vertex count multiplied by epsilon. Smaller values tighten convergence and may need more iterations."
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1,
                    "description": "Upper bound on Katz iterations."
                }
            })
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0,
                    "description": "Convergence tolerance on the summed change in hub scores between consecutive iterations. Smaller values tighten convergence and may need more iterations."
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1,
                    "description": "Upper bound on HITS iterations."
                },
                "normalize": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.normalize,
                    "description": "When true, the final hub and authority score arrays are each scaled to L1 norm 1.0 before being returned."
                }
            })
        },
        CugraphAlgorithmParams::BetweennessCentrality(params) => betweenness_options_schema_json(
            params.normalized,
            Some(params.include_endpoints),
            params.exact_vertex_threshold,
        ),
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            betweenness_options_schema_json(params.normalized, None, params.exact_vertex_threshold)
        },
        CugraphAlgorithmParams::Sssp(params) => json!({
            "cutoff": {
                "scope": "algorithm",
                "type": "Float64|null",
                "default": params.cutoff,
                "min": 0.0,
                "description": "Maximum edge-weight sum to explore from the source; vertices farther than the cutoff are reported unreachable. Null means no cutoff."
            }
        }),
        CugraphAlgorithmParams::CoreNumber(params) => {
            json!({
                "degree_type": {
                    "scope": "algorithm",
                    "type": "Utf8",
                    "allowed_values": ["in", "out", "in_out"],
                    "default": params.degree_type.as_str(),
                    "description": "Which edges count toward a vertex's degree in the k-core decomposition: incoming (in), outgoing (out), or both (in_out)."
                }
            })
        },
        CugraphAlgorithmParams::KCore(params) => {
            json!({
                "k": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.k,
                    "min": 1,
                    "description": "Order of the core to extract: vertices whose core number is below k are dropped along with their edges."
                },
                "degree_type": {
                    "scope": "algorithm",
                    "type": "Utf8",
                    "allowed_values": ["in", "out", "in_out"],
                    "default": params.degree_type.as_str(),
                    "description": "Degree direction used by the underlying core-number computation: incoming (in), outgoing (out), or both (in_out)."
                }
            })
        },
        CugraphAlgorithmParams::Leiden(params) => {
            json!({
                "max_level": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_level,
                    "min": 1,
                    "description": "Maximum number of hierarchy levels (coarsening rounds) to run. Lowering it truncates the hierarchy earlier."
                },
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0,
                    "description": "Resolution parameter (gamma) in the modularity formula. Higher values produce more, smaller communities; lower values produce fewer, larger ones."
                },
                "theta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.theta,
                    "exclusive_min": 0.0,
                    "description": "Scales the modularity gain in the Leiden refinement phase when computing the probability of joining a random community. Larger values make refinement more random."
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed,
                    "description": "Seed for the native random-number generator state that drives the randomized refinement phase. Different seeds explore different refinements."
                }
            })
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({
                "min_weight": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.min_weight,
                    "min": 0.0,
                    "description": "Floor for edge weights in the final Louvain graph: an edge that no ensemble partition placed inside a community keeps this weight instead of zero."
                },
                "ensemble_size": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.ensemble_size,
                    "min": 1,
                    "description": "Number of truncated Louvain runs on permuted inputs whose partitions vote on the final edge weights. Larger ensembles smooth the weights at proportionally higher cost."
                },
                "max_level": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_level,
                    "min": 1,
                    "description": "Maximum number of hierarchy levels for the final Louvain pass."
                },
                "threshold": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.threshold,
                    "min": 0.0,
                    "description": "Minimum modularity gain for each level of the final Louvain pass; a level continues only while the gain exceeds it."
                },
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0,
                    "description": "Resolution parameter (gamma) in the modularity formula. Higher values produce more, smaller communities; lower values produce fewer, larger ones."
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed,
                    "description": "Seed for the native random-number generator state that permutes the input for each ensemble run. Different seeds produce different ensembles."
                }
            })
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            json!({
                "n_clusters": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.n_clusters,
                    "min": 1,
                    "description": "Number of clusters the k-means stage assigns."
                },
                "n_eigenvectors": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.n_eigenvectors,
                    "min": 1,
                    "description": "Number of leading eigenvectors computed for the spectral embedding that k-means clusters."
                },
                "evs_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.evs_tolerance,
                    "min": 0.0,
                    "description": "Convergence tolerance for the eigenvalue solver. Smaller values compute a more accurate embedding at higher cost."
                },
                "evs_max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.evs_max_iterations,
                    "min": 1,
                    "max": i32::MAX,
                    "description": "Maximum eigenvalue-solver iterations."
                },
                "k_means_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.k_means_tolerance,
                    "min": 0.0,
                    "description": "Convergence tolerance for the k-means stage."
                },
                "k_means_max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.k_means_max_iterations,
                    "min": 1,
                    "max": i32::MAX,
                    "description": "Maximum k-means iterations."
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed,
                    "description": "Seed for the random initialization (Lanczos start vector and k-means centroid seeding). It does not make the partition bit-identical across runs; same-seed replays may land in a different but equally valid assignment."
                }
            })
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => {
            json!({
                "max_iter": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iter,
                    "min": 1,
                    "max": i32::MAX,
                    "description": "Number of layout iterations to run; the layout stops after this many without error. cuGraph suggests 50-100 iterations for quick results and discourages more than 1000."
                },
                "outbound_attraction_distribution": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.outbound_attraction_distribution,
                    "description": "Distributes attraction along outbound edges so hubs attract less and are pushed toward the borders."
                },
                "lin_log_mode": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.lin_log_mode,
                    "description": "Switches the attraction model from lin-lin to lin-log, which makes clusters tighter; scaling_ratio usually needs readjusting."
                },
                "prevent_overlapping": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.prevent_overlapping,
                    "valid_when": "true requires vertex_attributes_table and vertex_radius_col",
                    "description": "Prevents vertices from overlapping, using the per-vertex radius side input."
                },
                "overlap_scaling_ratio": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.overlap_scaling_ratio,
                    "exclusive_min": 0.0,
                    "description": "Scales the repulsion force between two overlapping vertices when prevent_overlapping is true."
                },
                "edge_weight_influence": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.edge_weight_influence,
                    "min": 0.0,
                    "description": "How much edge weights influence attraction: 0 ignores weights, 1 is the normal weighting, and larger values amplify heavy edges."
                },
                "jitter_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.jitter_tolerance,
                    "min": 0.0,
                    "description": "How much swinging is tolerated when adapting speed; lower values give less speed and more precision, and values above 1 are discouraged."
                },
                "barnes_hut_optimize": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.barnes_hut_optimize,
                    "description": "Uses the Barnes-Hut approximation for repulsion instead of the slower exact computation."
                },
                "barnes_hut_theta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.barnes_hut_theta,
                    "min": 0.0,
                    "max": 1.0,
                    "description": "Barnes-Hut trade-off in [0, 1]: 1 favors speed, 0 favors accuracy. Used only when barnes_hut_optimize is true."
                },
                "scaling_ratio": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.scaling_ratio,
                    "exclusive_min": 0.0,
                    "description": "Repulsion strength; larger values spread the layout out. Readjust it when switching lin_log_mode."
                },
                "strong_gravity_mode": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.strong_gravity_mode,
                    "description": "Uses a stronger gravity law that pulls distant vertices toward the center harder; it can dominate the other forces."
                },
                "gravity": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.gravity,
                    "min": 0.0,
                    "description": "Strength of the pull toward the center, which keeps disconnected islands from drifting away."
                },
                "verbose": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.verbose,
                    "description": "Makes cuGraph print convergence information at each iteration."
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed,
                    "description": "Seed for the random initial positions used when initial_positions_table is not supplied. It does not make layouts reproducible; reruns may place individual vertices far apart."
                },
                "vertex_attributes_table": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "description": "Optional relation with exactly one row per graph vertex carrying radius, mobility, and/or mass attributes.",
                    "side_input": {
                        "role": "vertex_attributes",
                        "column_options": ["vertex_attributes_vertex_col", "vertex_radius_col", "vertex_mobility_col", "vertex_mass_col"]
                    }
                },
                "vertex_attributes_vertex_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "vertex_attributes_table",
                    "expected_type_ref": "vertex_domain",
                    "description": "Vertex identifier column in vertex_attributes_table; must match the edge endpoint domain."
                },
                "vertex_radius_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "vertex_attributes_table",
                    "expected_type_ref": "Float32",
                    "description": "Per-vertex radius, used when prevent_overlapping is true."
                },
                "vertex_mobility_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "vertex_attributes_table",
                    "expected_type_ref": "Float32",
                    "description": "Per-vertex mobility: a scaling factor on the vertex's displacement at each iteration."
                },
                "vertex_mass_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "vertex_attributes_table",
                    "expected_type_ref": "Float32",
                    "description": "Per-vertex mass, which controls attraction to other vertices; when omitted, each vertex's mass is its degree plus one."
                },
                "initial_positions_table": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "description": "Optional relation with exactly one warm-start x/y position per graph vertex; when omitted, initial positions are random.",
                    "side_input": {
                        "role": "initial_positions",
                        "column_options": ["initial_positions_vertex_col", "initial_x_col", "initial_y_col"]
                    }
                },
                "initial_positions_vertex_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "initial_positions_table",
                    "expected_type_ref": "vertex_domain",
                    "description": "Vertex identifier column in initial_positions_table; must match the edge endpoint domain."
                },
                "initial_x_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "initial_positions_table",
                    "expected_type_ref": "Float32",
                    "description": "Starting x-axis position per vertex."
                },
                "initial_y_col": {
                    "scope": "side_input",
                    "type": "Utf8|null",
                    "default": null,
                    "column_of": "initial_positions_table",
                    "expected_type_ref": "Float32",
                    "description": "Starting y-axis position per vertex."
                }
            })
        },
        CugraphAlgorithmParams::Similarity(_) => similarity_options_schema_json(),
        CugraphAlgorithmParams::DegreesAll(_)
        | CugraphAlgorithmParams::InDegreesAll(_)
        | CugraphAlgorithmParams::OutDegreesAll(_)
        | CugraphAlgorithmParams::WeaklyConnectedComponents(_)
        | CugraphAlgorithmParams::StronglyConnectedComponents(_)
        | CugraphAlgorithmParams::TriangleCountAll(_)
        | CugraphAlgorithmParams::MinimumSpanningTree(_) => json!({}),
    }
}

fn pagerank_iteration_options(params: &CugraphPageRankParams) -> Map<String, Value> {
    Map::from_iter([
        (
            "alpha".to_string(),
            json!({
                "scope": "algorithm",
                "type": "Float64",
                "default": params.alpha,
                "min": 0.0,
                "max": 1.0,
                "description": "Damping factor: the probability that the random walk follows an outgoing edge instead of restarting. Higher values let distant structure influence scores more; lower values keep rank closer to the restart distribution."
            }),
        ),
        (
            "epsilon".to_string(),
            json!({
                "scope": "algorithm",
                "type": "Float64",
                "default": params.epsilon,
                "exclusive_min": 0.0,
                "description": "Convergence tolerance: iteration stops once the L1 sum of score changes between consecutive iterations is below the vertex count multiplied by epsilon. Smaller values tighten convergence and may need more iterations."
            }),
        ),
        (
            "max_iterations".to_string(),
            json!({
                "scope": "algorithm",
                "type": "UInt32",
                "default": params.max_iterations,
                "min": 1,
                "description": "Upper bound on PageRank iterations. If the bound is reached before epsilon is met, the scores computed so far are returned."
            }),
        ),
    ])
}

fn similarity_options_schema_json() -> Value {
    json!({
        "vertex_pairs_table": {
            "scope": "side_input",
            "type": "Utf8",
            "required": true,
            "description": "Table or view containing the explicit candidate vertex pairs.",
            "side_input": {
                "role": "vertex_pairs",
                "column_options": ["first_vertex_col", "second_vertex_col"]
            }
        },
        "first_vertex_col": {
            "scope": "side_input",
            "type": "Utf8",
            "required": true,
            "description": "First endpoint column in vertex_pairs_table.",
            "column_of": "vertex_pairs_table",
            "expected_type_ref": "vertex_pairs_domain"
        },
        "second_vertex_col": {
            "scope": "side_input",
            "type": "Utf8",
            "required": true,
            "description": "Second endpoint column in vertex_pairs_table.",
            "column_of": "vertex_pairs_table",
            "expected_type_ref": "vertex_pairs_domain"
        }
    })
}

fn betweenness_options_schema_json(
    normalized: bool,
    include_endpoints: Option<bool>,
    exact_vertex_threshold: u64,
) -> Value {
    let mut object = Map::new();
    object.insert(
        "seeds".to_string(),
        json!({
            "scope": "algorithm",
            "type": "List<Int64>|List<Utf8>|null",
            "default": null,
            "mutually_exclusive_with": "k",
            "description": "Explicit homogeneous integer or string seed vertices for approximate betweenness. Null requests exact all-vertex betweenness unless k is set."
        }),
    );
    object.insert(
        "k".to_string(),
        json!({
            "scope": "algorithm",
            "type": "UInt64|null",
            "default": null,
            "min": 1,
            "mutually_exclusive_with": "seeds",
            "description": "Deterministic approximate seed count. Execution uses the first k distinct graph vertices in stable order and refuses k larger than the actual vertex count."
        }),
    );
    object.insert(
        "normalized".to_string(),
        json!({
            "scope": "algorithm",
            "type": "Boolean",
            "default": normalized,
            "description": "When true, scores are scaled by the maximum possible value for the vertex count and directedness so results lie in [0, 1]; when false, raw shortest-path counts are returned."
        }),
    );
    if let Some(include_endpoints) = include_endpoints {
        object.insert(
            "include_endpoints".to_string(),
            json!({
                "scope": "algorithm",
                "type": "Boolean",
                "default": include_endpoints,
                "description": "When true, the two endpoints of each shortest path count as lying on that path; the traditional formulation (false) excludes them."
            }),
        );
    }
    object.insert(
        "exact_vertex_threshold".to_string(),
        json!({
            "scope": "algorithm",
            "type": "UInt64",
            "default": exact_vertex_threshold,
            "description": "Maximum actual graph vertex count allowed for exact betweenness without explicit seeds or k."
        }),
    );
    Value::Object(object)
}

fn bfs_options_schema_json() -> Value {
    json!({
            "depth_limit": {
                "scope": "algorithm",
                "type": "Int64|null",
                "default": null,
                "min": 0,
                "description": "Maximum BFS depth in hops; null means unbounded traversal."
            },
            "source_vertices_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_with": "source_vertex_col",
                "description": "Optional relation containing BFS source vertices. Use source_vertex=NULL in the positional SQL argument when this option is set.",
                "side_input": {
                    "role": "source_vertices",
                    "column_option": "source_vertex_col"
                }
            },
            "source_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_with": "source_vertices_table",
                "description": "Column in source_vertices_table containing BFS source vertex identifiers.",
                "column_of": "source_vertices_table",
                "expected_type_ref": "vertex_domain"
            },
            "output_mode": {
                "scope": "algorithm",
                "type": "Utf8",
                "allowed_values": ["raw", "normalized", "path"],
                "default": "raw",
                "description": "Selects the BFS result shape: raw traversal rows, normalized reachability rows, or one reconstructed source-to-target path."
            },
            "target_vertices_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "output_mode=path or return_target_info=true",
                "description": "Optional relation containing target vertices for path output or target reachability metadata.",
                "side_input": {
                    "role": "target_vertices",
                    "column_option": "target_vertex_col"
                }
            },
            "target_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "output_mode=path or return_target_info=true",
                "description": "Column in target_vertices_table containing target vertex identifiers.",
                "column_of": "target_vertices_table",
                "expected_type_ref": "vertex_domain"
            },
            "return_target_info": {
                "scope": "algorithm",
                "type": "Boolean",
                "default": false,
                "valid_when": "output_mode in [raw, normalized]",
                "description": "Adds target_found and target_distance columns when target_vertices_table is supplied."
            },
            "include_vertices_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "description": "Optional relation containing the vertex allowlist.",
                "side_input": {
                    "role": "include_vertices",
                    "column_option": "include_vertex_col"
                }
            },
            "include_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_vertices_table is set",
                "description": "Column in include_vertices_table containing allowed vertex identifiers.",
                "column_of": "include_vertices_table",
                "expected_type_ref": "vertex_domain"
            },
            "exclude_vertices_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "description": "Optional relation containing the vertex denylist.",
                "side_input": {
                    "role": "exclude_vertices",
                    "column_option": "exclude_vertex_col"
                }
            },
            "exclude_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "exclude_vertices_table is set",
                "description": "Column in exclude_vertices_table containing excluded vertex identifiers.",
                "column_of": "exclude_vertices_table",
                "expected_type_ref": "vertex_domain"
            },
            "edge_id_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "valid_when": "integer vertex domain only",
                "description": "Edge-id column in the edge relation used by edge-id predicate options."
            },
            "include_edge_ids_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_with": "edge_id_col",
                "description": "Optional relation containing allowed edge identifiers.",
                "side_input": {
                    "role": "include_edge_ids",
                    "column_option": "include_edge_ids_col"
                }
            },
            "include_edge_ids_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "defaults_to": "edge_id_col when omitted and include_edge_ids_table is set",
                "description": "Column in include_edge_ids_table containing allowed edge identifiers.",
                "column_of": "include_edge_ids_table",
                "expected_type_ref": "edge_id_domain"
            },
            "include_edges_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "description": "Optional relation containing allowed edge endpoints and, when edge_id_col is configured, optional edge identifiers.",
                "side_input": {
                    "role": "include_edges",
                    "column_options": ["include_edge_src_col", "include_edge_dst_col", "include_edge_id_col"]
                }
            },
            "include_edge_src_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_edges_table is set",
                "description": "Source endpoint column in include_edges_table.",
                "column_of": "include_edges_table",
                "expected_type_ref": "vertex_domain"
            },
            "include_edge_dst_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_edges_table is set",
                "description": "Destination endpoint column in include_edges_table.",
                "column_of": "include_edges_table",
                "expected_type_ref": "vertex_domain"
            },
            "include_edge_id_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "valid_when": "edge_id_col is set",
                "description": "Optional edge-id column in include_edges_table; when omitted, include_edges matching uses only endpoint columns.",
                "column_of": "include_edges_table",
                "expected_type_ref": "edge_id_domain"
            }
    })
}

fn graph_options_value(options: &CugraphGraphOptions) -> Value {
    json!({
        "directed": options.directed,
        "renumber": options.renumber,
        "construction_policy": options.construction_policy.as_str(),
    })
}

pub(super) fn result_schemas_json(algorithm: CugraphAlgorithm) -> String {
    let (default_schema, schemas) = if algorithm == CugraphAlgorithm::Bfs {
        (
            "raw",
            json!({
                "raw": bfs_raw_mode_schema(false),
                "normalized": bfs_normalized_mode_schema(false),
                "raw_with_target_info": bfs_raw_mode_schema(true),
                "normalized_with_target_info": bfs_normalized_mode_schema(true),
                "path": bfs_path_mode_schema()
            }),
        )
    } else {
        let mut schema = schema_json_value(&stable_result_schema(algorithm));
        if let Some(contract) = algorithm.definition().logical_string_contract()
            && let Some(fields) = schema.get_mut("fields").and_then(Value::as_array_mut)
        {
            for field in fields {
                if contract.output_columns.iter().any(|column| field["name"] == column.name) {
                    field["data_type"] = json!("Int64|Utf8");
                }
            }
        }
        ("default", json!({"default": schema}))
    };
    json!({
        "default_schema": default_schema,
        "schemas": schemas,
        "call_specific_authority": "gpu_validate_call",
        "note": "Generic registry schemas; run gpu_validate_call for the concrete table-specific output schema."
    })
    .to_string()
}

fn bfs_raw_mode_schema(with_target_info: bool) -> Value {
    json!({"fields": bfs_raw_mode_fields(with_target_info)})
}

fn bfs_raw_mode_fields(with_target_info: bool) -> Vec<Value> {
    let mut fields = vec![
        metadata_field("vertex", "Int64|Utf8", false),
        metadata_field("distance", "Int64", false),
        metadata_field("predecessor", "Int64|Utf8", true),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    fields
}

fn bfs_normalized_mode_schema(with_target_info: bool) -> Value {
    let mut fields = vec![
        metadata_field("vertex", "Int64|Utf8", false),
        metadata_field("distance", "Int64", true),
        metadata_field("predecessor", "Int64|Utf8", true),
        metadata_field("reachable", "Boolean", false),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    json!({"fields": fields})
}

fn append_bfs_target_info_fields(fields: &mut Vec<Value>, with_target_info: bool) {
    if with_target_info {
        fields.push(metadata_field("target_found", "Boolean", false));
        fields.push(metadata_field("target_distance", "Int64", true));
    }
}

fn bfs_path_mode_schema() -> Value {
    json!({
        "fields": [
            metadata_field("path_index", "Int64", false),
            metadata_field("source", "Int64|Utf8", false),
            metadata_field("target", "Int64|Utf8", false),
            metadata_field("vertex", "Int64|Utf8", false),
            metadata_field("distance", "Int64", false)
        ]
    })
}

fn metadata_field(name: &str, data_type: &str, nullable: bool) -> Value {
    json!({"name": name, "data_type": data_type, "nullable": nullable})
}

pub(super) fn normalized_options_json(spec: &CugraphAlgorithmSpec) -> String {
    json!({
        "src_col": spec.src_col,
        "dst_col": spec.dst_col,
        "weight_col": spec.weight_col,
        "edge_id_col": spec.edge_id_col,
        "graph_options": graph_options_value(&spec.graph_options),
        "algorithm_options": params_json(&spec.params),
    })
    .to_string()
}

fn params_json(params: &CugraphAlgorithmParams) -> Value {
    match params {
        CugraphAlgorithmParams::PageRank(params) => {
            json!({"alpha": params.alpha, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::PersonalizedPageRank(params) => json!({
            "alpha": params.pagerank.alpha,
            "epsilon": params.pagerank.epsilon,
            "max_iterations": params.pagerank.max_iterations,
            "personalization": {
                "table": params.personalization.table_name.to_string(),
                "vertex_col": params.personalization.vertex_col,
                "value_col": params.personalization.value_col,
            },
        }),
        CugraphAlgorithmParams::Bfs(params) => json!({
            "source_vertex": bfs_source_json(&params.source),
            "depth_limit": params.depth_limit,
            "predicates": bfs_predicates_json(&params.predicates),
            "output": {
                "mode": params.output.mode.as_str(),
                "return_target_info": params.output.return_target_info,
            },
        }),
        CugraphAlgorithmParams::Louvain(params) => {
            json!({"max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution})
        },
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            json!({"alpha": params.alpha, "beta": params.beta, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations, "normalize": params.normalize})
        },
        CugraphAlgorithmParams::BetweennessCentrality(params) => {
            json!({"seeds": params.seeds.as_ref().map(|seeds| seeds.vertices.iter().map(vertex_selector_json).collect::<Vec<_>>()), "k": params.k, "normalized": params.normalized, "include_endpoints": params.include_endpoints, "exact_vertex_threshold": params.exact_vertex_threshold})
        },
        CugraphAlgorithmParams::EdgeBetweennessCentrality(params) => {
            json!({"seeds": params.seeds.as_ref().map(|seeds| seeds.vertices.iter().map(vertex_selector_json).collect::<Vec<_>>()), "k": params.k, "normalized": params.normalized, "exact_vertex_threshold": params.exact_vertex_threshold})
        },
        CugraphAlgorithmParams::Sssp(params) => {
            json!({"source_vertex": vertex_selector_json(&params.source_vertex), "cutoff": params.cutoff})
        },
        CugraphAlgorithmParams::CoreNumber(params) => json!({"degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::KCore(params) => json!({"k": params.k, "degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::Leiden(params) => {
            json!({"max_level": params.max_level, "resolution": params.resolution, "theta": params.theta, "seed": params.seed})
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({"min_weight": params.min_weight, "ensemble_size": params.ensemble_size, "max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution, "seed": params.seed})
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params) => {
            json!({"n_clusters": params.n_clusters, "n_eigenvectors": params.n_eigenvectors, "evs_tolerance": params.evs_tolerance, "evs_max_iterations": params.evs_max_iterations, "k_means_tolerance": params.k_means_tolerance, "k_means_max_iterations": params.k_means_max_iterations, "seed": params.seed})
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => {
            json!({
                "max_iter": params.max_iter,
                "outbound_attraction_distribution": params.outbound_attraction_distribution,
                "lin_log_mode": params.lin_log_mode,
                "prevent_overlapping": params.prevent_overlapping,
                "overlap_scaling_ratio": params.overlap_scaling_ratio,
                "edge_weight_influence": params.edge_weight_influence,
                "jitter_tolerance": params.jitter_tolerance,
                "barnes_hut_optimize": params.barnes_hut_optimize,
                "barnes_hut_theta": params.barnes_hut_theta,
                "scaling_ratio": params.scaling_ratio,
                "strong_gravity_mode": params.strong_gravity_mode,
                "gravity": params.gravity,
                "verbose": params.verbose,
                "seed": params.seed,
                "vertex_attributes_table": params.vertex_attributes.as_ref().map(|input| input.table_name.to_string()),
                "vertex_attributes_vertex_col": params.vertex_attributes.as_ref().map(|input| input.vertex_col.as_str()),
                "vertex_radius_col": params.vertex_attributes.as_ref().and_then(|input| input.radius_col.as_deref()),
                "vertex_mobility_col": params.vertex_attributes.as_ref().and_then(|input| input.mobility_col.as_deref()),
                "vertex_mass_col": params.vertex_attributes.as_ref().and_then(|input| input.mass_col.as_deref()),
                "initial_positions_table": params.initial_positions.as_ref().map(|input| input.table_name.to_string()),
                "initial_positions_vertex_col": params.initial_positions.as_ref().map(|input| input.vertex_col.as_str()),
                "initial_x_col": params.initial_positions.as_ref().map(|input| input.x_col.as_str()),
                "initial_y_col": params.initial_positions.as_ref().map(|input| input.y_col.as_str())
            })
        },
        CugraphAlgorithmParams::Similarity(params) => json!({
            "vertex_pairs": {
                "table": params.vertex_pairs.table_name.to_string(),
                "first_vertex_col": params.vertex_pairs.first_vertex_col,
                "second_vertex_col": params.vertex_pairs.second_vertex_col,
            }
        }),
        CugraphAlgorithmParams::DegreesAll(_)
        | CugraphAlgorithmParams::InDegreesAll(_)
        | CugraphAlgorithmParams::OutDegreesAll(_)
        | CugraphAlgorithmParams::WeaklyConnectedComponents(_)
        | CugraphAlgorithmParams::StronglyConnectedComponents(_)
        | CugraphAlgorithmParams::TriangleCountAll(_)
        | CugraphAlgorithmParams::MinimumSpanningTree(_) => json!({}),
    }
}

fn bfs_predicates_json(predicates: &CugraphBfsPredicateInputs) -> Value {
    json!({
        "include_vertices": predicates.include_vertices.as_ref().map(|input| {
            json!({"table": input.table_name.to_string(), "column": input.column})
        }),
        "exclude_vertices": predicates.exclude_vertices.as_ref().map(|input| {
            json!({"table": input.table_name.to_string(), "column": input.column})
        }),
        "target_vertices": predicates.target_vertices.as_ref().map(|input| {
            json!({"table": input.table_name.to_string(), "column": input.column})
        }),
        "include_edge_ids": predicates.include_edge_ids.as_ref().map(|input| {
            json!({"table": input.table_name.to_string(), "column": input.column})
        }),
        "include_edges": predicates.include_edges.as_ref().map(|input| {
            json!({
                "table": input.table_name.to_string(),
                "source_col": input.source_col,
                "destination_col": input.destination_col,
                "edge_id_col": input.edge_id_col,
            })
        }),
    })
}

fn bfs_source_json(source: &CugraphBfsSource) -> Value {
    match source {
        CugraphBfsSource::Scalar(source) => source_vertex_json(source),
        CugraphBfsSource::LiteralList(values) => Value::Array(values.iter().map(source_vertex_json).collect()),
        CugraphBfsSource::Relation(input) => json!({
            "table": input.table_name.to_string(),
            "column": input.column,
        }),
    }
}

fn source_vertex_json(source_vertex: &CugraphBfsSourceVertex) -> Value {
    match source_vertex {
        CugraphBfsSourceVertex::Int64(value) => json!(value),
        CugraphBfsSourceVertex::Utf8(value) => json!(value),
    }
}

fn vertex_selector_json(selector: &CugraphVertexSelector) -> Value {
    match selector {
        CugraphVertexSelector::Int64(value) => json!(value),
        CugraphVertexSelector::Utf8(value) => json!(value),
    }
}

pub(super) fn data_type_label(data_type: &DataType) -> String {
    match data_type {
        DataType::Int32 => "Int32".to_string(),
        DataType::Int64 => "Int64".to_string(),
        DataType::Float32 => "Float32".to_string(),
        DataType::Float64 => "Float64".to_string(),
        DataType::Utf8 => "Utf8".to_string(),
        DataType::LargeUtf8 => "LargeUtf8".to_string(),
        DataType::Utf8View => "Utf8View".to_string(),
        DataType::Boolean => "Boolean".to_string(),
        other => format!("{other:?}"),
    }
}
