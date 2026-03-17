use arrow::datatypes::{DataType, SchemaRef};
use serde_json::{Value, json};

use super::super::{
    CUGRAPH_VALIDATE_CALL_FUNCTION, CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec,
    CugraphBfsPredicateInputs, CugraphBfsSourceVertex, CugraphGraphOptions, CugraphSqlConfig,
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

pub(super) fn options_schema_json(algorithm: CugraphAlgorithm) -> String {
    let mut schema = json!({
        "graph_options": graph_options_schema_json(),
        "algorithm_options": algorithm_options_schema_json(algorithm)
    });
    if algorithm == CugraphAlgorithm::Bfs {
        schema.as_object_mut().expect("options schema is an object").insert(
            "output_schema_authority".to_string(),
            json!({
                "default_schema_field": "output_schema_json",
                "mode_schema_field": "mode_schemas_json",
                "call_specific_authority": CUGRAPH_VALIDATE_CALL_FUNCTION,
                "note": "output_schema_json is the concrete default/raw schema; inspect mode_schemas_json or run cugraph_validate_call for mode-specific and table-specific output schemas."
            }),
        );
    }
    schema.to_string()
}

fn graph_options_schema_json() -> Value {
    json!({
        "directed": {
            "scope": "graph",
            "type": "Boolean",
            "default": "session default",
            "description": "Whether graph construction treats edges as directed."
        },
        "renumber": {
            "scope": "graph",
            "type": "Boolean",
            "default": "session default",
            "description": "Whether graph construction may renumber external vertex identifiers internally."
        },
        "construction_policy": {
            "scope": "graph",
            "type": "Utf8",
            "allowed_values": ["python_cugraph", "raw_libcugraph"],
            "default": "session default",
            "description": "Backend policy used to construct the cuGraph graph from the edge relation."
        }
    })
}

fn algorithm_options_schema_json(algorithm: CugraphAlgorithm) -> Value {
    match CugraphAlgorithmParams::from_native_defaults(algorithm, None) {
        CugraphAlgorithmParams::PageRank(params) | CugraphAlgorithmParams::PageRankStrict(params) => {
            json!({
                "alpha": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.alpha,
                    "min": 0.0,
                    "max": 1.0
                },
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1
                }
            })
        },
        CugraphAlgorithmParams::Bfs(_) => bfs_options_schema_json(),
        CugraphAlgorithmParams::Louvain(params) => {
            json!({
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0
                }
            })
        },
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1
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
                    "max": 1.0
                },
                "beta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.beta,
                    "exclusive_min": 0.0
                },
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1
                }
            })
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({
                "epsilon": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.epsilon,
                    "exclusive_min": 0.0
                },
                "max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_iterations,
                    "min": 1
                },
                "normalize": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.normalize
                }
            })
        },
        CugraphAlgorithmParams::Sssp(params) => json!({
            "cutoff": {
                "scope": "algorithm",
                "type": "Float64|null",
                "default": params.cutoff,
                "min": 0.0
            }
        }),
        CugraphAlgorithmParams::CoreNumber(params) => {
            json!({
                "degree_type": {
                    "scope": "algorithm",
                    "type": "Utf8",
                    "allowed_values": ["in", "out", "in_out"],
                    "default": params.degree_type.as_str()
                }
            })
        },
        CugraphAlgorithmParams::KCore(params) => {
            json!({
                "k": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.k,
                    "min": 1
                },
                "degree_type": {
                    "scope": "algorithm",
                    "type": "Utf8",
                    "allowed_values": ["in", "out", "in_out"],
                    "default": params.degree_type.as_str()
                }
            })
        },
        CugraphAlgorithmParams::Leiden(params) => {
            json!({
                "max_level": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_level,
                    "min": 1
                },
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0
                },
                "theta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.theta,
                    "exclusive_min": 0.0
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed
                }
            })
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({
                "min_weight": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.min_weight,
                    "min": 0.0
                },
                "ensemble_size": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.ensemble_size,
                    "min": 1
                },
                "max_level": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.max_level,
                    "min": 1
                },
                "threshold": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.threshold,
                    "min": 0.0
                },
                "resolution": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.resolution,
                    "exclusive_min": 0.0
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed
                }
            })
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params)
        | CugraphAlgorithmParams::BalancedCutClustering(params) => {
            json!({
                "n_clusters": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.n_clusters,
                    "min": 1
                },
                "n_eigenvectors": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.n_eigenvectors,
                    "min": 1
                },
                "evs_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.evs_tolerance,
                    "min": 0.0
                },
                "evs_max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.evs_max_iterations,
                    "min": 1
                },
                "k_means_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.k_means_tolerance,
                    "min": 0.0
                },
                "k_means_max_iterations": {
                    "scope": "algorithm",
                    "type": "UInt32",
                    "default": params.k_means_max_iterations,
                    "min": 1
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed
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
                    "max": i32::MAX
                },
                "outbound_attraction_distribution": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.outbound_attraction_distribution
                },
                "lin_log_mode": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.lin_log_mode
                },
                "prevent_overlapping": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.prevent_overlapping,
                    "allowed_values": [false],
                    "valid_when": "unsupported until vertex_radius side inputs are exposed by the SQL API"
                },
                "overlap_scaling_ratio": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.overlap_scaling_ratio,
                    "exclusive_min": 0.0
                },
                "edge_weight_influence": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.edge_weight_influence,
                    "min": 0.0
                },
                "jitter_tolerance": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.jitter_tolerance,
                    "min": 0.0
                },
                "barnes_hut_optimize": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.barnes_hut_optimize
                },
                "barnes_hut_theta": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.barnes_hut_theta,
                    "min": 0.0,
                    "max": 1.0
                },
                "scaling_ratio": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.scaling_ratio,
                    "exclusive_min": 0.0
                },
                "strong_gravity_mode": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.strong_gravity_mode
                },
                "gravity": {
                    "scope": "algorithm",
                    "type": "Float64",
                    "default": params.gravity,
                    "min": 0.0
                },
                "verbose": {
                    "scope": "algorithm",
                    "type": "Boolean",
                    "default": params.verbose
                },
                "seed": {
                    "scope": "algorithm",
                    "type": "UInt64",
                    "default": params.seed
                }
            })
        },
        CugraphAlgorithmParams::DegreesAll(_)
        | CugraphAlgorithmParams::InDegreesAll(_)
        | CugraphAlgorithmParams::OutDegreesAll(_)
        | CugraphAlgorithmParams::WeaklyConnectedComponents(_)
        | CugraphAlgorithmParams::StronglyConnectedComponents(_)
        | CugraphAlgorithmParams::TriangleCountAll(_)
        | CugraphAlgorithmParams::MinimumSpanningTree(_) => json!({}),
    }
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
                    "column_option": "target_vertex_col",
                    "expected_columns": [
                        {
                            "name_from_option": "target_vertex_col",
                            "type": "same logical vertex domain as src_col and dst_col"
                        }
                    ]
                }
            },
            "target_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "output_mode=path or return_target_info=true",
                "description": "Column in target_vertices_table containing target vertex identifiers.",
                "column_of": "target_vertices_table",
                "expected_type": "same logical vertex domain as src_col and dst_col"
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
                    "column_option": "include_vertex_col",
                    "expected_columns": [
                        {
                            "name_from_option": "include_vertex_col",
                            "type": "same logical vertex domain as src_col and dst_col"
                        }
                    ]
                }
            },
            "include_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_vertices_table is set",
                "description": "Column in include_vertices_table containing allowed vertex identifiers.",
                "column_of": "include_vertices_table",
                "expected_type": "same logical vertex domain as src_col and dst_col"
            },
            "exclude_vertices_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "description": "Optional relation containing the vertex denylist.",
                "side_input": {
                    "role": "exclude_vertices",
                    "column_option": "exclude_vertex_col",
                    "expected_columns": [
                        {
                            "name_from_option": "exclude_vertex_col",
                            "type": "same logical vertex domain as src_col and dst_col"
                        }
                    ]
                }
            },
            "exclude_vertex_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "exclude_vertices_table is set",
                "description": "Column in exclude_vertices_table containing excluded vertex identifiers.",
                "column_of": "exclude_vertices_table",
                "expected_type": "same logical vertex domain as src_col and dst_col"
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
                    "column_option": "include_edge_ids_col",
                    "expected_columns": [
                        {
                            "name_from_option": "include_edge_ids_col",
                            "type": "same integer domain as edge_id_col"
                        }
                    ]
                }
            },
            "include_edge_ids_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "defaults_to": "edge_id_col when omitted and include_edge_ids_table is set",
                "description": "Column in include_edge_ids_table containing allowed edge identifiers.",
                "column_of": "include_edge_ids_table",
                "expected_type": "same integer domain as edge_id_col"
            },
            "include_edges_table": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "description": "Optional relation containing allowed edge endpoints and, when edge_id_col is configured, optional edge identifiers.",
                "side_input": {
                    "role": "include_edges",
                    "column_options": ["include_edge_src_col", "include_edge_dst_col", "include_edge_id_col"],
                    "expected_columns": [
                        {
                            "name_from_option": "include_edge_src_col",
                            "type": "same logical vertex domain as src_col and dst_col"
                        },
                        {
                            "name_from_option": "include_edge_dst_col",
                            "type": "same logical vertex domain as src_col and dst_col"
                        },
                        {
                            "name_from_option": "include_edge_id_col",
                            "type": "same integer domain as edge_id_col",
                            "optional_when": "edge_id_col is set"
                        }
                    ]
                }
            },
            "include_edge_src_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_edges_table is set",
                "description": "Source endpoint column in include_edges_table.",
                "column_of": "include_edges_table",
                "expected_type": "same logical vertex domain as src_col and dst_col"
            },
            "include_edge_dst_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "required_when": "include_edges_table is set",
                "description": "Destination endpoint column in include_edges_table.",
                "column_of": "include_edges_table",
                "expected_type": "same logical vertex domain as src_col and dst_col"
            },
            "include_edge_id_col": {
                "scope": "algorithm",
                "type": "Utf8|null",
                "default": null,
                "valid_when": "edge_id_col is set",
                "description": "Optional edge-id column in include_edges_table; when omitted, include_edges matching uses only endpoint columns.",
                "column_of": "include_edges_table",
                "expected_type": "same integer domain as edge_id_col"
            }
    })
}

pub(super) fn option_defaults_json(algorithm: CugraphAlgorithm, config: &CugraphSqlConfig) -> String {
    let graph_options = algorithm.effective_default_graph_options(&config.default_graph_options);
    json!({
        "graph_options": graph_options_value(&graph_options),
        "algorithm_options": algorithm_option_defaults(algorithm)
    })
    .to_string()
}

pub(super) fn graph_options_json(options: &CugraphGraphOptions) -> String {
    graph_options_value(options).to_string()
}

fn graph_options_value(options: &CugraphGraphOptions) -> Value {
    json!({
        "directed": options.directed,
        "renumber": options.renumber,
        "construction_policy": options.construction_policy.as_str(),
    })
}

fn algorithm_option_defaults(algorithm: CugraphAlgorithm) -> Value {
    match CugraphAlgorithmParams::from_native_defaults(algorithm, None) {
        CugraphAlgorithmParams::PageRank(params) | CugraphAlgorithmParams::PageRankStrict(params) => {
            json!({"alpha": params.alpha, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Bfs(params) => json!({
            "depth_limit": params.depth_limit,
            "edge_id_col": null,
            "include_vertices_table": null,
            "include_vertex_col": null,
            "exclude_vertices_table": null,
            "exclude_vertex_col": null,
            "target_vertices_table": null,
            "target_vertex_col": null,
            "include_edge_ids_table": null,
            "include_edge_ids_col": null,
            "include_edges_table": null,
            "include_edge_src_col": null,
            "include_edge_dst_col": null,
            "include_edge_id_col": null,
            "output_mode": params.output.mode.as_str(),
            "return_target_info": params.output.return_target_info,
        }),
        CugraphAlgorithmParams::Louvain(params) => json!({"resolution": params.resolution}),
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            json!({"alpha": params.alpha, "beta": params.beta, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations, "normalize": params.normalize})
        },
        CugraphAlgorithmParams::Sssp(params) => json!({"cutoff": params.cutoff}),
        CugraphAlgorithmParams::CoreNumber(params) => json!({"degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::KCore(params) => json!({"k": params.k, "degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::Leiden(params) => {
            json!({"max_level": params.max_level, "resolution": params.resolution, "theta": params.theta, "seed": params.seed})
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({"min_weight": params.min_weight, "ensemble_size": params.ensemble_size, "max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution, "seed": params.seed})
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params)
        | CugraphAlgorithmParams::BalancedCutClustering(params) => {
            json!({"n_clusters": params.n_clusters, "n_eigenvectors": params.n_eigenvectors, "evs_tolerance": params.evs_tolerance, "evs_max_iterations": params.evs_max_iterations, "k_means_tolerance": params.k_means_tolerance, "k_means_max_iterations": params.k_means_max_iterations, "seed": params.seed})
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => {
            json!({"max_iter": params.max_iter, "outbound_attraction_distribution": params.outbound_attraction_distribution, "lin_log_mode": params.lin_log_mode, "prevent_overlapping": params.prevent_overlapping, "overlap_scaling_ratio": params.overlap_scaling_ratio, "edge_weight_influence": params.edge_weight_influence, "jitter_tolerance": params.jitter_tolerance, "barnes_hut_optimize": params.barnes_hut_optimize, "barnes_hut_theta": params.barnes_hut_theta, "scaling_ratio": params.scaling_ratio, "strong_gravity_mode": params.strong_gravity_mode, "gravity": params.gravity, "verbose": params.verbose, "seed": params.seed})
        },
        CugraphAlgorithmParams::DegreesAll(_)
        | CugraphAlgorithmParams::InDegreesAll(_)
        | CugraphAlgorithmParams::OutDegreesAll(_)
        | CugraphAlgorithmParams::WeaklyConnectedComponents(_)
        | CugraphAlgorithmParams::StronglyConnectedComponents(_)
        | CugraphAlgorithmParams::TriangleCountAll(_)
        | CugraphAlgorithmParams::MinimumSpanningTree(_) => json!({}),
    }
}

pub(super) fn mode_schemas_json(algorithm: CugraphAlgorithm) -> String {
    if algorithm != CugraphAlgorithm::Bfs {
        return "{}".to_string();
    }
    json!({
        "raw": bfs_raw_mode_schema(false),
        "normalized": bfs_normalized_mode_schema(false),
        "raw_with_target_info": bfs_raw_mode_schema(true),
        "normalized_with_target_info": bfs_normalized_mode_schema(true),
        "path": bfs_path_mode_schema()
    })
    .to_string()
}

fn bfs_raw_mode_schema(with_target_info: bool) -> Value {
    let mut fields = vec![
        json!({"name": "vertex", "type": "Int64|Utf8", "nullable": false}),
        json!({"name": "distance", "type": "Int64", "nullable": false}),
        json!({"name": "predecessor", "type": "Int64|Utf8", "nullable": true}),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    Value::Array(fields)
}

fn bfs_normalized_mode_schema(with_target_info: bool) -> Value {
    let mut fields = vec![
        json!({"name": "vertex", "type": "Int64|Utf8", "nullable": false}),
        json!({"name": "distance", "type": "Int64", "nullable": true}),
        json!({"name": "predecessor", "type": "Int64|Utf8", "nullable": true}),
        json!({"name": "reachable", "type": "Boolean", "nullable": false}),
    ];
    append_bfs_target_info_fields(&mut fields, with_target_info);
    Value::Array(fields)
}

fn append_bfs_target_info_fields(fields: &mut Vec<Value>, with_target_info: bool) {
    if with_target_info {
        fields.push(json!({"name": "target_found", "type": "Boolean", "nullable": false}));
        fields.push(json!({"name": "target_distance", "type": "Int64", "nullable": true}));
    }
}

fn bfs_path_mode_schema() -> Value {
    json!([
        {"name": "path_index", "type": "Int64", "nullable": false},
        {"name": "source", "type": "Int64|Utf8", "nullable": false},
        {"name": "target", "type": "Int64|Utf8", "nullable": false},
        {"name": "vertex", "type": "Int64|Utf8", "nullable": false},
        {"name": "distance", "type": "Int64", "nullable": false}
    ])
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
        CugraphAlgorithmParams::PageRank(params) | CugraphAlgorithmParams::PageRankStrict(params) => {
            json!({"alpha": params.alpha, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Bfs(params) => json!({
            "source_vertex": source_vertex_json(&params.source_vertex),
            "depth_limit": params.depth_limit,
            "predicates": bfs_predicates_json(&params.predicates),
            "output": {
                "mode": params.output.mode.as_str(),
                "return_target_info": params.output.return_target_info,
            },
        }),
        CugraphAlgorithmParams::Louvain(params) => json!({"resolution": params.resolution}),
        CugraphAlgorithmParams::EigenvectorCentrality(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::KatzCentrality(params) => {
            json!({"alpha": params.alpha, "beta": params.beta, "epsilon": params.epsilon, "max_iterations": params.max_iterations})
        },
        CugraphAlgorithmParams::Hits(params) => {
            json!({"epsilon": params.epsilon, "max_iterations": params.max_iterations, "normalize": params.normalize})
        },
        CugraphAlgorithmParams::Sssp(params) => json!({"source_vertex": params.source_vertex, "cutoff": params.cutoff}),
        CugraphAlgorithmParams::CoreNumber(params) => json!({"degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::KCore(params) => json!({"k": params.k, "degree_type": params.degree_type.as_str()}),
        CugraphAlgorithmParams::Leiden(params) => {
            json!({"max_level": params.max_level, "resolution": params.resolution, "theta": params.theta, "seed": params.seed})
        },
        CugraphAlgorithmParams::Ecg(params) => {
            json!({"min_weight": params.min_weight, "ensemble_size": params.ensemble_size, "max_level": params.max_level, "threshold": params.threshold, "resolution": params.resolution, "seed": params.seed})
        },
        CugraphAlgorithmParams::SpectralModularityMaximization(params)
        | CugraphAlgorithmParams::BalancedCutClustering(params) => {
            json!({"n_clusters": params.n_clusters, "n_eigenvectors": params.n_eigenvectors, "evs_tolerance": params.evs_tolerance, "evs_max_iterations": params.evs_max_iterations, "k_means_tolerance": params.k_means_tolerance, "k_means_max_iterations": params.k_means_max_iterations, "seed": params.seed})
        },
        CugraphAlgorithmParams::ForceAtlas2(params) => {
            json!({"max_iter": params.max_iter, "outbound_attraction_distribution": params.outbound_attraction_distribution, "lin_log_mode": params.lin_log_mode, "prevent_overlapping": params.prevent_overlapping, "overlap_scaling_ratio": params.overlap_scaling_ratio, "edge_weight_influence": params.edge_weight_influence, "jitter_tolerance": params.jitter_tolerance, "barnes_hut_optimize": params.barnes_hut_optimize, "barnes_hut_theta": params.barnes_hut_theta, "scaling_ratio": params.scaling_ratio, "strong_gravity_mode": params.strong_gravity_mode, "gravity": params.gravity, "verbose": params.verbose, "seed": params.seed})
        },
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

fn source_vertex_json(source_vertex: &CugraphBfsSourceVertex) -> Value {
    match source_vertex {
        CugraphBfsSourceVertex::Int64(value) => json!(value),
        CugraphBfsSourceVertex::Utf8(value) => json!(value),
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
