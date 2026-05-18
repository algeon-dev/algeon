use datafusion::{
    common::{Result, ScalarValue},
    logical_expr::Expr,
};
use serde_json::Value;

use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsParams, CugraphGraphConstructionPolicy,
    CugraphGraphOptions, CugraphLouvainParams, CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference,
    plan_error, stable_result_schema,
};

pub fn parse_cugraph_algorithm_spec(
    algorithm: CugraphAlgorithm,
    args: &[Expr],
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    if !config.allows(algorithm) {
        return Err(plan_error(format!("cuGraph algorithm {:?} is not enabled in this session", algorithm)));
    }

    validate_arity(algorithm, args.len())?;
    let table_name = CugraphTableReference::parse(&literal_string_arg(args, 0, "table_name")?)?;

    let mut next = 1;
    let source_vertex = if algorithm == CugraphAlgorithm::Bfs {
        let value = literal_i64_arg(args, next, "source_vertex")?;
        next += 1;
        Some(value)
    } else {
        None
    };

    let (src_col, dst_col) = if args.len() >= next + 2 {
        let src = literal_string_arg(args, next, "src_col")?;
        let dst = literal_string_arg(args, next + 1, "dst_col")?;
        next += 2;
        (src, dst)
    } else {
        ("src".to_string(), "dst".to_string())
    };

    let weight_col = if args.len() > next {
        let weight = literal_optional_string_arg(args, next, "weight_col")?;
        next += 1;
        weight
    } else {
        None
    };

    let mut graph_options = config.default_graph_options.clone();
    let mut params = default_params(algorithm, source_vertex);
    if args.len() > next {
        let options_json = literal_string_arg(args, next, "options_json")?;
        parse_options_json(algorithm, &options_json, &mut graph_options, &mut params)?;
    }

    Ok(CugraphAlgorithmSpec {
        algorithm,
        table_name,
        src_col,
        dst_col,
        weight_col,
        graph_options,
        params,
        output_schema: stable_result_schema(algorithm),
    })
}

fn validate_arity(algorithm: CugraphAlgorithm, len: usize) -> Result<()> {
    let valid = match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::Louvain => matches!(len, 1 | 3 | 4 | 5),
        CugraphAlgorithm::Bfs => matches!(len, 2 | 4 | 5 | 6),
    };
    if valid {
        Ok(())
    } else {
        Err(plan_error(format!(
            "{} received {len} arguments, but expected {}",
            algorithm.function_name(),
            expected_arity(algorithm)
        )))
    }
}

fn expected_arity(algorithm: CugraphAlgorithm) -> &'static str {
    match algorithm {
        CugraphAlgorithm::PageRank | CugraphAlgorithm::Louvain => {
            "table_name [, src_col, dst_col, weight_col, options_json]"
        },
        CugraphAlgorithm::Bfs => "table_name, source_vertex [, src_col, dst_col, weight_col, options_json]",
    }
}

fn default_params(algorithm: CugraphAlgorithm, source_vertex: Option<i64>) -> CugraphAlgorithmParams {
    match algorithm {
        CugraphAlgorithm::PageRank => CugraphAlgorithmParams::PageRank(CugraphPageRankParams::default()),
        CugraphAlgorithm::Bfs => CugraphAlgorithmParams::Bfs(CugraphBfsParams {
            source_vertex: source_vertex.expect("BFS arity validation requires source_vertex"),
            depth_limit: None,
        }),
        CugraphAlgorithm::Louvain => CugraphAlgorithmParams::Louvain(CugraphLouvainParams::default()),
    }
}

fn parse_options_json(
    algorithm: CugraphAlgorithm,
    options_json: &str,
    graph_options: &mut CugraphGraphOptions,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    let value = serde_json::from_str::<Value>(options_json)
        .map_err(|error| plan_error(format!("invalid cuGraph options JSON: {error}")))?;
    let object = value.as_object().ok_or_else(|| plan_error("cuGraph options JSON must be an object"))?;

    for (key, value) in object {
        match key.as_str() {
            "directed" => graph_options.directed = json_bool(value, key)?,
            "renumber" => graph_options.renumber = json_bool(value, key)?,
            "construction_policy" => {
                let policy = json_string(value, key)?;
                graph_options.construction_policy = match policy.as_str() {
                    "python_cugraph" => CugraphGraphConstructionPolicy::PythonCugraph,
                    "raw_libcugraph" => CugraphGraphConstructionPolicy::RawLibcugraph,
                    other => {
                        return Err(plan_error(format!(
                            "unsupported cuGraph construction_policy '{other}', expected 'python_cugraph' or 'raw_libcugraph'"
                        )));
                    },
                };
            },
            "alpha" if algorithm == CugraphAlgorithm::PageRank => {
                let alpha = json_f64(value, key)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(plan_error("PageRank alpha must be in [0, 1]"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.alpha = alpha;
                }
            },
            "epsilon" if algorithm == CugraphAlgorithm::PageRank => {
                let epsilon = json_f64(value, key)?;
                if epsilon <= 0.0 {
                    return Err(plan_error("PageRank epsilon must be positive"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.epsilon = epsilon;
                }
            },
            "max_iterations" if algorithm == CugraphAlgorithm::PageRank => {
                let max_iterations = json_u32(value, key)?;
                if max_iterations < 1 {
                    return Err(plan_error("PageRank max_iterations must be at least 1"));
                }
                if let CugraphAlgorithmParams::PageRank(params) = params {
                    params.max_iterations = max_iterations;
                }
            },
            "depth_limit" if algorithm == CugraphAlgorithm::Bfs => {
                let depth_limit = json_i64(value, key)?;
                if depth_limit < 0 {
                    return Err(plan_error("BFS depth_limit must be non-negative"));
                }
                if let CugraphAlgorithmParams::Bfs(params) = params {
                    params.depth_limit = Some(depth_limit);
                }
            },
            "resolution" if algorithm == CugraphAlgorithm::Louvain => {
                let resolution = json_f64(value, key)?;
                if resolution <= 0.0 {
                    return Err(plan_error("Louvain resolution must be positive"));
                }
                if let CugraphAlgorithmParams::Louvain(params) = params {
                    params.resolution = resolution;
                }
            },
            _ => {
                return Err(plan_error(format!("unknown option '{key}' for {}", algorithm.function_name())));
            },
        }
    }

    Ok(())
}

fn literal_string_arg(args: &[Expr], index: usize, name: &str) -> Result<String> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Utf8(Some(value))
            | ScalarValue::LargeUtf8(Some(value))
            | ScalarValue::Utf8View(Some(value)) => Ok(value.clone()),
            _ => Err(plan_error(format!("{name} must be a non-null string literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn literal_optional_string_arg(args: &[Expr], index: usize, name: &str) -> Result<Option<String>> {
    match args.get(index) {
        Some(Expr::Literal(ScalarValue::Null, _)) => Ok(None),
        Some(Expr::Literal(ScalarValue::Utf8(None), _))
        | Some(Expr::Literal(ScalarValue::LargeUtf8(None), _))
        | Some(Expr::Literal(ScalarValue::Utf8View(None), _)) => Ok(None),
        Some(_) => literal_string_arg(args, index, name).map(Some),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn literal_i64_arg(args: &[Expr], index: usize, name: &str) -> Result<i64> {
    match args.get(index) {
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::Int8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::Int64(Some(value)) => Ok(*value),
            ScalarValue::UInt8(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt16(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt32(Some(value)) => Ok(i64::from(*value)),
            ScalarValue::UInt64(Some(value)) => {
                i64::try_from(*value).map_err(|_| plan_error(format!("{name} is too large for an Int64 source vertex")))
            },
            _ => Err(plan_error(format!("{name} must be a non-null integer literal"))),
        },
        Some(_) => Err(plan_error(format!("{name} must be a literal"))),
        None => Err(plan_error(format!("missing {name} argument"))),
    }
}

fn json_bool(value: &Value, key: &str) -> Result<bool> {
    value.as_bool().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a boolean")))
}

fn json_string(value: &Value, key: &str) -> Result<String> {
    value.as_str().map(ToOwned::to_owned).ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a string")))
}

fn json_f64(value: &Value, key: &str) -> Result<f64> {
    let parsed = value.as_f64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be a number")))?;
    if parsed.is_finite() { Ok(parsed) } else { Err(plan_error(format!("cuGraph option '{key}' must be finite"))) }
}

fn json_i64(value: &Value, key: &str) -> Result<i64> {
    value.as_i64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an integer")))
}

fn json_u32(value: &Value, key: &str) -> Result<u32> {
    let parsed =
        value.as_u64().ok_or_else(|| plan_error(format!("cuGraph option '{key}' must be an unsigned integer")))?;
    u32::try_from(parsed).map_err(|_| plan_error(format!("cuGraph option '{key}' exceeds u32::MAX")))
}
