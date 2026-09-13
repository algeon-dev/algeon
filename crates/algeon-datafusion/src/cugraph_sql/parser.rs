use arrow::{
    array::{Array, AsArray},
    datatypes::DataType,
};
use datafusion::{
    common::{Result, ScalarValue},
    logical_expr::Expr,
};

#[cfg(test)]
use super::CugraphVertexSelector;
use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsSource, CugraphBfsSourceVertex,
    CugraphGraphPredicateTable, CugraphSqlConfig, CugraphTableReference, options, plan_error, result_schema_for_params,
};

pub fn parse_cugraph_algorithm_spec(
    algorithm: CugraphAlgorithm,
    args: &[Expr],
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    // Keep the required-weight contract actionable even when the caller did
    // not supply enough positional arguments to reach `weight_col`.
    if algorithm.native_algorithm().weight_policy().requires_weight() && args.len() < 4 {
        return Err(plan_error(format!(
            "{} requires a weight column; supply one as the fourth argument",
            algorithm.function_name()
        )));
    }
    validate_arity(algorithm, args.len())?;
    let table_name = CugraphTableReference::parse(&literal_string_arg(args, 0, "table_name")?)?;

    let mut next = 1;
    let source = if algorithm.requires_source_vertex() {
        let value = literal_source_arg(args, next, algorithm)?;
        next += 1;
        value
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

    let (mut graph_options, mut params) = options::default_options(algorithm, config, source.clone());
    let mut edge_id_col = None;
    let has_options_json = args.len() > next;
    let mut source_relation = None;
    if args.len() > next {
        let options_json = literal_string_arg(args, next, "options_json")?;
        source_relation =
            options::parse_options_json(algorithm, &options_json, &mut graph_options, &mut edge_id_col, &mut params)?;
    }
    apply_source_arg(algorithm, source, source_relation, &mut params)?;
    if algorithm == CugraphAlgorithm::PersonalizedPageRank && !has_options_json {
        return Err(plan_error(
            "cugraph_personalized_pagerank requires options: personalization_table, personalization_vertex_col, personalization_value_col",
        ));
    }
    if algorithm.is_similarity() && !has_options_json {
        return Err(plan_error(format!(
            "{} requires options: vertex_pairs_table, first_vertex_col, second_vertex_col",
            algorithm.function_name()
        )));
    }
    options::validate_graph_directedness(algorithm, &graph_options)?;

    // MST is a weighted algorithm: require an explicit weight column so
    // the graph is built with edge weights rather than silently producing
    // a meaningless unweighted result.
    if algorithm.native_algorithm().weight_policy().requires_weight() && weight_col.is_none() {
        return Err(plan_error(
            "cugraph_minimum_spanning_tree requires a weight column; supply one as the fourth argument",
        ));
    }

    let output_schema = result_schema_for_params(algorithm, &params);
    Ok(CugraphAlgorithmSpec {
        algorithm,
        table_name,
        src_col,
        dst_col,
        weight_col,
        edge_id_col,
        graph_options,
        params,
        output_schema,
    })
}

fn validate_arity(algorithm: CugraphAlgorithm, len: usize) -> Result<()> {
    if algorithm.allowed_arg_counts().contains(&len) {
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
    if algorithm == CugraphAlgorithm::MinimumSpanningTree {
        "table_name, src_col, dst_col, weight_col [, options_json]"
    } else if algorithm.is_similarity() {
        "table_name, src_col, dst_col, weight_col, options_json"
    } else if algorithm == CugraphAlgorithm::Bfs {
        "table_name, source_vertex_or_vertices [, src_col, dst_col, weight_col, options_json]"
    } else if algorithm.requires_source_vertex() {
        "table_name, source_vertex [, src_col, dst_col, weight_col, options_json]"
    } else {
        "table_name [, src_col, dst_col, weight_col, options_json]"
    }
}

fn apply_source_arg(
    algorithm: CugraphAlgorithm,
    source: Option<CugraphBfsSource>,
    source_relation: Option<CugraphGraphPredicateTable>,
    params: &mut CugraphAlgorithmParams,
) -> Result<()> {
    match algorithm {
        CugraphAlgorithm::Bfs => {
            let CugraphAlgorithmParams::Bfs(params) = params else {
                return Err(plan_error("cugraph_bfs carried non-BFS parameters"));
            };
            match (source, source_relation) {
                (Some(_), Some(_)) => Err(plan_error(
                    "BFS source_vertex_or_vertices cannot be combined with source_vertices_table/source_vertex_col options",
                )),
                (Some(source), None) => {
                    params.source = source;
                    Ok(())
                },
                (None, Some(source_relation)) => {
                    params.source = CugraphBfsSource::Relation(source_relation);
                    Ok(())
                },
                (None, None) => Err(plan_error(
                    "BFS source_vertex_or_vertices may be NULL only when source_vertices_table and source_vertex_col are provided",
                )),
            }
        },
        _ if algorithm.requires_source_vertex() => {
            if source.is_none() {
                return Err(plan_error(format!("{} requires source_vertex", algorithm.function_name())));
            }
            if source_relation.is_some() {
                return Err(plan_error(format!(
                    "{} does not accept source_vertices_table/source_vertex_col options",
                    algorithm.function_name()
                )));
            }
            Ok(())
        },
        _ => {
            if source_relation.is_some() {
                return Err(plan_error(format!(
                    "{} does not accept source_vertices_table/source_vertex_col options",
                    algorithm.function_name()
                )));
            }
            Ok(())
        },
    }
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

fn literal_source_arg(args: &[Expr], index: usize, algorithm: CugraphAlgorithm) -> Result<Option<CugraphBfsSource>> {
    match args.get(index) {
        Some(Expr::Literal(ScalarValue::Null, _)) if algorithm == CugraphAlgorithm::Bfs => Ok(None),
        Some(Expr::Literal(value, _)) => match value {
            ScalarValue::List(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::LargeList(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::List(_) | ScalarValue::LargeList(_) => {
                Err(plan_error("source_vertex_or_vertices list literal is supported only by cugraph_bfs"))
            },
            _ => literal_source_vertex_value(value, algorithm).map(|value| value.map(CugraphBfsSource::Scalar)),
        },
        Some(Expr::ScalarFunction(function))
            if algorithm == CugraphAlgorithm::Bfs && function.name().eq_ignore_ascii_case("make_array") =>
        {
            Ok(Some(CugraphBfsSource::LiteralList(source_make_array_values(&function.args, algorithm)?)))
        },
        Some(_) if algorithm == CugraphAlgorithm::Bfs => {
            Err(plan_error("source_vertex_or_vertices must be a literal or list literal"))
        },
        Some(_) => Err(plan_error("source_vertex must be a literal")),
        None if algorithm == CugraphAlgorithm::Bfs => Err(plan_error("missing source_vertex_or_vertices argument")),
        None => Err(plan_error("missing source_vertex argument")),
    }
}

fn source_make_array_values(args: &[Expr], algorithm: CugraphAlgorithm) -> Result<Vec<CugraphBfsSourceVertex>> {
    if args.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        let Expr::Literal(value, _) = arg else {
            return Err(plan_error("source_vertex_or_vertices list literal elements must be literals"));
        };
        let Some(value) = literal_source_vertex_value(value, algorithm)? else {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        };
        values.push(value);
    }
    validate_source_list_values(values)
}

fn source_list_values(array: &dyn Array) -> Result<Vec<CugraphBfsSourceVertex>> {
    if array.len() != 1 {
        return Err(plan_error("source_vertex_or_vertices list literal must be a single list value"));
    }
    if array.is_null(0) {
        return Err(plan_error("source_vertex_or_vertices list literal must not be null"));
    }
    let values = match array.data_type() {
        DataType::List(_) => array.as_list::<i32>().value(0),
        DataType::LargeList(_) => array.as_list::<i64>().value(0),
        other => {
            return Err(plan_error(format!(
                "source_vertex_or_vertices list literal has unsupported array type {other:?}"
            )));
        },
    };
    if values.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let mut parsed = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        }
        let value = ScalarValue::try_from_array(values.as_ref(), row).map_err(|error| {
            plan_error(format!("failed to read source_vertex_or_vertices list literal element {row}: {error}"))
        })?;
        let Some(value) = literal_source_vertex_value(&value, CugraphAlgorithm::Bfs)? else {
            return Err(plan_error("source_vertex_or_vertices list literal must not contain null elements"));
        };
        parsed.push(value);
    }
    validate_source_list_values(parsed)
}

fn validate_source_list_values(values: Vec<CugraphBfsSourceVertex>) -> Result<Vec<CugraphBfsSourceVertex>> {
    if values.is_empty() {
        return Err(plan_error("source_vertex_or_vertices list literal must not be empty"));
    }
    let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
    if values.iter().any(|value| value.is_utf8() != first_is_utf8) {
        return Err(plan_error("source_vertex_or_vertices list literal must not mix integer and string elements"));
    }
    Ok(values)
}

fn literal_source_vertex_value(
    value: &ScalarValue,
    algorithm: CugraphAlgorithm,
) -> Result<Option<CugraphBfsSourceVertex>> {
    match value {
        ScalarValue::Null => Ok(None),
        ScalarValue::Int8(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int16(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int32(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::Int64(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(*value))),
        ScalarValue::UInt8(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt16(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt32(Some(value)) => Ok(Some(CugraphBfsSourceVertex::Int64(i64::from(*value)))),
        ScalarValue::UInt64(Some(value)) => i64::try_from(*value)
            .map(|value| Some(CugraphBfsSourceVertex::Int64(value)))
            .map_err(|_| plan_error("source_vertex exceeds i64::MAX")),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value))
            if matches!(algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp) =>
        {
            Ok(Some(CugraphBfsSourceVertex::Utf8(value.clone())))
        },
        ScalarValue::Utf8(Some(_)) | ScalarValue::LargeUtf8(Some(_)) | ScalarValue::Utf8View(Some(_)) => {
            Err(plan_error("source_vertex must be an integer literal for this cuGraph algorithm"))
        },
        ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Utf8View(None) => Ok(None),
        _ => Err(plan_error("source_vertex must be a non-null integer literal")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn str_arg(value: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(value.to_string())), None)
    }

    #[test]
    fn parses_betweenness_explicit_seed_options() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[0,2,4],"normalized":false,"include_endpoints":true,"exact_vertex_threshold":7}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::BetweennessCentrality(params) = spec.params else {
            panic!("expected betweenness params");
        };
        assert_eq!(
            params.seeds.expect("seeds").vertices,
            vec![CugraphVertexSelector::Int64(0), CugraphVertexSelector::Int64(2), CugraphVertexSelector::Int64(4)]
        );
        assert!(!params.normalized);
        assert!(params.include_endpoints);
        assert_eq!(params.exact_vertex_threshold, 7);
    }

    #[test]
    fn parses_betweenness_k_option() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EdgeBetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"k":10,"normalized":false}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::EdgeBetweennessCentrality(params) = spec.params else {
            panic!("expected edge betweenness params");
        };
        assert_eq!(params.k, Some(10));
        assert!(params.seeds.is_none());
        assert!(!params.normalized);
    }

    #[test]
    fn rejects_betweenness_seeds_with_k() {
        let error = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[0],"k":1}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("betweenness seeds and k are mutually exclusive"));
    }

    #[test]
    fn parses_betweenness_null_seeds_as_exact_all_vertices() {
        let spec = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::BetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":null,"normalized":true}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap();

        let CugraphAlgorithmParams::BetweennessCentrality(params) = spec.params else {
            panic!("expected betweenness params");
        };
        assert!(params.seeds.is_none());
        assert!(params.normalized);
    }

    #[test]
    fn rejects_empty_betweenness_seed_list() {
        let error = parse_cugraph_algorithm_spec(
            CugraphAlgorithm::EdgeBetweennessCentrality,
            &[
                str_arg("edges"),
                str_arg("src"),
                str_arg("dst"),
                Expr::Literal(ScalarValue::Utf8(None), None),
                str_arg(r#"{"seeds":[]}"#),
            ],
            &CugraphSqlConfig::default(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("betweenness seeds must not be empty"));
    }
}
