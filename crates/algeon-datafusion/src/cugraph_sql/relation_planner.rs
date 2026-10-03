use std::sync::Arc;

use arrow::{
    array::{Array, AsArray},
    datatypes::DataType,
};
use datafusion::{
    common::{DFSchema, Result as DataFusionResult, ScalarValue},
    logical_expr::{
        Expr, Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::{Query as SqlQuery, TableFactor},
};
use serde_json::{Map, Number, Value};

use super::{
    CugraphAlgorithm, CugraphBfsSource, CugraphBfsSourceVertex, CugraphLogicalNode, CugraphRelationRole,
    CugraphSqlConfig,
    definitions::{CugraphArgumentKind, arguments_for_algorithm},
    plan,
};
use crate::{
    error::Result,
    gpu_functions::{NamedArgumentDefinition, NamedArgumentKind, bind_named_arguments},
};

#[derive(Debug, Clone)]
pub struct CugraphRelationPlanner {
    config: CugraphSqlConfig,
}

impl CugraphRelationPlanner {
    pub fn new(config: CugraphSqlConfig) -> Self {
        Self { config }
    }
}

pub fn cugraph_relation_planner(config: CugraphSqlConfig) -> Arc<dyn RelationPlanner> {
    Arc::new(CugraphRelationPlanner::new(config))
}

impl RelationPlanner for CugraphRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<RelationPlanning> {
        let Some(algorithm) = recognized_algorithm(&relation, context) else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };
        let TableFactor::Table {
            alias,
            args: Some(function_args),
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            sample,
            index_hints,
            ..
        } = relation
        else {
            unreachable!("recognized cuGraph relation is a table function")
        };
        if !with_hints.is_empty()
            || version.is_some()
            || with_ordinality
            || !partitions.is_empty()
            || json_path.is_some()
            || sample.is_some()
            || !index_hints.is_empty()
        {
            return Err(plan("cuGraph relation functions accept named arguments and a result alias").into());
        }

        let definitions = argument_definitions(algorithm);
        let mut arguments = bind_named_arguments(algorithm.function_name(), function_args, &definitions, |ident| {
            context.normalize_ident(ident)
        })?;
        let mut roles = vec![CugraphRelationRole::Edges];
        let edges =
            arguments.take_relation("edges").expect("required cuGraph edges argument was checked by the named binder");
        let mut inputs = vec![plan_relation_query(edges, context)?];
        for definition in arguments_for_algorithm(algorithm) {
            let CugraphArgumentKind::Relation { .. } = definition.kind else {
                continue;
            };
            if definition.name == "edges" {
                continue;
            }
            let role = relation_role(definition.name)?;
            if let Some(query) = arguments.take_relation(definition.name) {
                roles.push(role);
                inputs.push(plan_relation_query(query, context)?);
            }
        }

        let mut source = None;
        if matches!(algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp)
            && let Some(expression) = arguments.take_value("source_vertex")
        {
            let expression = fold_literal(context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?)?;
            source = parse_named_source(expression, algorithm)?;
            if source.is_none() {
                return Err(plan(format!(
                    "{} source_vertex must be a non-null scalar vertex",
                    algorithm.function_name()
                ))
                .into());
            }
            if algorithm == CugraphAlgorithm::Bfs && !matches!(source, Some(CugraphBfsSource::Scalar(_))) {
                return Err(plan("cugraph_bfs source_vertex must be a non-null scalar vertex").into());
            }
        }
        if algorithm == CugraphAlgorithm::Bfs
            && let Some(expression) = arguments.take_value("source_vertices")
        {
            if source.is_some() {
                return Err(
                    plan("cugraph_bfs source_vertex, source_vertices, and sources are mutually exclusive").into()
                );
            }
            let expression = fold_literal(context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?)?;
            let parsed = parse_named_source(expression, algorithm)?;
            if !matches!(parsed, Some(CugraphBfsSource::LiteralList(_))) {
                return Err(plan("cugraph_bfs source_vertices must be a non-empty typed array").into());
            }
            source = parsed;
        }
        if algorithm == CugraphAlgorithm::Bfs {
            let relation_source = roles.contains(&CugraphRelationRole::Sources);
            if usize::from(source.is_some()) + usize::from(relation_source) != 1 {
                return Err(
                    plan("cugraph_bfs requires exactly one of source_vertex, source_vertices, or sources").into()
                );
            }
        }

        let mut values = Map::new();
        for definition in definitions.iter().filter(|definition| definition.kind == NamedArgumentKind::Value) {
            if matches!(definition.name, "source_vertex" | "source_vertices") {
                continue;
            }
            if let Some(expression) = arguments.take_value(definition.name) {
                let expression = context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?;
                values.insert(definition.name.to_string(), literal_json(expression, definition.name)?);
            }
        }
        debug_assert!(arguments.is_empty());
        let node = CugraphLogicalNode::try_new(algorithm, inputs, roles, values, source, &self.config)?;
        let plan = LogicalPlan::Extension(Extension { node: Arc::new(node) });
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, alias))))
    }
}

fn recognized_algorithm(relation: &TableFactor, context: &dyn RelationPlannerContext) -> Option<CugraphAlgorithm> {
    let TableFactor::Table { name, args: Some(_), .. } = relation else {
        return None;
    };
    if name.0.len() != 1 {
        return None;
    }
    let name = context.normalize_ident(name.0.first()?.as_ident()?.clone());
    CugraphAlgorithm::all().find(|algorithm| algorithm.function_name() == name)
}

fn plan_relation_query(
    query: Box<SqlQuery>,
    context: &mut dyn RelationPlannerContext,
) -> DataFusionResult<LogicalPlan> {
    context.plan(TableFactor::Derived { lateral: false, subquery: query, alias: None, sample: None })
}

pub(super) fn relation_role(name: &str) -> Result<CugraphRelationRole> {
    let role = match name {
        "personalization" => CugraphRelationRole::Personalization,
        "sources" => CugraphRelationRole::Sources,
        "include_vertices" => CugraphRelationRole::IncludeVertices,
        "exclude_vertices" => CugraphRelationRole::ExcludeVertices,
        "target_vertices" => CugraphRelationRole::TargetVertices,
        "include_edge_ids" => CugraphRelationRole::IncludeEdgeIds,
        "include_edges" => CugraphRelationRole::IncludeEdges,
        "vertex_attributes" => CugraphRelationRole::VertexAttributes,
        "initial_positions" => CugraphRelationRole::InitialPositions,
        "vertex_pairs" => CugraphRelationRole::VertexPairs,
        _ => return Err(plan(format!("unknown cuGraph relation role '{name}'"))),
    };
    Ok(role)
}

fn argument_definitions(algorithm: CugraphAlgorithm) -> Vec<NamedArgumentDefinition> {
    arguments_for_algorithm(algorithm)
        .iter()
        .map(|definition| NamedArgumentDefinition {
            name: definition.name,
            kind: match definition.kind {
                CugraphArgumentKind::Relation { .. } => NamedArgumentKind::Relation,
                CugraphArgumentKind::Value(_) => NamedArgumentKind::Value,
            },
            required: definition.required,
        })
        .collect()
}

/// Reduces `CAST(<literal> AS type)`, negated numeric literals, and
/// `make_array(...)` of those to plain literals so typed literal arguments bind
/// like untyped ones.
fn fold_literal(expression: Expr) -> DataFusionResult<Expr> {
    Ok(match expression {
        Expr::Cast(cast) => match fold_literal(*cast.expr)? {
            Expr::Literal(value, _) => Expr::Literal(value.cast_to(cast.field.data_type())?, None),
            _ => return Err(plan("cuGraph literal cast must contain a literal").into()),
        },
        Expr::Negative(inner) => match fold_literal(*inner)? {
            Expr::Literal(value, _) => Expr::Literal(value.arithmetic_negate()?, None),
            _ => return Err(plan("cuGraph negated literal must contain a numeric literal").into()),
        },
        Expr::ScalarFunction(mut function) if function.name().eq_ignore_ascii_case("make_array") => {
            function.args = function.args.into_iter().map(fold_literal).collect::<DataFusionResult<Vec<_>>>()?;
            Expr::ScalarFunction(function)
        },
        other => other,
    })
}

fn literal_json(expression: Expr, name: &str) -> DataFusionResult<Value> {
    match expression {
        Expr::Literal(value, _) => scalar_json(value, name),
        Expr::Negative(inner) => match literal_json(*inner, name)? {
            Value::Number(value) => value
                .as_f64()
                .and_then(|value| Number::from_f64(-value))
                .map(Value::Number)
                .ok_or_else(|| plan(format!("cuGraph argument '{name}' is not a finite numeric literal")).into()),
            _ => Err(plan(format!("cuGraph argument '{name}' must be a numeric literal")).into()),
        },
        Expr::Cast(cast) => {
            let Expr::Literal(value, _) = *cast.expr else {
                return Err(plan(format!("cuGraph argument '{name}' cast must contain a literal")).into());
            };
            scalar_json(value.cast_to(cast.field.data_type())?, name)
        },
        Expr::ScalarFunction(function) if function.name().eq_ignore_ascii_case("make_array") => function
            .args
            .into_iter()
            .map(|argument| literal_json(argument, name))
            .collect::<DataFusionResult<Vec<_>>>()
            .map(Value::Array),
        _ => Err(plan(format!("cuGraph argument '{name}' must be a scalar or typed-array literal")).into()),
    }
}

fn scalar_json(value: ScalarValue, name: &str) -> DataFusionResult<Value> {
    let value = match value {
        ScalarValue::Null
        | ScalarValue::Boolean(None)
        | ScalarValue::Int8(None)
        | ScalarValue::Int16(None)
        | ScalarValue::Int32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::UInt8(None)
        | ScalarValue::UInt16(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Float32(None)
        | ScalarValue::Float64(None)
        | ScalarValue::Utf8(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::Utf8View(None) => Value::Null,
        ScalarValue::Boolean(Some(value)) => Value::Bool(value),
        ScalarValue::Int8(Some(value)) => Value::Number(value.into()),
        ScalarValue::Int16(Some(value)) => Value::Number(value.into()),
        ScalarValue::Int32(Some(value)) => Value::Number(value.into()),
        ScalarValue::Int64(Some(value)) => Value::Number(value.into()),
        ScalarValue::UInt8(Some(value)) => Value::Number(value.into()),
        ScalarValue::UInt16(Some(value)) => Value::Number(value.into()),
        ScalarValue::UInt32(Some(value)) => Value::Number(value.into()),
        ScalarValue::UInt64(Some(value)) => Value::Number(value.into()),
        ScalarValue::Float32(Some(value)) => Number::from_f64(f64::from(value))
            .map(Value::Number)
            .ok_or_else(|| plan(format!("cuGraph argument '{name}' must be finite")))?,
        ScalarValue::Float64(Some(value)) => Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| plan(format!("cuGraph argument '{name}' must be finite")))?,
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Value::String(value)
        },
        ScalarValue::List(values) => {
            if values.len() != 1 || values.is_null(0) {
                return Err(plan(format!("cuGraph argument '{name}' must contain one non-null array literal")).into());
            }
            let array = values.value(0);
            let mut result = Vec::with_capacity(array.len());
            for index in 0..array.len() {
                result.push(scalar_json(ScalarValue::try_from_array(array.as_ref(), index)?, name)?);
            }
            Value::Array(result)
        },
        ScalarValue::LargeList(values) => {
            if values.len() != 1 || values.is_null(0) {
                return Err(plan(format!("cuGraph argument '{name}' must contain one non-null array literal")).into());
            }
            let array = values.value(0);
            let mut result = Vec::with_capacity(array.len());
            for index in 0..array.len() {
                result.push(scalar_json(ScalarValue::try_from_array(array.as_ref(), index)?, name)?);
            }
            Value::Array(result)
        },
        _ => return Err(plan(format!("cuGraph argument '{name}' has an unsupported literal type")).into()),
    };
    Ok(value)
}

fn parse_named_source(expression: Expr, algorithm: CugraphAlgorithm) -> Result<Option<CugraphBfsSource>> {
    match expression {
        Expr::Literal(ScalarValue::Null, _) => Ok(None),
        Expr::Literal(value, _) => match value {
            ScalarValue::List(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::LargeList(values) if algorithm == CugraphAlgorithm::Bfs => {
                Ok(Some(CugraphBfsSource::LiteralList(source_list_values(values.as_ref())?)))
            },
            ScalarValue::List(_) | ScalarValue::LargeList(_) => {
                Err(plan("source_vertices is supported only by cugraph_bfs"))
            },
            _ => literal_source_vertex_value(&value, algorithm).map(|value| value.map(CugraphBfsSource::Scalar)),
        },
        Expr::ScalarFunction(function)
            if algorithm == CugraphAlgorithm::Bfs && function.name().eq_ignore_ascii_case("make_array") =>
        {
            Ok(Some(CugraphBfsSource::LiteralList(source_make_array_values(&function.args, algorithm)?)))
        },
        _ if algorithm == CugraphAlgorithm::Bfs => Err(plan("source_vertex or source_vertices must be a literal")),
        _ => Err(plan("source_vertex must be a literal")),
    }
}

fn source_make_array_values(args: &[Expr], algorithm: CugraphAlgorithm) -> Result<Vec<CugraphBfsSourceVertex>> {
    if args.is_empty() {
        return Err(plan("source_vertices must not be empty"));
    }
    let mut values = Vec::with_capacity(args.len());
    for arg in args {
        let Expr::Literal(value, _) = arg else {
            return Err(plan("source_vertices elements must be literals"));
        };
        let Some(value) = literal_source_vertex_value(value, algorithm)? else {
            return Err(plan("source_vertices must not contain null elements"));
        };
        values.push(value);
    }
    validate_source_list_values(values)
}

fn source_list_values(array: &dyn Array) -> Result<Vec<CugraphBfsSourceVertex>> {
    if array.len() != 1 || array.is_null(0) {
        return Err(plan("source_vertices must be one non-null array literal"));
    }
    let values = match array.data_type() {
        DataType::List(_) => array.as_list::<i32>().value(0),
        DataType::LargeList(_) => array.as_list::<i64>().value(0),
        other => return Err(plan(format!("source_vertices has unsupported array type {other:?}"))),
    };
    if values.is_empty() {
        return Err(plan("source_vertices must not be empty"));
    }
    let mut parsed = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            return Err(plan("source_vertices must not contain null elements"));
        }
        let value = ScalarValue::try_from_array(values.as_ref(), row)
            .map_err(|error| plan(format!("failed to read source_vertices element {row}: {error}")))?;
        let Some(value) = literal_source_vertex_value(&value, CugraphAlgorithm::Bfs)? else {
            return Err(plan("source_vertices must not contain null elements"));
        };
        parsed.push(value);
    }
    validate_source_list_values(parsed)
}

fn validate_source_list_values(values: Vec<CugraphBfsSourceVertex>) -> Result<Vec<CugraphBfsSourceVertex>> {
    if values.is_empty() {
        return Err(plan("source_vertices must not be empty"));
    }
    let first_is_utf8 = values.first().is_some_and(CugraphBfsSourceVertex::is_utf8);
    if values.iter().any(|value| value.is_utf8() != first_is_utf8) {
        return Err(plan("source_vertices must not mix integer and string elements"));
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
            .map_err(|_| plan("source_vertex exceeds i64::MAX")),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value))
            if matches!(algorithm, CugraphAlgorithm::Bfs | CugraphAlgorithm::Sssp) =>
        {
            Ok(Some(CugraphBfsSourceVertex::Utf8(value.clone())))
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
        _ => Err(plan("source_vertex must be a non-null integer or string literal")),
    }
}
