use std::sync::Arc;

use datafusion::{
    common::{DFSchema, DataFusionError, Result, ScalarValue},
    logical_expr::{
        Expr, Extension, LogicalPlan,
        planner::{PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning},
    },
    sql::sqlparser::ast::{Expr as SqlExpr, TableFactor},
};
use serde_json::{Map, Number, Value};

use super::{CuvsLogicalNode, CuvsOperation, CuvsOptionKind, error::invalid_call, options::options_from_named_values};
use crate::gpu_functions::{NamedArgumentDefinition, NamedArgumentKind, bind_named_arguments};

#[derive(Debug, Default)]
pub struct CuvsRelationPlanner;

pub fn cuvs_relation_planner() -> Arc<dyn RelationPlanner> {
    Arc::new(CuvsRelationPlanner)
}

impl RelationPlanner for CuvsRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let Some(operation) = recognized_operation(&relation, context) else {
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
            unreachable!("recognized cuVS relation is a table function")
        };
        if !with_hints.is_empty()
            || version.is_some()
            || with_ordinality
            || !partitions.is_empty()
            || json_path.is_some()
            || sample.is_some()
            || !index_hints.is_empty()
        {
            return Err(invalid_call(
                operation,
                "unsupported_table_function_modifier",
                "cuVS relation functions accept named arguments and a result alias",
            ));
        }

        let definitions = argument_definitions(operation);
        let mut arguments =
            bind_named_arguments(operation.descriptor().sql_function_name, function_args, &definitions, |ident| {
                context.normalize_ident(ident)
            })?;
        let mut inputs = Vec::with_capacity(operation.descriptor().relation_roles.len());
        for role in operation.descriptor().relation_roles {
            let query = arguments
                .take_relation(role.as_str())
                .expect("required cuVS relation argument was checked by the named binder");
            inputs.push(context.plan(TableFactor::Derived {
                lateral: false,
                subquery: query,
                alias: None,
                sample: None,
            })?);
        }

        let mut values = Map::new();
        for option in operation.descriptor().options {
            if matches!(option.kind, CuvsOptionKind::RelationBinding) {
                continue;
            }
            if let Some(expression) = arguments.take_value(option.name) {
                values.insert(option.name.to_string(), value_argument(operation, option.name, expression, context)?);
            }
        }
        debug_assert!(arguments.is_empty());
        let options_json = options_from_named_values(operation, &inputs, values)?;
        let node = CuvsLogicalNode::try_new(operation, inputs, &options_json)?;
        let plan = LogicalPlan::Extension(Extension { node: Arc::new(node) });
        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(plan, alias))))
    }
}

fn recognized_operation(relation: &TableFactor, context: &dyn RelationPlannerContext) -> Option<CuvsOperation> {
    let TableFactor::Table { name, args: Some(_), .. } = relation else {
        return None;
    };
    if name.0.len() != 1 {
        return None;
    }
    let ident = name.0.first()?.as_ident()?.clone();
    CuvsOperation::from_sql_function_name(&context.normalize_ident(ident))
}

fn argument_definitions(operation: CuvsOperation) -> Vec<NamedArgumentDefinition> {
    let descriptor = operation.descriptor();
    descriptor
        .relation_roles
        .iter()
        .map(|role| NamedArgumentDefinition { name: role.as_str(), kind: NamedArgumentKind::Relation, required: true })
        .chain(descriptor.options.iter().filter_map(|option| {
            (!matches!(option.kind, CuvsOptionKind::RelationBinding)).then_some(NamedArgumentDefinition {
                name: option.name,
                kind: NamedArgumentKind::Value,
                required: option.required,
            })
        }))
        .collect()
}

fn value_argument(
    operation: CuvsOperation,
    name: &'static str,
    expression: SqlExpr,
    context: &mut dyn RelationPlannerContext,
) -> Result<Value> {
    let expression = context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?;
    scalar_literal(operation, name, expression).and_then(|value| scalar_json(operation, name, value))
}

fn scalar_literal(operation: CuvsOperation, name: &'static str, expression: Expr) -> Result<ScalarValue> {
    match expression {
        Expr::Literal(value, _) => Ok(value),
        Expr::Negative(inner) => negate_scalar(operation, name, scalar_literal(operation, name, *inner)?),
        Expr::Cast(cast) => scalar_literal(operation, name, *cast.expr)?
            .cast_to(cast.field.data_type())
            .map_err(|_| invalid_value(operation, name)),
        _ => Err(invalid_value(operation, name)),
    }
}

fn negate_scalar(operation: CuvsOperation, name: &'static str, value: ScalarValue) -> Result<ScalarValue> {
    let invalid = || invalid_value(operation, name);
    match value {
        ScalarValue::Int8(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int8(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int16(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int16(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int32(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int32(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Int64(Some(value)) => {
            value.checked_neg().map(|value| ScalarValue::Int64(Some(value))).ok_or_else(invalid)
        },
        ScalarValue::Float32(Some(value)) => Ok(ScalarValue::Float32(Some(-value))),
        ScalarValue::Float64(Some(value)) => Ok(ScalarValue::Float64(Some(-value))),
        _ => Err(invalid()),
    }
}

fn scalar_json(operation: CuvsOperation, name: &'static str, value: ScalarValue) -> Result<Value> {
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
        ScalarValue::Int8(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int16(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int32(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Int64(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt8(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt16(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt32(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::UInt64(Some(value)) => Value::Number(Number::from(value)),
        ScalarValue::Float32(Some(value)) => {
            Number::from_f64(f64::from(value)).map(Value::Number).ok_or_else(|| invalid_value(operation, name))?
        },
        ScalarValue::Float64(Some(value)) => {
            Number::from_f64(value).map(Value::Number).ok_or_else(|| invalid_value(operation, name))?
        },
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Value::String(value)
        },
        _ => return Err(invalid_value(operation, name)),
    };
    Ok(value)
}

fn invalid_value(operation: CuvsOperation, name: &'static str) -> DataFusionError {
    invalid_call(operation, "value_argument_not_literal", format!("cuVS argument '{name}' must be a scalar literal"))
}
