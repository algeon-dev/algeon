//! Aggregate node validation: output naming, AVG and statistical output
//! contracts, reserved state alias collisions, and which aggregates read only
//! the validity of their argument.
//!
//! The backend proves it can bind the spec and reports the output names its
//! binding produces (`BackendCapabilities::aggregate_output_names`); the
//! declared output schema must list those names in that order, so the proof
//! and the schema check read one result. Two state contracts reserve internal
//! aliases a visible output may not reuse: AVG expands into sum and count
//! state columns (`AvgStateAliases`), and a global SUM, MIN or MAX partial
//! carries a valid-count column (`global_valid_count_state_alias`) that the
//! final merge sums to learn whether any non-null input existed.

use arrow::datatypes::SchemaRef as ArrowSchemaRef;

use super::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateSpec, ArrowDataType, AvgStateAliases, Expr, LiteralValue,
    QueryError, QueryErrorCode, Result, aggregate_spec_has_avg, aggregate_spec_has_distinct,
    aggregate_spec_has_floating_statistical_aggregate, avg_division_type, lower_column_expr, lower_named_column_expr,
};
use crate::{
    backend::{AggregateBinding, AnalysisContext},
    expr::alias_wrapped_column,
};

const GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX: &str = "__algeon_native_global_valid_count_";

/// Output name the statistical aggregate path gives a group expression: the
/// alias, else the column's display name, else the rendered expression.
pub(crate) fn host_group_expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias { alias, .. } => alias.clone(),
        Expr::Column(column) => column.display_name().to_string(),
        _ => expr.to_string(),
    }
}

/// Whether a `COUNT` argument is the planner's count-star representation: any
/// non-null literal. `LiteralValue::Null` is excluded because `COUNT` of a null
/// literal is zero.
pub fn global_count_star_expr(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal(
            LiteralValue::Boolean(_)
                | LiteralValue::Int8(_)
                | LiteralValue::Int16(_)
                | LiteralValue::Int32(_)
                | LiteralValue::Int64(_)
                | LiteralValue::UInt8(_)
                | LiteralValue::UInt16(_)
                | LiteralValue::UInt32(_)
                | LiteralValue::UInt64(_)
                | LiteralValue::Float32(_)
                | LiteralValue::Float64(_)
                | LiteralValue::Utf8(_)
                | LiteralValue::LargeUtf8(_)
                | LiteralValue::Decimal128 { .. }
                | LiteralValue::Date32(_)
                | LiteralValue::Timestamp { .. }
                | LiteralValue::Duration { .. }
                | LiteralValue::IntervalMonthDayNano { .. },
        )
    )
}

/// The parts of an aggregate that decide whether native execution reads its
/// argument values or only the argument's top-level validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateArgumentUse {
    /// Aggregate function applied to the argument.
    pub function: AggregateFunction,
    /// The aggregate uses SQL `DISTINCT` semantics.
    pub distinct: bool,
    /// The aggregate carries a SQL `FILTER (WHERE ...)` clause.
    pub filtered: bool,
    /// The aggregate has group keys.
    pub grouped: bool,
    /// The argument is a column, possibly aliased, not a computed expression.
    pub column_argument: bool,
}

impl AggregateArgumentUse {
    /// Whether the aggregate reads only the top-level validity of its
    /// argument: a global `COUNT(column)` without `DISTINCT` or `FILTER`.
    /// Such a count never dispatches a kernel over the values, so a carry
    /// type admitted by [`is_validity_count_input_type`](super::is_validity_count_input_type)
    /// is a valid argument.
    pub fn reads_only_top_level_validity(self) -> bool {
        self.function == AggregateFunction::Count
            && !self.distinct
            && !self.filtered
            && !self.grouped
            && self.column_argument
    }
}

pub(crate) fn aggregate_argument_use(spec: &AggregateSpec, expr: &AggregateExpr) -> AggregateArgumentUse {
    AggregateArgumentUse {
        function: expr.function,
        distinct: expr.distinct,
        filtered: expr.filter.is_some(),
        grouped: !spec.group_exprs.is_empty(),
        column_argument: alias_wrapped_column(&expr.expr).is_some(),
    }
}

pub(crate) fn validate_avg_state_alias_collisions(spec: &AggregateSpec, operation: &'static str) -> Result<()> {
    if !aggregate_spec_has_avg(spec) {
        return Ok(());
    }
    let reserved_aliases = spec
        .aggr_exprs
        .iter()
        .enumerate()
        .filter(|(_, expr)| expr.function == AggregateFunction::Avg)
        .flat_map(|(index, _)| AvgStateAliases::for_aggregate_index(index).output_names())
        .collect::<Vec<_>>();
    validate_reserved_aggregate_aliases(spec, operation, &reserved_aliases, "AVG")
}

pub fn validate_avg_output_fields(spec: &AggregateSpec, output_schema: &ArrowSchemaRef) -> Result<()> {
    for expr in spec.aggr_exprs.iter().filter(|expr| expr.function == AggregateFunction::Avg) {
        let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.aggregate.validate_avg",
                "avg aggregate output field is missing from output schema",
            )
            .with_fact("alias", expr.alias.as_str())
        })?;
        if avg_division_type(field.data_type()).is_none() {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                "query_engine.aggregate.validate_avg",
                "avg aggregate output schema must be Float32, Float64, or Decimal128",
            )
            .with_fact("alias", expr.alias.as_str())
            .with_fact("data_type", format!("{:?}", field.data_type())));
        }
    }
    Ok(())
}

pub fn global_valid_count_state_alias(alias: &str) -> String {
    format!("{GLOBAL_VALID_COUNT_STATE_ALIAS_PREFIX}{alias}")
}

pub(crate) fn validate_global_valid_count_state_alias_collisions(
    spec: &AggregateSpec,
    operation: &'static str,
) -> Result<()> {
    let reserved_aliases = spec
        .aggr_exprs
        .iter()
        .filter(|expr| {
            matches!(expr.function, AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max)
        })
        .map(|expr| global_valid_count_state_alias(expr.alias.as_str()))
        .collect::<Vec<_>>();
    validate_reserved_aggregate_aliases(spec, operation, &reserved_aliases, "global aggregate")
}

fn validate_reserved_aggregate_aliases(
    spec: &AggregateSpec,
    operation: &'static str,
    reserved_aliases: &[String],
    state: &'static str,
) -> Result<()> {
    if reserved_aliases.is_empty() {
        return Ok(());
    }
    let visible_aliases = spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, operation).map(|(_, alias)| alias))
        .chain(spec.aggr_exprs.iter().map(|expr| Ok(expr.alias.clone())))
        .collect::<Result<Vec<_>>>()?;
    for alias in visible_aliases {
        if reserved_aliases.iter().any(|reserved| reserved == &alias) {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                format!("visible aggregate output alias conflicts with reserved internal {state} state column"),
            )
            .with_fact("alias", alias)
            .with_fact("reserved_aliases", reserved_aliases.join(",")));
        }
    }
    Ok(())
}

/// Output and state contracts of one aggregate node. Global aggregates have
/// their own naming rule (no key columns) and valid-count state. A grouped spec
/// with a floating statistical aggregate runs through the statistical
/// aggregate execution contract, which names its outputs from the group
/// expressions and aliases itself; the libcudf binding proof does not apply to
/// it, and distinct arguments are outside it.
pub(super) fn validate_aggregate_capability(
    node: &AggregateNode,
    output_schema: &ArrowSchemaRef,
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        validate_global_valid_count_state_alias_collisions(
            &node.spec,
            "query_engine.global_aggregate.validate_valid_count_state_alias",
        )?;
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_floating_statistical_aggregate(&node.spec) {
        if aggregate_spec_has_distinct(&node.spec) {
            return Err(QueryError::new(
                QueryErrorCode::UnsupportedOperator,
                "query_engine.aggregate.validate_statistical_exact",
                "DataFusion-exact statistical aggregate path does not support distinct aggregate expressions",
            ));
        }
        let expected_output_names = node
            .spec
            .group_exprs
            .iter()
            .map(host_group_expr_output_name)
            .chain(node.spec.aggr_exprs.iter().map(|expr| expr.alias.clone()))
            .collect::<Vec<_>>();
        let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
        if expected_output_names != schema_output_names {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.aggregate.validate_statistical_exact",
                "DataFusion-exact statistical aggregate output names must match output schema fields",
            )
            .with_fact("aggregate_output_names", expected_output_names.join(","))
            .with_fact("schema_output_names", schema_output_names.join(",")));
        }
        return Ok(());
    }
    let validation_spec = aggregate_validation_spec(&node.spec);
    let output_names = ctx.backend().aggregate_output_names(&validation_spec, AggregateBinding::SingleStage)?;
    validate_aggregate_output_names(
        &output_names,
        output_schema,
        "query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    Ok(())
}

/// The spec given to the backend binding proof. That proof lowers every
/// argument as a column, while a filtered `COUNT(*)` carries a literal
/// argument the filtered-aggregate executor handles itself. The literal is
/// swapped for a synthetic column name so the proof still covers the rest of
/// the spec; the name never reaches execution.
fn aggregate_validation_spec(spec: &AggregateSpec) -> AggregateSpec {
    let mut validation = spec.clone();
    for (index, expr) in validation.aggr_exprs.iter_mut().enumerate() {
        if expr.filter.is_some()
            && expr.function == AggregateFunction::Count
            && !expr.distinct
            && global_count_star_expr(&expr.expr)
        {
            expr.expr = Expr::column(format!("__algeon_native_filtered_count_star_validation_{index}"));
        }
    }
    validation
}

pub fn validate_global_aggregate_capability(spec: &AggregateSpec, output_schema: &ArrowSchemaRef) -> Result<()> {
    if aggregate_spec_has_avg(spec) {
        validate_avg_state_alias_collisions(spec, "query_engine.global_aggregate.validate_avg_state_alias")?;
    }
    let expected_output_names = spec.aggr_exprs.iter().map(|expr| expr.alias.clone()).collect::<Vec<_>>();
    let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if expected_output_names != schema_output_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.global_aggregate.validate",
            "global aggregate output names must match output schema fields",
        )
        .with_fact("aggregate_output_names", expected_output_names.join(","))
        .with_fact("schema_output_names", schema_output_names.join(",")));
    }
    for expr in &spec.aggr_exprs {
        if expr.function == AggregateFunction::Avg {
            validate_avg_output_fields(spec, output_schema)?;
        }
        if expr.function.is_floating_statistical() {
            if expr.distinct {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedOperator,
                    "query_engine.global_aggregate.validate_statistical_exact",
                    "DataFusion-exact statistical aggregate path does not support distinct aggregate expressions",
                )
                .with_fact("alias", expr.alias.as_str()));
            }
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.global_aggregate.validate_statistical_exact",
                    "floating statistical aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            if !matches!(field.data_type(), ArrowDataType::Float64) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.global_aggregate.validate_statistical_exact",
                    "floating statistical aggregate output must be Float64",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("function", expr.function.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type())));
            }
        }
        validate_global_aggregate_input_expr(expr)?;
    }
    Ok(())
}

/// Global aggregates lower every argument as a column except the count-star
/// literal, which the executor counts without reading a column.
fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if !expr.distinct && global_count_star_expr(&expr.expr) => Ok(()),
        _ => lower_column_expr(&expr.expr, "query_engine.global_aggregate.validate_expr").map(|_| ()),
    }
}

pub fn validate_aggregate_output_names(
    output_names: &[String],
    output_schema: &ArrowSchemaRef,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if output_names != expected_output_names.as_slice() {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("aggregate_output_names", output_names.join(","))
            .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    Ok(())
}
