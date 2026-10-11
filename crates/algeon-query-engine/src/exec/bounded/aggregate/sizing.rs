//! Upper bound on the device bytes of one partial aggregate's output state.
//!
//! `partial_aggregate_work_reservation_upper_bound_bytes` bounds the partial-state frame that an
//! aggregate over one input frame can produce, from column types and row counts alone, without
//! running the aggregate. It has two callers, `stream.rs` and `source_adjacent.rs`. Each compares
//! the bound with the attempt's device budget (the immutable cap, a batching input) and, when the
//! bound is larger and retry budget remains, splits the input frame in halves before the partial
//! runs. The bound is neither a reservation nor an admission check. The query allocator alone
//! enforces the grant, and a frame that is not split but still does not fit fails at its
//! allocation like any other. `None` means the byte model cannot bound the state (a median, or a
//! dtype the model does not cover), and the frame runs whole.
//!
//! The bound applies the per-column formula of `crate::materialization_sizing` (shared with join
//! output sizing) to the partial-state schema with the right row count:
//!
//! - A global aggregate has 1 output row.
//! - A grouped aggregate has at most `input_rows` rows, because every input row could be a unique
//!   group key. Grouping sets multiply that by the number of sets. No libcudf grouped-aggregate
//!   preflight gives a tighter row count; this is the only deterministic ceiling.
//!
//! The caller builds the per-column descriptors from the partial-state schema. Grouped partial
//! output starts with the group key columns (`lower_partial_aggregate_spec` places
//! `output_key_names` first), followed by the aggregate state columns. A string-typed column needs
//! `input_max_string_length_bytes` and the total string bytes of the input column, measured on the
//! device under a budget token (`string_byte_length_stats_for_partial_sizing`).

use std::collections::BTreeMap;

use arrow::datatypes::SchemaRef as ArrowSchemaRef;
use query_planning::prepared::{AVG_COUNT_STATE_TYPE, avg_sum_state_type};

use super::{
    AggregateFunction, AggregateNode, AggregateSpec, ArrowDataType, AvgStateAliases, BoundedExecutionAttempt, Expr,
    Field, GpuDataFrame, PartialAggregateShape, QueryError, QueryErrorCode, Result, cudf_query_error,
    execution_budget_token_for_boundary, global_valid_count_state_alias, lower_column_expr, lower_named_column_expr,
    partial_aggregate_upper_bound_bytes, partial_global_aggregate_output_schema,
};
use crate::materialization_sizing::JoinOutputColumnUpperBoundInput;

pub(super) fn partial_aggregate_work_reservation_upper_bound_bytes(
    execution: &mut BoundedExecutionAttempt<'_>,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: &GpuDataFrame,
) -> Result<Option<u64>> {
    let shape = if aggregate.spec.group_exprs.is_empty() {
        PartialAggregateShape::Global
    } else if let Some(grouping_sets) = &aggregate.spec.grouping_sets {
        PartialAggregateShape::Grouped {
            input_rows: (input.num_rows() as u64).saturating_mul(grouping_sets.groups.len() as u64),
        }
    } else {
        PartialAggregateShape::Grouped { input_rows: input.num_rows() as u64 }
    };
    let Some(columns) =
        partial_aggregate_upper_bound_columns(execution, aggregate, input_schema, aggregate_output_schema, input)?
    else {
        return Ok(None);
    };
    partial_aggregate_upper_bound_bytes(shape, &columns)
}

fn partial_aggregate_upper_bound_columns(
    execution: &mut BoundedExecutionAttempt<'_>,
    aggregate: &AggregateNode,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: &GpuDataFrame,
) -> Result<Option<Vec<JoinOutputColumnUpperBoundInput>>> {
    if aggregate.spec.group_exprs.is_empty() {
        return global_partial_aggregate_upper_bound_columns(
            execution,
            &aggregate.spec,
            input_schema,
            aggregate_output_schema,
            input,
        );
    }
    grouped_partial_aggregate_upper_bound_columns(
        execution,
        &aggregate.spec,
        input_schema,
        aggregate_output_schema,
        input,
    )
}

fn global_partial_aggregate_upper_bound_columns(
    execution: &mut BoundedExecutionAttempt<'_>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: &GpuDataFrame,
) -> Result<Option<Vec<JoinOutputColumnUpperBoundInput>>> {
    let partial_schema = partial_global_aggregate_output_schema(spec, input_schema, aggregate_output_schema)?;
    let mut fields_by_name =
        partial_schema.fields().iter().map(|field| (field.name().as_str(), field.as_ref())).collect::<BTreeMap<_, _>>();
    let mut columns = Vec::with_capacity(partial_schema.fields().len());

    for (index, expr) in spec.aggr_exprs.iter().enumerate() {
        match expr.function {
            AggregateFunction::Avg => {
                let state = AvgStateAliases::for_aggregate_index(index);
                columns.push(fixed_partial_column(take_partial_field(
                    &mut fields_by_name,
                    state.sum(),
                    "global AVG sum state",
                )?));
                columns.push(fixed_partial_column(take_partial_field(
                    &mut fields_by_name,
                    state.count(),
                    "global AVG count state",
                )?));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                columns.push(partial_column_for_field_and_optional_input(
                    execution,
                    take_partial_field(&mut fields_by_name, expr.alias.as_str(), "global aggregate value state")?,
                    Some(&expr.expr),
                    input_schema,
                    input,
                )?);
                columns.push(fixed_partial_column(take_partial_field(
                    &mut fields_by_name,
                    global_valid_count_state_alias(expr.alias.as_str()).as_str(),
                    "global aggregate valid-count state",
                )?));
            },
            // A median has no partial state, so no partial size bounds it.
            AggregateFunction::Median => return Ok(None),
            AggregateFunction::Count | AggregateFunction::NUnique => {
                columns.push(partial_column_for_field_and_optional_input(
                    execution,
                    take_partial_field(&mut fields_by_name, expr.alias.as_str(), "global aggregate state")?,
                    None,
                    input_schema,
                    input,
                )?);
            },
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop => {
                columns.push(partial_column_for_field_and_optional_input(
                    execution,
                    take_partial_field(&mut fields_by_name, expr.alias.as_str(), "global aggregate state")?,
                    None,
                    input_schema,
                    input,
                )?);
            },
        }
    }

    Ok(Some(columns))
}

fn grouped_partial_aggregate_upper_bound_columns(
    execution: &mut BoundedExecutionAttempt<'_>,
    spec: &AggregateSpec,
    input_schema: &ArrowSchemaRef,
    aggregate_output_schema: &ArrowSchemaRef,
    input: &GpuDataFrame,
) -> Result<Option<Vec<JoinOutputColumnUpperBoundInput>>> {
    // State sizing reads output types; input expressions are materialized by
    // the operator owner before kernel lowering (including COUNT literals).
    let mut columns = Vec::with_capacity(spec.group_exprs.len() + spec.aggr_exprs.len().saturating_mul(2));

    for expr in &spec.group_exprs {
        let (input_column, output_name) =
            lower_named_column_expr(expr, "query_engine.bounded.aggregate.partial_sizing_group_key")?;
        let field = input_schema.field_with_name(input_column.as_str()).map_err(|error| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.bounded.aggregate.partial_sizing",
                "partial aggregate group key is missing from its input schema",
            )
            .with_fact("column", input_column.as_str())
            .with_fact("output_name", output_name.as_str())
            .with_fact("arrow_error", error.to_string())
        })?;
        columns.push(partial_column_for_field_and_input_column(execution, field, input_column.as_str(), input)?);
    }

    if let Some(grouping_sets) = &spec.grouping_sets {
        let field =
            aggregate_output_schema.field_with_name(grouping_sets.grouping_id_alias.as_str()).map_err(|error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.bounded.aggregate.partial_sizing",
                    "partial aggregate grouping-id column is missing from its output schema",
                )
                .with_fact("grouping_id_alias", grouping_sets.grouping_id_alias.as_str())
                .with_fact("arrow_error", error.to_string())
            })?;
        columns.push(fixed_partial_column(field));
    }

    for expr in &spec.aggr_exprs {
        match expr.function {
            AggregateFunction::Avg => {
                let input_type = aggregate_input_data_type(&expr.expr, input_schema)?;
                let output_field = aggregate_output_schema.field_with_name(expr.alias.as_str()).map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.bounded.aggregate.partial_sizing",
                        "AVG output field is missing from aggregate output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("arrow_error", error.to_string())
                })?;
                columns.push(JoinOutputColumnUpperBoundInput {
                    data_type: avg_sum_state_type(&input_type, output_field.data_type())?,
                    input_max_string_length_bytes: None,
                    input_total_string_bytes: None,
                });
                columns.push(JoinOutputColumnUpperBoundInput {
                    data_type: AVG_COUNT_STATE_TYPE,
                    input_max_string_length_bytes: None,
                    input_total_string_bytes: None,
                });
            },
            AggregateFunction::Count | AggregateFunction::NUnique => {
                let field = aggregate_output_schema.field_with_name(expr.alias.as_str()).map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.bounded.aggregate.partial_sizing",
                        "COUNT output field is missing from aggregate output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("arrow_error", error.to_string())
                })?;
                columns.push(fixed_partial_column(field));
            },
            AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => {
                let field = aggregate_output_schema.field_with_name(expr.alias.as_str()).map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.bounded.aggregate.partial_sizing",
                        "aggregate output field is missing from aggregate output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("arrow_error", error.to_string())
                })?;
                columns.push(partial_column_for_field_and_optional_input(
                    execution,
                    field,
                    Some(&expr.expr),
                    input_schema,
                    input,
                )?);
            },
            AggregateFunction::Median => return Ok(None),
            AggregateFunction::StddevSamp
            | AggregateFunction::StddevPop
            | AggregateFunction::VarSamp
            | AggregateFunction::VarPop => {
                let field = aggregate_output_schema.field_with_name(expr.alias.as_str()).map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "query_engine.bounded.aggregate.partial_sizing",
                        "aggregate output field is missing from aggregate output schema",
                    )
                    .with_fact("alias", expr.alias.as_str())
                    .with_fact("arrow_error", error.to_string())
                })?;
                columns.push(partial_column_for_field_and_optional_input(
                    execution,
                    field,
                    Some(&expr.expr),
                    input_schema,
                    input,
                )?);
            },
        }
    }

    Ok(Some(columns))
}

fn take_partial_field<'a>(
    fields_by_name: &mut BTreeMap<&'a str, &'a Field>,
    name: &str,
    role: &'static str,
) -> Result<&'a Field> {
    fields_by_name.remove(name).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.aggregate.partial_sizing",
            "partial aggregate state field is missing from its derived schema",
        )
        .with_fact("field", name)
        .with_fact("role", role)
    })
}

fn fixed_partial_column(field: &Field) -> JoinOutputColumnUpperBoundInput {
    JoinOutputColumnUpperBoundInput {
        data_type: field.data_type().clone(),
        input_max_string_length_bytes: None,
        input_total_string_bytes: None,
    }
}

fn partial_column_for_field_and_optional_input(
    execution: &mut BoundedExecutionAttempt<'_>,
    field: &Field,
    input_expr: Option<&Expr>,
    input_schema: &ArrowSchemaRef,
    input: &GpuDataFrame,
) -> Result<JoinOutputColumnUpperBoundInput> {
    if !partial_sizing_dtype_needs_string_max(field.data_type()) {
        return Ok(fixed_partial_column(field));
    }
    let Some(input_expr) = input_expr else {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.aggregate.partial_sizing",
            "string partial aggregate state requires an input column for max-length sizing",
        )
        .with_fact("field", field.name().as_str())
        .with_fact("data_type", format!("{:?}", field.data_type())));
    };
    let input_column = lower_column_expr(input_expr, "query_engine.bounded.aggregate.partial_sizing_string")?;
    let schema_field = input_schema.field_with_name(input_column.as_str()).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.aggregate.partial_sizing",
            "string partial aggregate input column is missing from input schema",
        )
        .with_fact("column", input_column.as_str())
        .with_fact("field", field.name().as_str())
        .with_fact("arrow_error", error.to_string())
    })?;
    if !partial_sizing_dtype_needs_string_max(schema_field.data_type()) {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.aggregate.partial_sizing",
            "string partial aggregate output was derived from a non-string input column",
        )
        .with_fact("column", input_column.as_str())
        .with_fact("input_type", format!("{:?}", schema_field.data_type()))
        .with_fact("output_type", format!("{:?}", field.data_type())));
    }
    partial_column_for_field_and_input_column(execution, field, input_column.as_str(), input)
}

fn partial_column_for_field_and_input_column(
    execution: &mut BoundedExecutionAttempt<'_>,
    field: &Field,
    input_column: &str,
    input: &GpuDataFrame,
) -> Result<JoinOutputColumnUpperBoundInput> {
    let (input_max_string_length_bytes, input_total_string_bytes) =
        if partial_sizing_dtype_needs_string_max(field.data_type()) {
            let stats = string_byte_length_stats_for_partial_sizing(execution, input, input_column)?;
            (Some(stats.max_len), Some(stats.total_bytes))
        } else {
            (None, None)
        };
    Ok(JoinOutputColumnUpperBoundInput {
        data_type: field.data_type().clone(),
        input_max_string_length_bytes,
        input_total_string_bytes,
    })
}

#[derive(Debug, Clone, Copy)]
struct StringByteLengthStats {
    max_len: u64,
    total_bytes: u64,
}

fn string_byte_length_stats_for_partial_sizing(
    execution: &mut BoundedExecutionAttempt<'_>,
    input: &GpuDataFrame,
    input_column: &str,
) -> Result<StringByteLengthStats> {
    if input.num_rows() == 0 {
        return Ok(StringByteLengthStats { max_len: 0, total_bytes: 0 });
    }
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "compiled_aggregate_string_length_sizing",
        "query_engine.bounded.aggregate.string_length_sizing.budget_token",
        "failed to create bounded native partial aggregate string-length sizing budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            return Err(error);
        },
    };

    let series = match input.frame().series(input_column) {
        Ok(series) => series,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.string_length_sizing",
                "failed to access string input column for partial aggregate sizing",
                error,
            )
            .with_fact("column", input_column));
        },
    };
    let byte_counts = match series.on(token.exec()).str().byte_count() {
        Ok(byte_counts) => byte_counts,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.string_length_sizing",
                "failed to compute string byte counts for partial aggregate sizing",
                error,
            )
            .with_fact("column", input_column));
        },
    };
    let max_len = match byte_counts.on(token.exec()).max_value::<i32>() {
        Ok(max_len) => max_len,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.string_length_sizing",
                "failed to reduce string byte-count sizing column",
                error,
            )
            .with_fact("column", input_column));
        },
    };
    let total_bytes = match byte_counts.on(token.exec()).sum_value::<i64>() {
        Ok(total_bytes) => total_bytes,
        Err(error) => {
            return Err(cudf_query_error(
                "query_engine.bounded.aggregate.string_length_sizing",
                "failed to sum string byte-count sizing column",
                error,
            )
            .with_fact("column", input_column));
        },
    };
    Ok(StringByteLengthStats { max_len: max_len.unwrap_or(0).max(0) as u64, total_bytes: total_bytes.max(0) as u64 })
}

fn aggregate_input_data_type(expr: &Expr, input_schema: &ArrowSchemaRef) -> Result<ArrowDataType> {
    let input_column = lower_column_expr(expr, "query_engine.bounded.aggregate.partial_sizing_input_type")?;
    input_schema.field_with_name(input_column.as_str()).map(|field| field.data_type().clone()).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.bounded.aggregate.partial_sizing",
            "aggregate input column is missing from input schema",
        )
        .with_fact("column", input_column.as_str())
        .with_fact("arrow_error", error.to_string())
    })
}

fn partial_sizing_dtype_needs_string_max(data_type: &ArrowDataType) -> bool {
    matches!(data_type, ArrowDataType::Utf8 | ArrowDataType::LargeUtf8 | ArrowDataType::Utf8View)
}
