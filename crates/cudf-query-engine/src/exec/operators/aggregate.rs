use std::collections::HashMap;

use arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray, UInt64Array,
};

use super::*;

pub(in crate::exec) fn validate_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_stddev_samp(&node.spec) {
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
                "cudf_query_engine.aggregate.validate_stddev_exact",
                "DataFusion-exact stddev aggregate output names must match output schema fields",
            )
            .with_fact("aggregate_output_names", expected_output_names.join(","))
            .with_fact("schema_output_names", schema_output_names.join(",")));
        }
        return Ok(());
    }
    let aggregate = lower_aggregate_spec(&node.spec)?;
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.aggregate.validate",
        "aggregate output names must match output schema fields",
    )?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    Ok(())
}

pub(in crate::exec) fn validate_final_aggregate_capability(
    node: &AggregateNode,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if node.spec.group_exprs.is_empty() {
        return validate_global_aggregate_capability(&node.spec, output_schema);
    }
    if aggregate_spec_has_stddev_samp(&node.spec) {
        return validate_aggregate_capability(node, output_schema);
    }
    let aggregate = lower_final_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_state_alias_collisions(&node.spec, "cudf_query_engine.final_merge.validate_avg_state_alias")?;
        validate_avg_output_fields(&node.spec, output_schema)?;
        return Ok(());
    }
    validate_aggregate_output_names(
        &aggregate,
        output_schema,
        "cudf_query_engine.final_merge.validate",
        "final aggregate output names must match output schema fields",
    )
}

pub(in crate::exec) fn validate_global_aggregate_capability(
    spec: &AggregateSpec,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<()> {
    if aggregate_spec_has_avg(spec) {
        validate_avg_state_alias_collisions(spec, "cudf_query_engine.global_aggregate.validate_avg_state_alias")?;
    }
    let expected_output_names = spec.aggr_exprs.iter().map(|expr| expr.alias.clone()).collect::<Vec<_>>();
    let schema_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if expected_output_names != schema_output_names {
        return Err(QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.global_aggregate.validate",
            "global aggregate output names must match output schema fields",
        )
        .with_fact("aggregate_output_names", expected_output_names.join(","))
        .with_fact("schema_output_names", schema_output_names.join(",")));
    }
    for expr in &spec.aggr_exprs {
        if expr.function == AggregateFunction::Avg {
            validate_avg_output_fields(spec, output_schema)?;
        }
        if expr.function == AggregateFunction::StddevSamp {
            let field = output_schema.field_with_name(expr.alias.as_str()).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "cudf_query_engine.global_aggregate.validate_stddev_exact",
                    "stddev_samp aggregate output field is missing from output schema",
                )
                .with_fact("alias", expr.alias.as_str())
            })?;
            if !matches!(field.data_type(), ArrowDataType::Float64) {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "cudf_query_engine.global_aggregate.validate_stddev_exact",
                    "stddev_samp aggregate output must be Float64",
                )
                .with_fact("alias", expr.alias.as_str())
                .with_fact("data_type", format!("{:?}", field.data_type())));
            }
        }
        validate_global_aggregate_input_expr(expr)?;
    }
    Ok(())
}

pub(in crate::exec) fn validate_global_aggregate_input_expr(expr: &AggregateExpr) -> Result<()> {
    match expr.function {
        AggregateFunction::Count if global_count_star_expr(&expr.expr) => Ok(()),
        _ => lower_column_expr(&expr.expr, "cudf_query_engine.global_aggregate.validate_expr").map(|_| ()),
    }
}

pub(in crate::exec) fn validate_aggregate_output_names(
    aggregate: &LoweredAggregate,
    output_schema: &arrow::datatypes::SchemaRef,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();
    if aggregate.output_names != expected_output_names {
        return Err(QueryError::new(QueryErrorCode::InvalidPlan, operation, message)
            .with_fact("aggregate_output_names", aggregate.output_names.join(","))
            .with_fact("schema_output_names", expected_output_names.join(",")));
    }

    Ok(())
}

pub(in crate::exec) fn execute_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_stddev_samp(&node.spec) {
        return execute_datafusion_exact_host_aggregate(context, node, input_schema, input, output_schema, metrics);
    }
    if node.spec.group_exprs.is_empty() {
        return execute_global_aggregate(context, &node.spec, input, output_schema, metrics);
    }

    // Lowering AVG through libcudf's groupby `MEAN` (`GroupByAgg::Mean`) keeps
    // the result scale equal to the source scale on `fixed_point` columns and
    // truncates the division remainder, so a `Decimal128(7, 2)` input averaged
    // to a `Decimal128(11, 6)` output rounds e.g. `34.65 / 2` to `17.32`
    // instead of `17.325`. Route AVG through the same explicit `sum + count +
    // finalize` state used by the partial/final and global aggregate paths so
    // the division happens at the output scale.
    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate(&overrides, frame, &aggregate, "cudf_query_engine.aggregate.execute")?;
    output = finalize_avg_output_columns(context, output, &node.spec, output_schema)?;
    output = cast_reduction_output_columns(output, &node.spec, output_schema)?;
    output = cast_count_output_columns(output, &node.spec, output_schema)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if aggregate_spec_has_stddev_samp(&node.spec) {
        return execute_datafusion_exact_host_aggregate(context, node, input_schema, input, output_schema, metrics);
    }
    if node.spec.group_exprs.is_empty() {
        return execute_global_aggregate_with_budget(context, &node.spec, input, output_schema, token, metrics);
    }

    // See `execute_aggregate` for why AVG must not lower to libcudf groupby
    // `MEAN`; the same finalize-at-output-scale contract applies here.
    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    if aggregate_spec_has_avg(&node.spec) {
        validate_avg_output_fields(&node.spec, output_schema)?;
    }
    let expected_output_names = output_schema.fields().iter().map(|field| field.name().clone()).collect::<Vec<_>>();

    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute",
        "failed to execute bounded native cudf aggregate",
    )?;
    output = finalize_avg_output_columns_with_budget(context, output, &node.spec, output_schema, token, metrics)?;
    output = cast_reduction_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
    output = cast_count_output_columns_with_budget(output, &node.spec, output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select",
            "failed to select native cudf aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_partial_aggregate_with_budget(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    aggregate_output_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    if node.spec.group_exprs.is_empty() {
        let partial_spec = partial_global_aggregate_spec(&node.spec);
        let output_schema = partial_global_aggregate_output_schema(&node.spec, input_schema, aggregate_output_schema)?;
        return execute_global_aggregate_with_budget(context, &partial_spec, input, &output_schema, token, metrics);
    }

    let aggregate = lower_partial_aggregate_spec(&node.spec)?;
    let expected_output_names = aggregate.output_names.clone();
    let frame = input.into_frame();
    let overrides = execution_overrides(context);
    let timer = Instant::now();
    let mut output = execute_groupby_aggregate_with_budget(
        &overrides,
        frame,
        &aggregate,
        token,
        metrics,
        "cudf_query_engine.aggregate.execute_partial",
        "failed to execute bounded native partial cudf aggregate",
    )?;
    output = cast_count_output_columns_with_budget(output, &node.spec, aggregate_output_schema, token, metrics)?;
    output = output.select(expected_output_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_partial",
            "failed to select native partial aggregate output columns",
            error,
        )
    })?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn execute_groupby_aggregate(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct(overrides, frame, aggregate, operation);
    }

    let mut output = frame
        .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
        .options(GroupByOptions::builder().include_null_keys(true).build())
        .with_execution(overrides)
        .agg(&aggregate.aggregations)
        .map_err(|error| QueryError::from_cudf(operation, "failed to execute native cudf aggregate", error))?;
    rename_groupby_key_columns(&mut output, aggregate, "cudf_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

pub(in crate::exec) fn execute_groupby_aggregate_with_budget(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    metrics: &mut QueryMetrics,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    if aggregate.aggregations.is_empty() {
        return execute_groupby_distinct_with_budget(overrides, frame, aggregate, token, operation, message);
    }

    let mut output = budgeted_value(
        frame
            .groupby(aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>())
            .options(GroupByOptions::builder().include_null_keys(true).build())
            .agg_with_budget(&aggregate.aggregations, token)
            .map_err(|error| QueryError::from_cudf_with_budget_details(operation, message, error))?,
        metrics,
        operation,
    );
    rename_groupby_key_columns(&mut output, aggregate, "cudf_query_engine.aggregate.rename_keys")?;
    Ok(output)
}

fn rename_groupby_key_columns(
    output: &mut DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<()> {
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        *output = output
            .rename_columns(key_renames)
            .map_err(|error| QueryError::from_cudf(operation, "failed to rename native cudf aggregate keys", error))?;
    }
    Ok(())
}

fn execute_groupby_distinct(
    overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    operation: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let mut output = frame
        .with_execution(overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| {
            QueryError::from_cudf(operation, "failed to execute native cudf group-only aggregate", error)
        })?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    output.select(aggregate.output_key_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_group_only_keys",
            "failed to select native group-only aggregate keys",
            error,
        )
    })
}

fn execute_groupby_distinct_with_budget(
    _overrides: &ExecutionOverrides<'_>,
    frame: DataFrame,
    aggregate: &LoweredAggregate,
    token: &ExecutionBudgetToken,
    operation: &'static str,
    message: &'static str,
) -> Result<DataFrame> {
    let key_columns = aggregate.key_columns.iter().map(String::as_str).collect::<Vec<_>>();
    let budget_overrides = token.execution_overrides();
    let mut output = frame
        .with_execution(&budget_overrides)
        .drop_duplicates(
            DropDuplicatesOptions::builder().subset(key_columns.as_slice()).keep(DuplicateKeepOption::KeepAny).build(),
        )
        .map_err(|error| attach_budget_token_failure_facts(QueryError::from_cudf(operation, message, error), token))?;
    let key_renames = aggregate
        .key_columns
        .iter()
        .zip(&aggregate.output_key_names)
        .filter(|(input, output)| input != output)
        .map(|(input, output)| (input.as_str(), output.as_str()))
        .collect::<Vec<_>>();
    if !key_renames.is_empty() {
        output = output.rename_columns(key_renames).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.aggregate.rename_group_only_keys",
                "failed to rename native group-only aggregate keys",
                error,
            )
        })?;
    }
    output.select(aggregate.output_key_names.iter().map(String::as_str).collect::<Vec<_>>()).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.select_group_only_keys",
            "failed to select native group-only aggregate keys",
            error,
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum HostGroupKeyValue {
    Null,
    Boolean(bool),
    Int32(i32),
    Int64(i64),
    UInt64(u64),
    Float64(u64),
    Utf8(String),
}

#[derive(Debug, Clone)]
enum HostAggregateState {
    Count { count: u64 },
    Sum { sum: f64, seen: bool },
    Min { value: f64, seen: bool },
    Max { value: f64, seen: bool },
    Avg { sum: f64, count: u64 },
    StddevSamp { count: u64, mean: f64, m2: f64 },
}

impl HostAggregateState {
    fn new(function: AggregateFunction) -> Self {
        match function {
            AggregateFunction::Count => Self::Count { count: 0 },
            AggregateFunction::Sum => Self::Sum { sum: 0.0, seen: false },
            AggregateFunction::Min => Self::Min { value: 0.0, seen: false },
            AggregateFunction::Max => Self::Max { value: 0.0, seen: false },
            AggregateFunction::Avg => Self::Avg { sum: 0.0, count: 0 },
            AggregateFunction::StddevSamp => Self::StddevSamp { count: 0, mean: 0.0, m2: 0.0 },
        }
    }

    fn update(&mut self, value: Option<f64>) {
        match self {
            Self::Count { count } => {
                if value.is_some() {
                    *count += 1;
                }
            },
            Self::Sum { sum, seen } => {
                if let Some(value) = value {
                    *sum += value;
                    *seen = true;
                }
            },
            Self::Min { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value < *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Max { value: current, seen } => {
                if let Some(value) = value {
                    if !*seen || value > *current {
                        *current = value;
                    }
                    *seen = true;
                }
            },
            Self::Avg { sum, count } => {
                if let Some(value) = value {
                    *sum += value;
                    *count += 1;
                }
            },
            Self::StddevSamp { count, mean, m2 } => {
                if let Some(value) = value {
                    let new_count = *count + 1;
                    let delta1 = value - *mean;
                    let new_mean = delta1 / new_count as f64 + *mean;
                    let delta2 = value - new_mean;
                    *m2 += delta1 * delta2;
                    *mean = new_mean;
                    *count = new_count;
                }
            },
        }
    }

    fn evaluate_f64(&self) -> Option<f64> {
        match self {
            Self::Count { count } => Some(*count as f64),
            Self::Sum { sum, seen } => seen.then_some(*sum),
            Self::Min { value, seen } | Self::Max { value, seen } => seen.then_some(*value),
            Self::Avg { sum, count } => (*count > 0).then_some(*sum / *count as f64),
            Self::StddevSamp { count, m2, .. } => {
                if *count <= 1 {
                    None
                } else {
                    Some((*m2 / (*count - 1) as f64).sqrt())
                }
            },
        }
    }

    fn evaluate_i64(&self) -> Option<i64> {
        match self {
            Self::Count { count } => Some(*count as i64),
            _ => self.evaluate_f64().map(|value| value as i64),
        }
    }
}

struct HostAggregateGroup {
    keys: Vec<HostGroupKeyValue>,
    aggregates: Vec<HostAggregateState>,
}

fn execute_datafusion_exact_host_aggregate(
    context: &ExecutionContext,
    node: &AggregateNode,
    input_schema: &arrow::datatypes::SchemaRef,
    input: GpuDataFrame,
    output_schema: &arrow::datatypes::SchemaRef,
    metrics: &mut QueryMetrics,
) -> Result<GpuDataFrame> {
    let timer = Instant::now();
    let frame = sort_by_datafusion_order_if_present(context, input.into_frame(), metrics)?;
    let schema = schema_with_datafusion_order_columns(&frame, input_schema);
    let handle = GpuResultHandle::new(context, GpuDataFrame::new(frame), schema)?;
    let batch = handle.to_host_record_batch()?;
    let output = execute_datafusion_exact_host_aggregate_batch(context, node, &batch, output_schema)?;
    metrics.operator_time += timer.elapsed();
    Ok(GpuDataFrame::new(output))
}

fn execute_datafusion_exact_host_aggregate_batch(
    context: &ExecutionContext,
    node: &AggregateNode,
    batch: &RecordBatch,
    output_schema: &arrow::datatypes::SchemaRef,
) -> Result<DataFrame> {
    let key_columns = node
        .spec
        .group_exprs
        .iter()
        .map(|expr| lower_named_column_expr(expr, "cudf_query_engine.aggregate.stddev_exact.lower_key"))
        .collect::<Result<Vec<_>>>()?;
    let aggregate_columns =
        node.spec.aggr_exprs.iter().map(|expr| aggregate_input_column(expr)).collect::<Result<Vec<_>>>()?;
    let mut group_indices = HashMap::<Vec<HostGroupKeyValue>, usize>::new();
    let mut groups = Vec::<HostAggregateGroup>::new();
    let row_count = batch.num_rows();

    for row in 0..row_count {
        let key = key_columns
            .iter()
            .map(|(input_name, _)| host_group_key_value(batch, input_name, row))
            .collect::<Result<Vec<_>>>()?;
        let group_index = match group_indices.get(&key) {
            Some(index) => *index,
            None => {
                let index = groups.len();
                group_indices.insert(key.clone(), index);
                groups.push(HostAggregateGroup {
                    keys: key,
                    aggregates: node
                        .spec
                        .aggr_exprs
                        .iter()
                        .map(|expr| HostAggregateState::new(expr.function))
                        .collect(),
                });
                index
            },
        };
        for ((expr, input_column), state) in
            node.spec.aggr_exprs.iter().zip(&aggregate_columns).zip(&mut groups[group_index].aggregates)
        {
            let value = match input_column {
                HostAggregateInput::Column(name) => host_numeric_value(batch, name, row)?,
                HostAggregateInput::NonNullLiteral => Some(1.0),
                HostAggregateInput::NullLiteral => None,
            };
            if matches!(expr.function, AggregateFunction::Count) {
                state.update(value.map(|_| 1.0));
            } else {
                state.update(value);
            }
        }
    }

    if key_columns.is_empty() && groups.is_empty() {
        groups.push(HostAggregateGroup {
            keys: Vec::new(),
            aggregates: node.spec.aggr_exprs.iter().map(|expr| HostAggregateState::new(expr.function)).collect(),
        });
    }

    let mut arrays = Vec::<ArrayRef>::with_capacity(output_schema.fields().len());
    for (output_index, field) in output_schema.fields().iter().enumerate() {
        if output_index < key_columns.len() {
            arrays.push(host_key_output_array(&groups, output_index, field.data_type())?);
        } else {
            let aggregate_index = output_index - key_columns.len();
            arrays.push(host_aggregate_output_array(&groups, aggregate_index, field.data_type())?);
        }
    }
    let output = RecordBatch::try_new(Arc::clone(output_schema), arrays).map_err(|error| {
        QueryError::new(
            QueryErrorCode::Internal,
            "cudf_query_engine.aggregate.stddev_exact.output",
            "failed to construct DataFusion-exact stddev aggregate Arrow output",
        )
        .with_source(error)
    })?;
    DataFrame::from_record_batch(context.runtime(), &output).map_err(|error| {
        QueryError::from_cudf(
            "cudf_query_engine.aggregate.stddev_exact.output",
            "failed to upload DataFusion-exact stddev aggregate output",
            error,
        )
    })
}

fn host_group_expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias { alias, .. } => alias.clone(),
        Expr::Column(column) => column.display_name().to_string(),
        _ => expr.to_string(),
    }
}

enum HostAggregateInput {
    Column(String),
    NonNullLiteral,
    NullLiteral,
}

fn aggregate_input_column(expr: &AggregateExpr) -> Result<HostAggregateInput> {
    if matches!(expr.function, AggregateFunction::Count) && global_count_star_expr(&expr.expr) {
        return Ok(HostAggregateInput::NonNullLiteral);
    }
    aggregate_input_expr(&expr.expr)
}

fn aggregate_input_expr(expr: &Expr) -> Result<HostAggregateInput> {
    match expr {
        Expr::Column(column) => Ok(HostAggregateInput::Column(column.name().to_string())),
        Expr::Alias { expr, .. } => aggregate_input_expr(expr),
        Expr::Literal(LiteralValue::Null(_)) => Ok(HostAggregateInput::NullLiteral),
        Expr::Literal(_) => Ok(HostAggregateInput::NonNullLiteral),
        _ => Err(QueryError::new(
            QueryErrorCode::UnsupportedExpression,
            "cudf_query_engine.aggregate.stddev_exact.lower_expr",
            "DataFusion-exact stddev aggregate requires prepared column aggregate inputs",
        )
        .with_fact("expression", expr.to_string())),
    }
}

fn host_group_key_value(batch: &RecordBatch, column: &str, row: usize) -> Result<HostGroupKeyValue> {
    let array = batch.column(batch.schema().index_of(column).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.aggregate.stddev_exact.key",
            "group key column is missing from aggregate input batch",
        )
        .with_fact("column", column)
        .with_source(error)
    })?);
    if array.is_null(row) {
        return Ok(HostGroupKeyValue::Null);
    }
    match array.data_type() {
        ArrowDataType::Boolean => Ok(HostGroupKeyValue::Boolean(
            array.as_any().downcast_ref::<BooleanArray>().expect("validated BooleanArray").value(row),
        )),
        ArrowDataType::Int32 => Ok(HostGroupKeyValue::Int32(
            array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array").value(row),
        )),
        ArrowDataType::Date32 => Ok(HostGroupKeyValue::Int32(
            array.as_any().downcast_ref::<Date32Array>().expect("validated Date32Array").value(row),
        )),
        ArrowDataType::Int64 => Ok(HostGroupKeyValue::Int64(
            array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array").value(row),
        )),
        ArrowDataType::UInt64 => Ok(HostGroupKeyValue::UInt64(
            array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array").value(row),
        )),
        ArrowDataType::Float64 => Ok(HostGroupKeyValue::Float64(
            array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array").value(row).to_bits(),
        )),
        ArrowDataType::Utf8 => Ok(HostGroupKeyValue::Utf8(
            array.as_any().downcast_ref::<StringArray>().expect("validated StringArray").value(row).to_string(),
        )),
        ArrowDataType::LargeUtf8 => Ok(HostGroupKeyValue::Utf8(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("validated LargeStringArray")
                .value(row)
                .to_string(),
        )),
        ArrowDataType::Utf8View => Ok(HostGroupKeyValue::Utf8(
            array.as_any().downcast_ref::<StringViewArray>().expect("validated StringViewArray").value(row).to_string(),
        )),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.key",
            "DataFusion-exact stddev aggregate does not support this group key type",
        )
        .with_fact("column", column)
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_numeric_value(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<f64>> {
    let array = batch.column(batch.schema().index_of(column).map_err(|error| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "cudf_query_engine.aggregate.stddev_exact.value",
            "aggregate input column is missing from aggregate input batch",
        )
        .with_fact("column", column)
        .with_source(error)
    })?);
    if array.is_null(row) {
        return Ok(None);
    }
    match array.data_type() {
        ArrowDataType::Int32 => {
            Ok(Some(array.as_any().downcast_ref::<Int32Array>().expect("validated Int32Array").value(row) as f64))
        },
        ArrowDataType::Int64 => {
            Ok(Some(array.as_any().downcast_ref::<Int64Array>().expect("validated Int64Array").value(row) as f64))
        },
        ArrowDataType::UInt64 => {
            Ok(Some(array.as_any().downcast_ref::<UInt64Array>().expect("validated UInt64Array").value(row) as f64))
        },
        ArrowDataType::Float64 => {
            Ok(Some(array.as_any().downcast_ref::<Float64Array>().expect("validated Float64Array").value(row)))
        },
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.value",
            "DataFusion-exact stddev aggregate requires numeric aggregate inputs",
        )
        .with_fact("column", column)
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_key_output_array(
    groups: &[HostAggregateGroup],
    key_index: usize,
    data_type: &ArrowDataType,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Boolean => Ok(Arc::new(BooleanArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Boolean(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int32 => Ok(Arc::new(Int32Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int32(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Date32 => Ok(Arc::new(Date32Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int32(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Int64(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::UInt64 => Ok(Arc::new(UInt64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::UInt64(value)) => Some(*value),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Float64(value)) => Some(f64::from_bits(*value)),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8 => Ok(Arc::new(StringArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.clone()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::LargeUtf8 => Ok(Arc::new(LargeStringArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.clone()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Utf8View => Ok(Arc::new(StringViewArray::from(
            groups
                .iter()
                .map(|group| match group.keys.get(key_index) {
                    Some(HostGroupKeyValue::Utf8(value)) => Some(value.as_str()),
                    Some(HostGroupKeyValue::Null) => None,
                    _ => None,
                })
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.key_output",
            "DataFusion-exact stddev aggregate does not support this output group key type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}

fn host_aggregate_output_array(
    groups: &[HostAggregateGroup],
    aggregate_index: usize,
    data_type: &ArrowDataType,
) -> Result<ArrayRef> {
    match data_type {
        ArrowDataType::Int64 => Ok(Arc::new(Int64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_i64))
                .collect::<Vec<_>>(),
        ))),
        ArrowDataType::Float64 => Ok(Arc::new(Float64Array::from(
            groups
                .iter()
                .map(|group| group.aggregates.get(aggregate_index).and_then(HostAggregateState::evaluate_f64))
                .collect::<Vec<_>>(),
        ))),
        data_type => Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "cudf_query_engine.aggregate.stddev_exact.aggregate_output",
            "DataFusion-exact stddev aggregate does not support this aggregate output type",
        )
        .with_fact("data_type", format!("{data_type:?}"))),
    }
}
