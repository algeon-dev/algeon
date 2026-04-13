use super::*;

pub(super) fn dimension_scan_admitted(source: &LocalParquetSource, max_dimension_rows: u64) -> bool {
    let mut total_rows = 0u64;
    for file in &source.files {
        let Some(path) = file.location.as_local_path() else {
            return false;
        };
        let Ok(reader) = SerializedFileReader::try_from(path) else {
            return false;
        };
        let metadata = reader.metadata();
        let row_groups = selected_or_all_row_groups(file, metadata.num_row_groups());
        for row_group in row_groups {
            let Some(row_group_metadata) = metadata.row_group(row_group).num_rows().try_into().ok() else {
                return false;
            };
            total_rows = total_rows.saturating_add(row_group_metadata);
            if total_rows > max_dimension_rows {
                return false;
            }
        }
    }
    true
}

pub(super) fn derive_dimension_key_filter(
    source: &LocalParquetSource,
    key_column: &str,
) -> Option<Option<DimensionKeyFilter>> {
    let predicate = source.reader_predicate.as_ref()?;
    let projection_indices = reader_projection_indices(source, key_column, predicate)?;
    let mut accumulator = DimensionKeyAccumulator::default();
    for source_file in &source.files {
        let path = source_file.location.as_local_path()?;
        let file = File::open(path).ok()?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
        let row_group_count = builder.metadata().num_row_groups();
        let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
        let row_groups = selected_or_all_row_groups(source_file, row_group_count);
        let reader = builder.with_projection(mask).with_row_groups(row_groups).build().ok()?;
        for batch in reader {
            let batch = batch.ok()?;
            accumulator.observe_batch(&batch, key_column, predicate)?;
        }
    }
    Some(accumulator.finish())
}

pub(crate) fn derive_i64_expr_range_from_filtered_subtree(
    plan: &QueryPlan,
    node_id: NodeId,
    expr: &Expr,
    max_rows: u64,
) -> Option<Option<KeyRange>> {
    derive_i64_expr_range_from_filtered_subtree_inner(plan, node_id, expr.clone(), None, max_rows)
}

fn derive_i64_expr_range_from_filtered_subtree_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    expr: Expr,
    predicate: Option<Expr>,
    max_rows: u64,
) -> Option<Option<KeyRange>> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => {
            let SourceNode::LocalParquet(source) = source.as_ref() else {
                return None;
            };
            if !dimension_scan_admitted(source, max_rows) {
                return None;
            }
            let predicate = combine_predicates(source.reader_predicate.clone(), predicate);
            derive_i64_expr_range_from_local_parquet_source(source, &expr, predicate.as_ref())
        },
        PlanNode::Filter(filter) => derive_i64_expr_range_from_filtered_subtree_inner(
            plan,
            filter.input,
            expr,
            combine_predicates(predicate, Some(filter.predicate.clone())),
            max_rows,
        ),
        PlanNode::Projection(projection) => {
            let expr = remap_projection_general_expr(&expr, projection)?;
            let predicate = remap_optional_projection_general_expr(predicate, projection)?;
            derive_i64_expr_range_from_filtered_subtree_inner(plan, projection.input, expr, predicate, max_rows)
        },
        PlanNode::Aggregate(aggregate) => {
            if aggregate.spec.grouping_sets.is_some() || !aggregate.spec.aggr_exprs.is_empty() {
                return None;
            }
            let expr = remap_aggregate_group_general_expr(&expr, aggregate)?;
            let predicate = remap_optional_aggregate_group_general_expr(predicate, aggregate)?;
            derive_i64_expr_range_from_filtered_subtree_inner(plan, aggregate.input, expr, predicate, max_rows)
        },
        PlanNode::Sort(sort) => {
            derive_i64_expr_range_from_filtered_subtree_inner(plan, sort.input, expr, predicate, max_rows)
        },
        PlanNode::Join(_)
        | PlanNode::Limit(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn remap_optional_projection_general_expr(
    expr: Option<Expr>,
    projection: &crate::plan::ProjectionNode,
) -> Option<Option<Expr>> {
    match expr {
        Some(expr) => remap_projection_general_expr(&expr, projection).map(Some),
        None => Some(None),
    }
}

fn remap_optional_aggregate_group_general_expr(
    expr: Option<Expr>,
    aggregate: &crate::plan::AggregateNode,
) -> Option<Option<Expr>> {
    match expr {
        Some(expr) => remap_aggregate_group_general_expr(&expr, aggregate).map(Some),
        None => Some(None),
    }
}

fn remap_projection_general_expr(expr: &Expr, projection: &crate::plan::ProjectionNode) -> Option<Expr> {
    remap_unqualified_columns(expr, &mut |column| {
        let output = projection.outputs.iter().find(|output| output.field.name() == column.name())?;
        Some(strip_alias(output.expr.clone()))
    })
}

fn remap_aggregate_group_general_expr(expr: &Expr, aggregate: &crate::plan::AggregateNode) -> Option<Expr> {
    remap_unqualified_columns(expr, &mut |column| {
        aggregate
            .spec
            .group_exprs
            .iter()
            .find(|expr| aggregate_group_output_name(expr) == Some(column.name()))
            .cloned()
            .map(strip_alias)
    })
}

fn strip_alias(expr: Expr) -> Expr {
    match expr {
        Expr::Alias { expr, .. } => strip_alias(*expr),
        _ => expr,
    }
}

fn combine_predicates(left: Option<Expr>, right: Option<Expr>) -> Option<Expr> {
    match (left, right) {
        (Some(left), Some(right)) => Some(Expr::binary(left, BinaryOp::And, right)),
        (Some(expr), None) | (None, Some(expr)) => Some(expr),
        (None, None) => None,
    }
}

fn derive_i64_expr_range_from_local_parquet_source(
    source: &LocalParquetSource,
    expr: &Expr,
    predicate: Option<&Expr>,
) -> Option<Option<KeyRange>> {
    let projection_indices = reader_projection_indices_for_exprs(source, expr, predicate)?;
    let mut range = None;
    for source_file in &source.files {
        let path = source_file.location.as_local_path()?;
        let file = File::open(path).ok()?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
        let row_group_count = builder.metadata().num_row_groups();
        let mask = ProjectionMask::roots(builder.parquet_schema(), projection_indices.iter().copied());
        let row_groups = selected_or_all_row_groups(source_file, row_group_count);
        let reader = builder.with_projection(mask).with_row_groups(row_groups).build().ok()?;
        for batch in reader {
            let batch = batch.ok()?;
            observe_i64_expr_range_batch(&mut range, &batch, expr, predicate)?;
        }
    }
    Some(range)
}

fn observe_i64_expr_range_batch(
    range: &mut Option<KeyRange>,
    batch: &RecordBatch,
    expr: &Expr,
    predicate: Option<&Expr>,
) -> Option<()> {
    for row in 0..batch.num_rows() {
        if let Some(predicate) = predicate
            && !eval_bool(predicate, batch, row)?
        {
            continue;
        }
        let value = match eval_scalar(expr, batch, row)? {
            Scalar::Null => continue,
            value => scalar_to_i64(value)?,
        };
        *range = Some(match *range {
            Some(existing) => KeyRange { min: existing.min.min(value), max: existing.max.max(value) },
            None => KeyRange { min: value, max: value },
        });
    }
    Some(())
}

#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    String(String),
    Null,
}

pub(super) fn eval_bool(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<bool> {
    match eval_scalar(expr, batch, row)? {
        Scalar::Boolean(value) => Some(value),
        Scalar::Null => Some(false),
        _ => None,
    }
}

fn eval_scalar(expr: &Expr, batch: &RecordBatch, row: usize) -> Option<Scalar> {
    match expr {
        Expr::Column(column) => array_scalar(batch.column(batch.schema().index_of(column.name()).ok()?).as_ref(), row),
        Expr::Literal(value) => literal_scalar(value),
        Expr::Alias { expr, .. } => eval_scalar(expr, batch, row),
        Expr::Cast { expr, target_type } => cast_scalar(eval_scalar(expr, batch, row)?, target_type),
        Expr::Binary { left, op, right, .. } => {
            eval_binary(eval_scalar(left, batch, row)?, *op, eval_scalar(right, batch, row)?)
        },
        Expr::Not(inner) => match eval_scalar(inner, batch, row)? {
            Scalar::Boolean(value) => Some(Scalar::Boolean(!value)),
            Scalar::Null => Some(Scalar::Null),
            _ => None,
        },
        Expr::IsNull(inner) => Some(Scalar::Boolean(matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::IsNotNull(inner) => Some(Scalar::Boolean(!matches!(eval_scalar(inner, batch, row)?, Scalar::Null))),
        Expr::InList { expr, list, negated } => {
            let value = eval_scalar(expr, batch, row)?;
            if matches!(value, Scalar::Null) {
                return Some(Scalar::Null);
            }
            let mut matched = false;
            for item in list {
                if scalar_eq(&value, &eval_scalar(item, batch, row)?)? {
                    matched = true;
                    break;
                }
            }
            Some(Scalar::Boolean(if *negated { !matched } else { matched }))
        },
        Expr::Like { expr, pattern, escape, negated } => match eval_scalar(expr, batch, row)? {
            Scalar::String(value) => {
                let matched = like_matches(&value, pattern, escape.as_deref())?;
                Some(Scalar::Boolean(if *negated { !matched } else { matched }))
            },
            Scalar::Null => Some(Scalar::Null),
            _ => None,
        },
        Expr::TryCast { .. }
        | Expr::Substring { .. }
        | Expr::Lower(_)
        | Expr::Upper(_)
        | Expr::Trim { .. }
        | Expr::LTrim { .. }
        | Expr::RTrim { .. }
        | Expr::StringContains { .. }
        | Expr::StartsWith { .. }
        | Expr::EndsWith { .. }
        | Expr::StringLength { .. }
        | Expr::RegexpReplace { .. }
        | Expr::Concat(_)
        | Expr::Abs(_)
        | Expr::Sqrt { .. }
        | Expr::Round { .. }
        | Expr::IsDistinctFrom { .. }
        | Expr::IsNotDistinctFrom { .. }
        | Expr::Case { .. }
        | Expr::Extract { .. }
        | Expr::DateTrunc { .. } => None,
    }
}

fn array_scalar(array: &dyn Array, row: usize) -> Option<Scalar> {
    if array.is_null(row) {
        return Some(Scalar::Null);
    }
    match array.data_type() {
        DataType::Boolean => Some(Scalar::Boolean(array.as_any().downcast_ref::<BooleanArray>()?.value(row))),
        DataType::Int32 => Some(Scalar::Int64(i64::from(array.as_any().downcast_ref::<Int32Array>()?.value(row)))),
        DataType::Date32 => Some(Scalar::Int64(i64::from(array.as_any().downcast_ref::<Date32Array>()?.value(row)))),
        DataType::Int64 => Some(Scalar::Int64(array.as_any().downcast_ref::<Int64Array>()?.value(row))),
        DataType::UInt32 => Some(Scalar::UInt64(u64::from(array.as_any().downcast_ref::<UInt32Array>()?.value(row)))),
        DataType::UInt64 => Some(Scalar::UInt64(array.as_any().downcast_ref::<UInt64Array>()?.value(row))),
        DataType::Float32 => {
            Some(Scalar::Float64(f64::from(array.as_any().downcast_ref::<Float32Array>()?.value(row))))
        },
        DataType::Float64 => Some(Scalar::Float64(array.as_any().downcast_ref::<Float64Array>()?.value(row))),
        DataType::Utf8 => Some(Scalar::String(array.as_any().downcast_ref::<StringArray>()?.value(row).to_string())),
        DataType::LargeUtf8 => {
            Some(Scalar::String(array.as_any().downcast_ref::<LargeStringArray>()?.value(row).to_string()))
        },
        DataType::Utf8View => {
            Some(Scalar::String(array.as_any().downcast_ref::<StringViewArray>()?.value(row).to_string()))
        },
        _ => None,
    }
}

pub(super) fn array_i64_value(array: &dyn Array, row: usize) -> Option<Option<i64>> {
    if array.is_null(row) {
        return Some(None);
    }
    match array.data_type() {
        DataType::Int32 => Some(Some(i64::from(array.as_any().downcast_ref::<Int32Array>()?.value(row)))),
        DataType::Date32 => Some(Some(i64::from(array.as_any().downcast_ref::<Date32Array>()?.value(row)))),
        DataType::Int64 => Some(Some(array.as_any().downcast_ref::<Int64Array>()?.value(row))),
        DataType::UInt32 => Some(Some(i64::from(array.as_any().downcast_ref::<UInt32Array>()?.value(row)))),
        DataType::UInt64 => i64::try_from(array.as_any().downcast_ref::<UInt64Array>()?.value(row)).ok().map(Some),
        _ => None,
    }
}

fn literal_scalar(value: &LiteralValue) -> Option<Scalar> {
    match value {
        LiteralValue::Boolean(value) => Some(Scalar::Boolean(*value)),
        LiteralValue::Int8(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Int16(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Int32(value) | LiteralValue::Date32(value) => Some(Scalar::Int64(i64::from(*value))),
        LiteralValue::Date64(value) => Some(Scalar::Int64(*value)),
        LiteralValue::Int64(value) => Some(Scalar::Int64(*value)),
        LiteralValue::UInt8(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt16(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt32(value) => Some(Scalar::UInt64(u64::from(*value))),
        LiteralValue::UInt64(value) => Some(Scalar::UInt64(*value)),
        LiteralValue::Float32(value) => Some(Scalar::Float64(f64::from(*value))),
        LiteralValue::Float64(value) => Some(Scalar::Float64(*value)),
        LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value) => Some(Scalar::String(value.clone())),
        LiteralValue::Null(_) => Some(Scalar::Null),
        LiteralValue::Decimal128 { .. }
        | LiteralValue::Time32 { .. }
        | LiteralValue::Time64 { .. }
        | LiteralValue::Timestamp { .. }
        | LiteralValue::Duration { .. }
        | LiteralValue::IntervalMonthDayNano { .. } => None,
    }
}

fn cast_scalar(value: Scalar, target_type: &DataType) -> Option<Scalar> {
    if matches!(value, Scalar::Null) {
        return Some(Scalar::Null);
    }
    match target_type {
        DataType::Int32 | DataType::Int64 | DataType::Date32 => scalar_to_i64(value).map(Scalar::Int64),
        DataType::UInt32 | DataType::UInt64 => scalar_to_u64(value).map(Scalar::UInt64),
        DataType::Float32 | DataType::Float64 => scalar_to_f64(value).map(Scalar::Float64),
        DataType::Boolean => match value {
            Scalar::Boolean(value) => Some(Scalar::Boolean(value)),
            _ => None,
        },
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => match value {
            Scalar::String(value) => Some(Scalar::String(value)),
            _ => None,
        },
        _ => None,
    }
}

fn eval_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Option<Scalar> {
    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        return eval_boolean_binary(left, op, right);
    }
    if matches!(left, Scalar::Null) || matches!(right, Scalar::Null) {
        return Some(Scalar::Null);
    }
    match op {
        BinaryOp::Eq => scalar_eq(&left, &right).map(Scalar::Boolean),
        BinaryOp::NotEq => scalar_eq(&left, &right).map(|value| Scalar::Boolean(!value)),
        BinaryOp::Lt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering < 0)),
        BinaryOp::LtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering <= 0)),
        BinaryOp::Gt => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering > 0)),
        BinaryOp::GtEq => scalar_cmp(&left, &right).map(|ordering| Scalar::Boolean(ordering >= 0)),
        BinaryOp::Add => eval_checked_i64_binary(left, right, i64::checked_add).map(Scalar::Int64),
        BinaryOp::Subtract => eval_checked_i64_binary(left, right, i64::checked_sub).map(Scalar::Int64),
        BinaryOp::Multiply => eval_checked_i64_binary(left, right, i64::checked_mul).map(Scalar::Int64),
        BinaryOp::Divide
        | BinaryOp::Modulo
        | BinaryOp::BitwiseAnd
        | BinaryOp::BitwiseOr
        | BinaryOp::BitwiseXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight
        | BinaryOp::And
        | BinaryOp::Or => None,
    }
}

fn eval_checked_i64_binary(left: Scalar, right: Scalar, op: fn(i64, i64) -> Option<i64>) -> Option<i64> {
    op(scalar_to_i64(left)?, scalar_to_i64(right)?)
}

fn eval_boolean_binary(left: Scalar, op: BinaryOp, right: Scalar) -> Option<Scalar> {
    let left = match left {
        Scalar::Boolean(value) => value,
        Scalar::Null => false,
        _ => return None,
    };
    let right = match right {
        Scalar::Boolean(value) => value,
        Scalar::Null => false,
        _ => return None,
    };
    match op {
        BinaryOp::And => Some(Scalar::Boolean(left && right)),
        BinaryOp::Or => Some(Scalar::Boolean(left || right)),
        _ => None,
    }
}

fn scalar_eq(left: &Scalar, right: &Scalar) -> Option<bool> {
    if matches!((left, right), (Scalar::Null, _) | (_, Scalar::Null)) {
        return Some(false);
    }
    if let (Scalar::String(left), Scalar::String(right)) = (left, right) {
        return Some(left == right);
    }
    Some(scalar_cmp(left, right)? == 0)
}

fn scalar_cmp(left: &Scalar, right: &Scalar) -> Option<i8> {
    if let (Some(left), Some(right)) = (scalar_to_i64(left.clone()), scalar_to_i64(right.clone())) {
        return Some(left.cmp(&right) as i8);
    }
    let left = scalar_to_f64(left.clone())?;
    let right = scalar_to_f64(right.clone())?;
    left.partial_cmp(&right).map(|ordering| match ordering {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    })
}

fn scalar_to_i64(value: Scalar) -> Option<i64> {
    match value {
        Scalar::Int64(value) => Some(value),
        Scalar::UInt64(value) => i64::try_from(value).ok(),
        Scalar::Float64(value) => exact_i64_from_f64(value),
        Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => None,
    }
}

fn exact_i64_from_f64(value: f64) -> Option<i64> {
    if !value.is_finite() || value.fract() != 0.0 || value < i64::MIN as f64 || value >= 9_223_372_036_854_775_808.0 {
        return None;
    }
    let converted = value as i64;
    ((converted as f64) == value).then_some(converted)
}

fn scalar_to_u64(value: Scalar) -> Option<u64> {
    match value {
        Scalar::Int64(value) => u64::try_from(value).ok(),
        Scalar::UInt64(value) => Some(value),
        Scalar::Float64(value) if value.fract() == 0.0 && value >= 0.0 => Some(value as u64),
        Scalar::Boolean(_) | Scalar::Float64(_) | Scalar::String(_) | Scalar::Null => None,
    }
}

fn scalar_to_f64(value: Scalar) -> Option<f64> {
    match value {
        Scalar::Int64(value) => Some(value as f64),
        Scalar::UInt64(value) => Some(value as f64),
        Scalar::Float64(value) => Some(value),
        Scalar::Boolean(_) | Scalar::String(_) | Scalar::Null => None,
    }
}

// Tokenized LIKE patterns let escaped literals and collapsed '%' runs share one matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LikeToken {
    Literal(char),
    AnyOne,
    AnyMany,
}

fn like_matches(value: &str, pattern: &str, escape: Option<&str>) -> Option<bool> {
    let escape = match escape {
        Some(escape) => {
            let mut chars = escape.chars();
            let escape = chars.next()?;
            if chars.next().is_some() {
                return None;
            }
            Some(escape)
        },
        None => None,
    };
    let tokens = like_tokens(pattern, escape)?;
    let value = value.chars().collect::<Vec<_>>();
    let mut current = vec![false; value.len() + 1];
    current[0] = true;

    for token in tokens {
        let mut next = vec![false; value.len() + 1];
        match token {
            LikeToken::Literal(expected) => {
                for index in 0..value.len() {
                    if current[index] && value[index] == expected {
                        next[index + 1] = true;
                    }
                }
            },
            LikeToken::AnyOne => {
                for index in 0..value.len() {
                    if current[index] {
                        next[index + 1] = true;
                    }
                }
            },
            LikeToken::AnyMany => {
                let mut reachable = false;
                for index in 0..=value.len() {
                    reachable |= current[index];
                    if reachable {
                        next[index] = true;
                    }
                }
            },
        }
        current = next;
    }

    Some(current[value.len()])
}

fn like_tokens(pattern: &str, escape: Option<char>) -> Option<Vec<LikeToken>> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        if Some(ch) == escape {
            let escaped = chars.next()?;
            if !matches!(escaped, '%' | '_') && Some(escaped) != escape {
                return None;
            }
            tokens.push(LikeToken::Literal(escaped));
            continue;
        }
        match ch {
            '%' if !matches!(tokens.last(), Some(LikeToken::AnyMany)) => tokens.push(LikeToken::AnyMany),
            '%' => {},
            '_' => tokens.push(LikeToken::AnyOne),
            _ => tokens.push(LikeToken::Literal(ch)),
        }
    }
    Some(tokens)
}
