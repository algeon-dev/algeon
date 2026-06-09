#[derive(Debug, Clone, Copy)]
struct SharedScalarFilteredMaterializationEstimate {
    input_rows: u64,
    estimated_materialization_bytes: u64,
}

fn estimate_shared_scalar_filtered_materialization(
    family: &[AggregateMaterializationSignature],
) -> Option<SharedScalarFilteredMaterializationEstimate> {
    let representative = family.first()?;
    if !representative.group_exprs.is_empty() {
        return None;
    }
    let input_rows = family.iter().filter_map(|signature| signature.source_rows).max()?;
    let original_frame_retained_bytes = family.iter().filter_map(|signature| signature.source_bytes).max().unwrap_or(0);
    let unique_filter_predicate_count =
        family.iter().map(|signature| signature.exact_source_signature.as_str()).collect::<BTreeSet<_>>().len();
    let predicate_bytes =
        u64::try_from(unique_filter_predicate_count).ok()?.saturating_mul(input_rows).saturating_mul(2);
    let mut value_bytes = 0u64;
    for signature in family {
        for aggregate_expr in &signature.aggregate_expr_specs {
            let (width, nullable) = shared_aggregate_value_width_and_nullability(aggregate_expr, signature);
            let validity = if nullable { shared_null_mask_bytes(input_rows) } else { 0 };
            value_bytes = value_bytes
                .saturating_add(input_rows.saturating_mul(width))
                .saturating_add(validity)
                .saturating_add(64);
        }
    }
    Some(SharedScalarFilteredMaterializationEstimate {
        input_rows,
        estimated_materialization_bytes: original_frame_retained_bytes
            .saturating_add(value_bytes)
            .saturating_add(predicate_bytes),
    })
}

fn shared_aggregate_value_width_and_nullability(
    aggregate_expr: &AggregateExpr,
    signature: &AggregateMaterializationSignature,
) -> (u64, bool) {
    match &aggregate_expr.expr {
        Expr::Column(column) => signature
            .source_columns
            .values()
            .find_map(|columns| columns.get(column.name()))
            .map(|column| {
                (shared_fixed_width_dtype_size(column.field.data_type()).unwrap_or(16), column.field.is_nullable())
            })
            .unwrap_or((16, true)),
        Expr::Literal(value) => (shared_literal_width(value), matches!(value, crate::expr::LiteralValue::Null(_))),
        _ => (16, true),
    }
}

fn shared_literal_width(value: &crate::expr::LiteralValue) -> u64 {
    match value {
        crate::expr::LiteralValue::Boolean(_) => 1,
        crate::expr::LiteralValue::Int8(_) | crate::expr::LiteralValue::UInt8(_) => 1,
        crate::expr::LiteralValue::Int16(_) | crate::expr::LiteralValue::UInt16(_) => 2,
        crate::expr::LiteralValue::Int32(_)
        | crate::expr::LiteralValue::UInt32(_)
        | crate::expr::LiteralValue::Float32(_)
        | crate::expr::LiteralValue::Date32(_) => 4,
        crate::expr::LiteralValue::Int64(_)
        | crate::expr::LiteralValue::UInt64(_)
        | crate::expr::LiteralValue::Float64(_)
        | crate::expr::LiteralValue::Timestamp { .. }
        | crate::expr::LiteralValue::Duration { .. } => 8,
        crate::expr::LiteralValue::Decimal128 { .. } | crate::expr::LiteralValue::IntervalMonthDayNano { .. } => 16,
        crate::expr::LiteralValue::Null(_)
        | crate::expr::LiteralValue::Utf8(_)
        | crate::expr::LiteralValue::LargeUtf8(_) => 16,
    }
}

fn shared_fixed_width_dtype_size(data_type: &DataType) -> Option<u64> {
    match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date32 | DataType::Time32(_) => Some(4),
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => Some(8),
        DataType::Decimal128(_, _) => Some(16),
        DataType::Decimal256(_, _) => Some(32),
        _ => None,
    }
}

fn shared_null_mask_bytes(rows: u64) -> u64 {
    if rows == 0 { 0 } else { rows.div_ceil(8).div_ceil(64) * 64 }
}
