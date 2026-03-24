#[derive(Debug, Clone, Copy)]
struct SharedScalarFilteredMaterializationEstimate {
    input_rows: u64,
    estimated_materialization_bytes: u64,
}

/// Internal sizing policy for shared scalar filtered-aggregate materialization estimates.
///
/// These estimates are optimizer diagnostics and admission evidence only. Keep
/// them internal until runtime budget reporting can expose a stable production
/// tuning model for shared filtered aggregate materialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SharedScalarFilteredMaterializationEstimatePolicy {
    shared_filter_predicate_bytes_per_row: u64,
    shared_value_slot_overhead_bytes: u64,
    shared_unknown_value_width_bytes: u64,
    shared_null_mask_bits_per_byte: u64,
    shared_null_mask_alignment_bytes: u64,
}

impl SharedScalarFilteredMaterializationEstimatePolicy {
    const DEFAULT_SHARED_FILTER_PREDICATE_BYTES_PER_ROW: u64 = 2;
    const DEFAULT_SHARED_VALUE_SLOT_OVERHEAD_BYTES: u64 = 64;
    const DEFAULT_SHARED_UNKNOWN_VALUE_WIDTH_BYTES: u64 = 16;
    const DEFAULT_SHARED_NULL_MASK_BITS_PER_BYTE: u64 = 8;
    const DEFAULT_SHARED_NULL_MASK_ALIGNMENT_BYTES: u64 = 64;

    fn filter_predicate_bytes(self, input_rows: u64, unique_filter_predicate_count: usize) -> Option<u64> {
        Some(
            u64::try_from(unique_filter_predicate_count)
                .ok()?
                .saturating_mul(input_rows)
                .saturating_mul(self.shared_filter_predicate_bytes_per_row),
        )
    }

    fn value_slot_overhead_bytes(self) -> u64 {
        self.shared_value_slot_overhead_bytes
    }

    fn unknown_value_width_bytes(self) -> u64 {
        self.shared_unknown_value_width_bytes
    }

    fn null_mask_bytes(self, rows: u64) -> u64 {
        if rows == 0 {
            0
        } else {
            rows.div_ceil(self.shared_null_mask_bits_per_byte)
                .div_ceil(self.shared_null_mask_alignment_bytes)
                * self.shared_null_mask_alignment_bytes
        }
    }
}

impl Default for SharedScalarFilteredMaterializationEstimatePolicy {
    fn default() -> Self {
        Self {
            shared_filter_predicate_bytes_per_row: Self::DEFAULT_SHARED_FILTER_PREDICATE_BYTES_PER_ROW,
            shared_value_slot_overhead_bytes: Self::DEFAULT_SHARED_VALUE_SLOT_OVERHEAD_BYTES,
            shared_unknown_value_width_bytes: Self::DEFAULT_SHARED_UNKNOWN_VALUE_WIDTH_BYTES,
            shared_null_mask_bits_per_byte: Self::DEFAULT_SHARED_NULL_MASK_BITS_PER_BYTE,
            shared_null_mask_alignment_bytes: Self::DEFAULT_SHARED_NULL_MASK_ALIGNMENT_BYTES,
        }
    }
}

fn estimate_shared_scalar_filtered_materialization(
    family: &[AggregateMaterializationSignature],
) -> Option<SharedScalarFilteredMaterializationEstimate> {
    let policy = SharedScalarFilteredMaterializationEstimatePolicy::default();
    let representative = family.first()?;
    if !representative.group_exprs.is_empty() {
        return None;
    }
    let input_rows = family.iter().filter_map(|signature| signature.source_rows).max()?;
    let original_frame_retained_bytes = family.iter().filter_map(|signature| signature.source_bytes).max().unwrap_or(0);
    let unique_filter_predicate_count =
        family.iter().map(|signature| signature.exact_source_signature.as_str()).collect::<BTreeSet<_>>().len();
    let predicate_bytes = policy.filter_predicate_bytes(input_rows, unique_filter_predicate_count)?;
    let mut value_bytes = 0u64;
    for signature in family {
        for aggregate_expr in &signature.aggregate_expr_specs {
            let (width, nullable) = shared_aggregate_value_width_and_nullability(policy, aggregate_expr, signature);
            let validity = if nullable { policy.null_mask_bytes(input_rows) } else { 0 };
            value_bytes = value_bytes
                .saturating_add(input_rows.saturating_mul(width))
                .saturating_add(validity)
                .saturating_add(policy.value_slot_overhead_bytes());
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
    policy: SharedScalarFilteredMaterializationEstimatePolicy,
    aggregate_expr: &AggregateExpr,
    signature: &AggregateMaterializationSignature,
) -> (u64, bool) {
    match &aggregate_expr.expr {
        Expr::Column(column) => signature
            .source_columns
            .values()
            .find_map(|columns| columns.get(column.name()))
            .map(|column| {
                (
                    shared_fixed_width_dtype_size(column.field.data_type())
                        .unwrap_or(policy.unknown_value_width_bytes()),
                    column.field.is_nullable(),
                )
            })
            .unwrap_or((policy.unknown_value_width_bytes(), true)),
        Expr::Literal(value) => (shared_literal_width(value), matches!(value, crate::expr::LiteralValue::Null(_))),
        _ => (policy.unknown_value_width_bytes(), true),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_scalar_filtered_materialization_estimate_policy_preserves_predicate_bytes() {
        let policy = SharedScalarFilteredMaterializationEstimatePolicy::default();

        assert_eq!(policy.filter_predicate_bytes(10, 3), Some(60));
    }

    #[test]
    fn shared_scalar_filtered_materialization_estimate_policy_preserves_null_mask_alignment() {
        let policy = SharedScalarFilteredMaterializationEstimatePolicy::default();

        assert_eq!(policy.null_mask_bytes(0), 0);
        assert_eq!(policy.null_mask_bytes(1), 64);
        assert_eq!(policy.null_mask_bytes(512), 64);
        assert_eq!(policy.null_mask_bytes(513), 128);
    }

    #[test]
    fn shared_scalar_filtered_materialization_estimate_policy_preserves_unknown_width_and_overhead() {
        let policy = SharedScalarFilteredMaterializationEstimatePolicy::default();

        assert_eq!(policy.unknown_value_width_bytes(), 16);
        assert_eq!(policy.value_slot_overhead_bytes(), 64);
    }
}
