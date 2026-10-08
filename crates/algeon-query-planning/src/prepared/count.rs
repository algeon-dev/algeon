//! COUNT as mergeable partial state.
//!
//! A merge sums partial counts, and a sum over no rows is null. A grouped merge
//! has at least one partial row per group. A global merge has none when its
//! input is empty, so its merged counts are null where COUNT is zero. Every
//! rewrite that merges counts declares the merged field nullable for a global
//! merge and restores zero with [`zero_filled_count_expr`] wherever the count
//! becomes an output.

use arrow::datatypes::Field;

use super::{ArrowDataType, Expr, QueryError, QueryErrorCode, Result};
use crate::expr::{CaseWhen, LiteralValue};

/// Returns the field of a merged count column, nullable for a global merge.
pub(crate) fn merged_count_field(field: Field, global_merge: bool) -> Field {
    if global_merge { field.with_nullable(true) } else { field }
}

/// Returns `count` with a null read as zero, for a count of the integer
/// `count_type`.
pub(crate) fn zero_filled_count_expr(count: Expr, count_type: &ArrowDataType) -> Result<Expr> {
    let zero = LiteralValue::integer_zero(count_type).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.aggregate.count_state_type",
            "a count state must have an integer type",
        )
        .with_fact("count_type", format!("{count_type:?}"))
    })?;
    Ok(Expr::case(vec![CaseWhen::new(Expr::is_null(count.clone()), Expr::literal(zero))], count))
}
