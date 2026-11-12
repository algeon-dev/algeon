use arrow::{array::Array, datatypes::SchemaRef};
use datafusion::logical_expr::{Expr, LogicalPlan};
use query_planning::util::StableFnv1a64;

use super::{DataFusionError, DataFusionResult, RecordBatch, array_value_to_string, execution};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParityStatus {
    NotRun,
    Matched,
    Mismatched,
    CpuFailed,
    NativeFailed,
}

impl ParityStatus {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::NotRun => "not_run",
            Self::Matched => "matched",
            Self::Mismatched => "mismatched",
            Self::CpuFailed => "cpu_failed",
            Self::NativeFailed => "native_failed",
        }
    }
}

pub(super) fn parity_status(
    cpu: Option<&execution::QueryMeasurement>,
    native: Option<&execution::QueryMeasurement>,
) -> ParityStatus {
    match (cpu, native) {
        (Some(cpu), Some(native)) if cpu.output_fingerprint == native.output_fingerprint => ParityStatus::Matched,
        (Some(_), Some(_)) => ParityStatus::Mismatched,
        (None, Some(_)) => ParityStatus::CpuFailed,
        (Some(_), None) => ParityStatus::NativeFailed,
        (None, None) => ParityStatus::NotRun,
    }
}

/// Output columns whose value sequence the query's top-level ORDER BY fixes,
/// in ORDER BY order, read from the SQL planner's unoptimized logical plan.
///
/// An empty list means the query has no top-level ORDER BY. When a key is not
/// an output column (an expression, or a hidden key the planner projects away
/// above the Sort), every output column is returned, so the whole row sequence
/// must match.
pub(super) fn order_key_columns(plan: &LogicalPlan) -> Vec<usize> {
    let all_columns = || (0..plan.schema().fields().len()).collect();
    let mut node = plan;
    let mut projected = false;
    loop {
        match node {
            LogicalPlan::Limit(limit) => node = limit.input.as_ref(),
            LogicalPlan::Projection(projection) => {
                projected = true;
                node = projection.input.as_ref();
            },
            LogicalPlan::Sort(sort) if !projected => {
                return sort
                    .expr
                    .iter()
                    .map(|sort_expr| match &sort_expr.expr {
                        Expr::Column(column) => sort.input.schema().index_of_column(column).ok(),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(all_columns);
            },
            LogicalPlan::Sort(_) => return all_columns(),
            _ => return Vec::new(),
        }
    }
}

/// Fingerprints query output under ORDER BY semantics. Two outputs get the same
/// fingerprint when they have the same schema, the same sequence of
/// `order_key_columns` values and the same multiset of rows, so rows may move
/// only within runs of equal ORDER BY key values, and an output without ORDER BY
/// keys compares as a multiset. Record batch boundaries do not affect the
/// result. A tie run cut by a LIMIT can legitimately keep different rows in two
/// engines; those outputs still get different fingerprints.
pub(super) fn fingerprint_batches(
    schema: &SchemaRef,
    order_key_columns: &[usize],
    batches: &[RecordBatch],
) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("algeon-datafusion-tpch-output-v4");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(DataFusionError::Plan(
                "TPC-H output fingerprint requires collected batches to match the planned output schema".to_string(),
            ));
        }
    }
    if let Some(column) = order_key_columns.iter().find(|&&column| column >= schema.fields().len()) {
        return Err(DataFusionError::Plan(format!(
            "TPC-H output fingerprint ORDER BY key column {column} is outside the {}-column output schema",
            schema.fields().len()
        )));
    }
    hasher.update_field("schema_metadata");
    hasher.update_metadata(schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&field.data_type().to_string());
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        hasher.update_metadata(field.metadata());
    }
    let row_count = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    hasher.update_field(&row_count.to_string());
    hasher.update_field(&schema.fields().len().to_string());
    hasher.update_field("order_key_columns");
    hasher.update_field(&order_key_columns.len().to_string());
    for column in order_key_columns {
        hasher.update_field(&column.to_string());
    }
    hasher.update_field("order_key_sequence");
    let mut row_hashes = Vec::with_capacity(row_count);
    for batch in batches {
        for row in 0..batch.num_rows() {
            for &column in order_key_columns {
                hash_value(&mut hasher, batch.column(column).as_ref(), row)?;
            }
            let mut row_hasher = StableFnv1a64::new();
            for array in batch.columns() {
                hash_value(&mut row_hasher, array.as_ref(), row)?;
            }
            row_hashes.push(row_hasher.finish());
        }
    }
    // Sorted per-row hashes make the row multiset independent of output order.
    row_hashes.sort_unstable();
    hasher.update_field("row_multiset");
    for row_hash in row_hashes {
        hasher.update_bytes(&row_hash.to_le_bytes());
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}

fn hash_value(hasher: &mut StableFnv1a64, array: &dyn Array, row: usize) -> DataFusionResult<()> {
    if array.is_null(row) {
        hasher.update_field("null");
    } else {
        hasher.update_field("value");
        hasher.update_field(&array_value_to_string(array, row)?);
    }
    Ok(())
}
