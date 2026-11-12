use algeon_datafusion::native_report::bench_support::{
    ParityFloatTolerance, ParityRowOrdering, ParitySortKey, ParityStatus as AdapterParityStatus,
    compare_record_batches_with_ordering,
};
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

/// Compares the last measured CPU and native outputs value by value under the
/// CPU query's ORDER BY. Float32 and Float64 values match within `tolerance`;
/// every other value must be equal. An output that cannot be compared counts
/// as mismatched.
pub(super) fn parity_status(
    cpu: Option<&execution::QueryMeasurement>,
    native: Option<&execution::QueryMeasurement>,
    tolerance: ParityFloatTolerance,
) -> ParityStatus {
    match (cpu, native) {
        (Some(cpu), Some(native)) => {
            let comparison = compare_record_batches_with_ordering(
                &cpu.output.schema,
                &cpu.output.batches,
                &native.output.schema,
                &native.output.batches,
                &cpu.output.row_ordering,
                tolerance,
            );
            if comparison.status == AdapterParityStatus::Matched {
                ParityStatus::Matched
            } else {
                ParityStatus::Mismatched
            }
        },
        (None, Some(_)) => ParityStatus::CpuFailed,
        (Some(_), None) => ParityStatus::NativeFailed,
        (None, None) => ParityStatus::NotRun,
    }
}

/// Row order the query's top-level ORDER BY fixes, read from the SQL planner's
/// unoptimized logical plan.
///
/// A query without a top-level ORDER BY is unordered. When a key is not an
/// output column (an expression, or a hidden key the planner projects away
/// above the Sort), the whole emitted row sequence must match.
pub(super) fn row_ordering(plan: &LogicalPlan) -> ParityRowOrdering {
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
                        Expr::Column(column) => sort
                            .input
                            .schema()
                            .index_of_column(column)
                            .ok()
                            .map(|column| ParitySortKey { column, descending: !sort_expr.asc }),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                    .map_or(ParityRowOrdering::Ordered, ParityRowOrdering::OrderBy);
            },
            LogicalPlan::Sort(_) => return ParityRowOrdering::Ordered,
            _ => return ParityRowOrdering::Unordered,
        }
    }
}

/// Fingerprints query output exactly under ORDER BY semantics. Two outputs get
/// the same fingerprint when they have the same schema, the same sequence of
/// ORDER BY key values and the same multiset of rows, so rows may move only
/// within runs of equal key values, and an unordered output compares as a
/// multiset. Record batch boundaries do not affect the result. Values hash by
/// their rendered text, so a last-ulp float difference changes the fingerprint
/// although [`parity_status`] matches it. A tie run cut by a LIMIT can
/// legitimately keep different rows in two engines; those outputs still get
/// different fingerprints.
pub(super) fn fingerprint_batches(
    schema: &SchemaRef,
    row_ordering: &ParityRowOrdering,
    batches: &[RecordBatch],
) -> DataFusionResult<String> {
    let order_key_columns = match row_ordering {
        ParityRowOrdering::Unordered => Vec::new(),
        ParityRowOrdering::Ordered => (0..schema.fields().len()).collect(),
        ParityRowOrdering::OrderBy(keys) => keys.iter().map(|key| key.column).collect(),
    };
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
    for column in &order_key_columns {
        hasher.update_field(&column.to_string());
    }
    hasher.update_field("order_key_sequence");
    let mut row_hashes = Vec::with_capacity(row_count);
    for batch in batches {
        for row in 0..batch.num_rows() {
            for &column in &order_key_columns {
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
