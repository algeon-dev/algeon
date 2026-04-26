use nexus_query_engine::harness::StableFnv1a64;

use super::*;

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

pub(super) fn fingerprint_batches(
    schema: &arrow::datatypes::SchemaRef,
    batches: &[RecordBatch],
) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-tpch-output-v3");
    for batch in batches {
        if batch.schema().as_ref() != schema.as_ref() {
            return Err(DataFusionError::Plan(
                "TPC-H output fingerprint requires collected batches to match the planned output schema".to_string(),
            ));
        }
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
    hasher.update_field(&batches.iter().map(RecordBatch::num_rows).sum::<usize>().to_string());
    hasher.update_field(&schema.fields().len().to_string());
    for batch in batches {
        for row in 0..batch.num_rows() {
            for column in 0..batch.num_columns() {
                let array = batch.column(column);
                if array.is_null(row) {
                    hasher.update_field("null");
                } else {
                    hasher.update_field("value");
                    hasher.update_field(&array_value_to_string(array.as_ref(), row)?);
                }
            }
        }
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}
