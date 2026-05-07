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

pub(super) fn fingerprint_batches(batches: &[RecordBatch]) -> DataFusionResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-cudf-tpch-output-v1");
    if let Some(batch) = batches.first() {
        for field in batch.schema().fields() {
            hasher.update_field(field.name());
            hasher.update_field(&field.data_type().to_string());
        }
    }
    for batch in batches {
        hasher.update_field(&batch.num_rows().to_string());
        hasher.update_field(&batch.num_columns().to_string());
        for row in 0..batch.num_rows() {
            for column in 0..batch.num_columns() {
                hasher.update_field(&array_value_to_string(batch.column(column).as_ref(), row)?);
            }
        }
    }
    Ok(format!("output-fnv1a64:{:016x}", hasher.finish()))
}

#[derive(Debug, Clone)]
struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    fn new() -> Self {
        Self { state: 0xcbf29ce484222325 }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0xff]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}
