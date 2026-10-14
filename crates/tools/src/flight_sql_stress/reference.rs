//! Reference results for `flight-sql-stress` result checking.
//!
//! A reference is one statement's result stored as an Arrow IPC file named
//! `<statement>.arrow`. Checking compares a later result against it with the
//! TPC-DS parity comparator, so the order and float rules match
//! `native_tpcds_parity_report`: rows compare in order only when the
//! statement's top-level `ORDER BY` names every output column, otherwise as a
//! sorted multiset, and float columns accept the configured absolute or
//! relative error.

#[cfg(any(test, feature = "flight-sql-stress-check"))]
use std::io::Cursor;
use std::path::{Path, PathBuf};
#[cfg(not(test))]
use std::{fs, io::Write as _};

#[cfg(feature = "flight-sql-stress-check")]
use algeon_datafusion::native_report::bench_support::{
    ParityFloatTolerance, ParityStatus, compare_record_batches, sql_top_level_order_by_covers_output_columns,
};
#[cfg(any(test, feature = "flight-sql-stress-check"))]
use arrow::ipc::reader::FileReader;
use arrow::{array::RecordBatch, datatypes::SchemaRef, ipc::writer::FileWriter};

#[cfg(feature = "flight-sql-stress-check")]
use super::CheckOutcome;
#[cfg(any(not(test), feature = "flight-sql-stress-check"))]
use super::sha256_bytes;

/// One successful result as the client decoded it.
#[derive(Debug, Clone)]
pub(super) struct ResultCapture {
    pub(super) schema: SchemaRef,
    pub(super) batches: Vec<RecordBatch>,
}

impl ResultCapture {
    pub(super) fn rows(&self) -> u64 {
        self.batches.iter().map(|batch| batch.num_rows() as u64).sum()
    }
}

pub(super) fn reference_path(dir: &Path, statement: &str) -> PathBuf {
    dir.join(format!("{statement}.arrow"))
}

pub(super) fn encode_reference(capture: &ResultCapture) -> Result<Vec<u8>, String> {
    let error = |_| "cannot encode the result as Arrow IPC".to_string();
    let mut writer = FileWriter::try_new(Vec::new(), &capture.schema).map_err(error)?;
    for batch in &capture.batches {
        writer.write(batch).map_err(error)?;
    }
    writer.finish().map_err(error)?;
    writer.into_inner().map_err(error)
}

#[cfg(any(test, feature = "flight-sql-stress-check"))]
pub(super) fn decode_reference(bytes: &[u8]) -> Result<ResultCapture, String> {
    let reader = FileReader::try_new(Cursor::new(bytes), None).map_err(|_| "reference is not an Arrow IPC file")?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(|_| "reference Arrow IPC batches cannot be decoded")?;
    Ok(ResultCapture { schema, batches })
}

/// Writes `capture` to a reference file that must not exist yet and returns
/// the file's SHA-256.
#[cfg(not(test))]
pub(super) fn write_reference(path: &Path, capture: &ResultCapture) -> Result<String, String> {
    let bytes = encode_reference(capture)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| format!("cannot create reference {}", path.display()))?;
    file.write_all(&bytes).map_err(|_| format!("cannot write reference {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

/// Absolute and relative float error accepted by a check.
#[cfg(feature = "flight-sql-stress-check")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FloatTolerance {
    pub(super) absolute: f64,
    pub(super) relative: f64,
}

/// A loaded reference and its comparison order.
#[cfg(feature = "flight-sql-stress-check")]
#[derive(Debug)]
pub(super) struct Reference {
    pub(super) sha256: String,
    pub(super) ordered: bool,
    pub(super) rows: u64,
    result: ResultCapture,
}

#[cfg(feature = "flight-sql-stress-check")]
impl Reference {
    /// Decodes a reference for the statement `sql`. A reference that does not
    /// match itself (for example, a type the comparator cannot canonicalize)
    /// is rejected, because every check against it would report a mismatch.
    pub(super) fn from_bytes(bytes: &[u8], sql: &str, tolerance: FloatTolerance) -> Result<Self, String> {
        let result = decode_reference(bytes)?;
        let ordered = sql_top_level_order_by_covers_output_columns(sql, &result.schema);
        let reference = Self { sha256: sha256_bytes(bytes), ordered, rows: result.rows(), result };
        if reference.check(&reference.result, tolerance) != CheckOutcome::Match {
            return Err("reference cannot be compared with the parity comparator".to_string());
        }
        Ok(reference)
    }

    pub(super) fn check(&self, capture: &ResultCapture, tolerance: FloatTolerance) -> CheckOutcome {
        let comparison = compare_record_batches(
            &self.result.schema,
            &self.result.batches,
            &capture.schema,
            &capture.batches,
            self.ordered,
            ParityFloatTolerance::new(tolerance.absolute, tolerance.relative),
        );
        if comparison.status == ParityStatus::Matched { CheckOutcome::Match } else { CheckOutcome::Mismatch }
    }
}
