use std::fmt::{Display, Formatter};

use super::{WorkloadError, WorkloadResult, fixture::TINY_SUPPLIER_ROWS};

/// Supplier rows per unit of scale factor (TPC-H specification clause 4.2.5).
const SUPPLIERS_PER_SCALE_FACTOR: u64 = 10_000;
const FRACTIONAL_DIGITS: usize = 4;

/// Scale factor of a TPC-H dataset.
///
/// The value is held as the supplier cardinality, `SF * 10,000`, so two
/// scales compare exactly and a dataset's scale is read from its supplier row
/// count. Display and [`TpchScaleFactor::from_label`] use the `sf<decimal>`
/// label, for example `sf100` or `sf0.01`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TpchScaleFactor {
    supplier_rows: u64,
}

impl TpchScaleFactor {
    /// Scale of a dataset with `supplier_rows` suppliers. A dataset without
    /// suppliers has no scale.
    pub fn from_supplier_rows(supplier_rows: u64) -> WorkloadResult<Self> {
        if supplier_rows == 0 {
            return Err(WorkloadError::InvalidDataset(
                "TPC-H supplier table is empty, so the dataset has no scale factor".to_string(),
            ));
        }
        Ok(Self { supplier_rows })
    }

    /// Parses a declared scale-factor label.
    ///
    /// `sf<decimal>` with at most four fractional digits declares that scale.
    /// `tiny` declares the scale of
    /// [`write_tiny_tpch_parquet_dataset`](super::write_tiny_tpch_parquet_dataset).
    /// Any other label, such as `unknown`, declares no scale and returns
    /// `None`. A malformed `sf` label is an error.
    pub fn from_label(label: &str) -> WorkloadResult<Option<Self>> {
        if label == "tiny" {
            return Self::from_supplier_rows(TINY_SUPPLIER_ROWS).map(Some);
        }
        let Some(value) = label.strip_prefix("sf").or_else(|| label.strip_prefix("SF")) else {
            return Ok(None);
        };
        let invalid = || {
            WorkloadError::InvalidDataset(format!(
                "invalid TPC-H scale-factor label {label}; expected sf<decimal> with at most {FRACTIONAL_DIGITS} fractional digits"
            ))
        };
        let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
        let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
        if whole.is_empty() || !digits(whole) || !digits(fraction) || fraction.len() > FRACTIONAL_DIGITS {
            return Err(invalid());
        }
        let whole = whole.parse::<u64>().map_err(|_| invalid())?;
        let fraction = format!("{fraction:0<FRACTIONAL_DIGITS$}").parse::<u64>().map_err(|_| invalid())?;
        let supplier_rows = whole
            .checked_mul(SUPPLIERS_PER_SCALE_FACTOR)
            .and_then(|rows| rows.checked_add(fraction))
            .ok_or_else(invalid)?;
        Self::from_supplier_rows(supplier_rows).map(Some)
    }

    /// Returns this scale when `label` declares no scale or declares the same
    /// one, and an error when the declaration disagrees.
    pub fn check_declared(self, label: &str) -> WorkloadResult<Self> {
        match Self::from_label(label)? {
            Some(declared) if declared != self => Err(WorkloadError::InvalidDataset(format!(
                "declared TPC-H scale factor {label} disagrees with the data: {} supplier rows is {self}",
                self.supplier_rows
            ))),
            _ => Ok(self),
        }
    }

    /// Q11 `FRACTION` (specification clause 2.4.11.3) as qgen substitutes it.
    pub(super) fn q11_fraction(self) -> String {
        // dbgen/varsub.c case 11 prints Q11_FRACTION (0.0001) / flt_scale with "%11.10f".
        format!("{:.10}", 0.0001 / (self.supplier_rows as f64 / SUPPLIERS_PER_SCALE_FACTOR as f64))
    }
}

impl Display for TpchScaleFactor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let whole = self.supplier_rows / SUPPLIERS_PER_SCALE_FACTOR;
        let fraction = self.supplier_rows % SUPPLIERS_PER_SCALE_FACTOR;
        if fraction == 0 {
            return write!(f, "sf{whole}");
        }
        let fraction = format!("{fraction:0FRACTIONAL_DIGITS$}");
        write!(f, "sf{whole}.{}", fraction.trim_end_matches('0'))
    }
}
