use std::time::Duration;

use super::{SourceReadFact, SourceReadTotals};

#[test]
fn every_counter_in_a_fact_advances_even_after_one_saturates() {
    let mut totals = SourceReadTotals { parquet_files: u64::MAX, ..SourceReadTotals::default() };

    let exact = totals.apply(&SourceReadFact::ParquetSource {
        files: 1,
        known_bytes: 10,
        selected_row_groups: 3,
        read_columns: 2,
    });

    assert!(!exact, "a saturated counter must make the fold report inexact totals");
    assert_eq!(totals.parquet_files, u64::MAX);
    assert_eq!(totals.parquet_bytes_known, 10);
    assert_eq!(totals.row_groups_selected, 3);
    assert_eq!(totals.read_columns, 2);
}

#[test]
fn durations_saturate_instead_of_wrapping() {
    let mut totals = SourceReadTotals { parquet_read_time: Duration::MAX, ..SourceReadTotals::default() };

    assert!(!totals.apply(&SourceReadFact::ParquetReadTime(Duration::from_secs(1))));
    assert_eq!(totals.parquet_read_time, Duration::MAX);
}
