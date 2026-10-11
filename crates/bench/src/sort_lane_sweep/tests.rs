use std::{slice, sync::Arc};

use arrow::array::{Int64Array, RecordBatch};

use super::{fingerprint_ordered, percentile, summarize, sweep_schema};

#[test]
fn ordered_fingerprint_preserves_rows_across_batch_boundaries() {
    let batch = output_batch(vec![Some(4), None, Some(2), Some(1)], vec![Some(40), Some(30), None, Some(10)]);
    let empty = RecordBatch::new_empty(sweep_schema());
    let reference = fingerprint_ordered(slice::from_ref(&batch)).unwrap();
    for (layout, batches) in [
        ("one then three", vec![batch.slice(0, 1), batch.slice(1, 3)]),
        ("two then two with empty", vec![batch.slice(0, 2), empty.clone(), batch.slice(2, 2)]),
        ("three then one with edge empties", vec![empty.clone(), batch.slice(0, 3), batch.slice(3, 1), empty]),
    ] {
        assert_eq!(reference, fingerprint_ordered(&batches).unwrap(), "layout={layout}");
    }

    let reordered = vec![batch.slice(1, 3), batch.slice(0, 1)];
    assert_ne!(reference, fingerprint_ordered(&reordered).unwrap(), "row order must affect the fingerprint");

    let changed = output_batch(vec![Some(4), None, Some(2), Some(1)], vec![Some(40), Some(30), None, Some(11)]);
    assert_ne!(reference, fingerprint_ordered(&[changed]).unwrap(), "cell values must affect the fingerprint");
}

fn output_batch(keys: Vec<Option<i64>>, values: Vec<Option<i64>>) -> RecordBatch {
    RecordBatch::try_new(sweep_schema(), vec![Arc::new(Int64Array::from(keys)), Arc::new(Int64Array::from(values))])
        .unwrap()
}

#[test]
fn summary_reports_median_p95_and_stddev() {
    let values: Vec<f64> = (1..=10).map(f64::from).collect();
    let (median, p95, stddev) = summarize(&values);
    assert_eq!(median, 6.0);
    assert_eq!(p95, 10.0);
    assert!((stddev - 2.8722813232690143).abs() < 1e-9, "stddev={stddev}");
    assert!(percentile(&[], 0.5).is_nan(), "empty samples stay NaN, never zero");
}
