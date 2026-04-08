//! Tests for Series Display and Debug formatting.

use cudf::{Series, TypeId};

#[test]
fn test_display_reports_series_metadata_without_values() {
    let series = Series::new("id", &[101i32, 202, 303]).unwrap();

    let output = format!("{}", series);

    assert!(output.contains("Series"), "output should identify a Series: {}", output);
    assert!(output.contains("name=id"), "output should contain the series name: {}", output);
    assert!(
        output.contains("dtype=Int32") || output.contains("dtype=int32") || output.contains("dtype=INT32"),
        "output should contain the dtype: {}",
        output
    );
    assert!(output.contains("len=3"), "output should contain the length: {}", output);
    assert!(!output.contains("101"), "output should not materialize integer values: {}", output);
    assert!(!output.contains("202"), "output should not materialize integer values: {}", output);
}

#[test]
fn test_display_reports_nullable_metadata() {
    let series = Series::new_nullable("score", &[10i32, 0, 30], &[true, false, true]).unwrap();

    let output = format!("{}", series);

    assert!(output.contains("nullable"), "output should indicate nullable metadata: {}", output);
    assert!(output.contains("nulls=1"), "output should indicate null count: {}", output);
}

#[test]
fn test_display_unnamed_series_uses_placeholder_name() {
    let series = Series::from_column(cudf::core::Column::empty(TypeId::Float64).unwrap()).unwrap();

    let output = format!("{}", series);

    assert!(output.contains("name=<unnamed>"), "output should identify unnamed series: {}", output);
    assert!(output.contains("len=0"), "output should contain zero length: {}", output);
}

#[test]
fn test_debug_reports_extended_metadata() {
    let series = Series::new_nullable("flag", &[true, false, true], &[true, false, true]).unwrap();

    let output = format!("{:?}", series);

    assert!(output.contains("Series"), "debug output should contain 'Series': {}", output);
    assert!(output.contains("flag"), "debug output should contain the series name: {}", output);
    assert!(output.contains("nullable"), "debug output should contain nullable metadata: {}", output);
    assert!(output.contains("null_count"), "debug output should contain null count metadata: {}", output);
    assert!(!output.contains("true, false"), "debug output should not contain materialized values: {}", output);
}
