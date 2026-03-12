//! Tests for DataFrame Display and Debug formatting.

use cudf::column::Column;
use cudf::dataframe::DataFrame;

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn test_display_produces_readable_output() {
    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![
            Column::from_slice(&[1i32, 2, 3]).unwrap(),
            Column::from_slice(&[10.5f64, 20.0, 30.5]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);

    // Must contain column names.
    assert!(output.contains("id"), "output should contain column name 'id': {}", output);
    assert!(output.contains("score"), "output should contain column name 'score': {}", output);

    // Must contain row count info.
    assert!(output.contains("3"), "output should indicate 3 rows: {}", output);

    // Must contain type info.
    assert!(output.contains("Int32") || output.contains("int32") || output.contains("INT32"),
        "output should contain Int32 type: {}", output);
    assert!(output.contains("Float64") || output.contains("float64") || output.contains("FLOAT64"),
        "output should contain Float64 type: {}", output);

    // Must contain data values from first rows.
    assert!(output.contains("1"), "output should contain value 1: {}", output);
    assert!(output.contains("10.5"), "output should contain value 10.5: {}", output);
}

#[test]
fn test_display_empty_dataframe() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    let output = format!("{}", df);
    // Should not panic and should indicate emptiness.
    assert!(
        output.contains("0 column") || output.contains("empty") || output.contains("0 row"),
        "output should indicate empty DataFrame: {}",
        output
    );
}

#[test]
fn test_display_zero_rows() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            Column::empty(cudf::types::TypeId::Int32).unwrap(),
            Column::empty(cudf::types::TypeId::Float64).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("a"), "output should contain column name 'a': {}", output);
    assert!(output.contains("b"), "output should contain column name 'b': {}", output);
    assert!(output.contains("0"), "output should contain 0 rows: {}", output);
}

#[test]
fn test_display_shows_limited_rows() {
    // Create a DataFrame with more than the typical display limit.
    let n = 30;
    let ids: Vec<i32> = (0..n).collect();
    let vals: Vec<f64> = (0..n).map(|i| i as f64 * 1.1).collect();
    let df = DataFrame::new(
        vec!["id".to_string(), "val".to_string()],
        vec![
            Column::from_slice(&ids).unwrap(),
            Column::from_slice(&vals).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);

    // Should show row count.
    assert!(output.contains("30"), "output should mention 30 rows: {}", output);

    // Should contain the first row's data.
    assert!(output.contains("0"), "output should contain first row value: {}", output);
}

#[test]
fn test_display_nullable_column() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![
            Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);

    // Null values should be displayed somehow (e.g., "null", "NULL", "—", etc.)
    assert!(
        output.contains("null") || output.contains("NULL") || output.contains("N/A") || output.contains("—"),
        "output should indicate null value: {}",
        output
    );
}

#[test]
fn test_display_bool_column() {
    let df = DataFrame::new(
        vec!["flag".to_string()],
        vec![Column::from_slice(&[true, false, true]).unwrap()],
    )
    .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("flag"), "output should contain column name: {}", output);
    assert!(
        output.contains("true") || output.contains("True") || output.contains("TRUE"),
        "output should contain boolean value: {}",
        output
    );
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

#[test]
fn test_debug_produces_output() {
    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![
            Column::from_slice(&[1i32, 2, 3]).unwrap(),
            Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{:?}", df);

    // Must contain "DataFrame".
    assert!(output.contains("DataFrame"), "debug output should contain 'DataFrame': {}", output);

    // Must contain schema info.
    assert!(output.contains("id"), "debug output should contain column name 'id': {}", output);
    assert!(output.contains("score"), "debug output should contain column name 'score': {}", output);

    // Must contain column count.
    assert!(
        output.contains("2") || output.contains("num_columns"),
        "debug output should contain column count: {}",
        output
    );
}

#[test]
fn test_debug_contains_types() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![
            Column::from_slice(&[1i32]).unwrap(),
            Column::from_slice(&[1.0f64]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{:?}", df);

    // Debug should contain type information.
    assert!(
        output.contains("Int32") || output.contains("int32"),
        "debug output should contain Int32: {}",
        output
    );
    assert!(
        output.contains("Float64") || output.contains("float64"),
        "debug output should contain Float64: {}",
        output
    );
}

#[test]
fn test_debug_contains_row_count() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()],
    )
    .unwrap();

    let output = format!("{:?}", df);

    assert!(
        output.contains("5") || output.contains("num_rows"),
        "debug output should contain row count info: {}",
        output
    );
}

#[test]
fn test_debug_empty_dataframe() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    let output = format!("{:?}", df);
    assert!(output.contains("DataFrame"), "debug output should contain 'DataFrame': {}", output);
    assert!(output.contains("0"), "debug output should indicate 0 columns: {}", output);
}
