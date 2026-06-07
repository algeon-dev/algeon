//! Tests for DataFrame Display and Debug formatting.

use cudf::{DataFrame, TypeId, core::Column};

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

#[test]
fn test_display_reports_schema_metadata_without_values() {
    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![
            Column::from_slice(&[101i32, 202, 303]).unwrap(),
            Column::from_slice(&[11.125f64, 22.25, 33.375]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);

    assert!(output.contains("DataFrame"), "output should identify a DataFrame: {}", output);
    assert!(output.contains("2 columns"), "output should contain column count: {}", output);
    assert!(output.contains("3 rows"), "output should contain row count: {}", output);
    assert!(output.contains("id"), "output should contain column name 'id': {}", output);
    assert!(output.contains("score"), "output should contain column name 'score': {}", output);
    assert!(
        output.contains("Int32") || output.contains("int32") || output.contains("INT32"),
        "output should contain Int32 type: {}",
        output
    );
    assert!(
        output.contains("Float64") || output.contains("float64") || output.contains("FLOAT64"),
        "output should contain Float64 type: {}",
        output
    );
    assert!(!output.contains("101"), "output should not materialize integer values: {}", output);
    assert!(!output.contains("11.125"), "output should not materialize float values: {}", output);
}

#[test]
fn test_display_empty_dataframe() {
    let df = DataFrame::new(vec![], vec![]).unwrap();
    let output = format!("{}", df);
    assert_eq!(output, "DataFrame [0 columns x 0 rows]");
}

#[test]
fn test_display_zero_rows() {
    let df = DataFrame::new(
        vec!["a".to_string(), "b".to_string()],
        vec![Column::empty(TypeId::Int32).unwrap(), Column::empty(TypeId::Float64).unwrap()],
    )
    .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("DataFrame"), "output should identify a DataFrame: {}", output);
    assert!(output.contains("2 columns"), "output should contain column count: {}", output);
    assert!(output.contains("0 rows"), "output should contain zero row count: {}", output);
    assert!(output.contains("a"), "output should contain column name 'a': {}", output);
    assert!(output.contains("b"), "output should contain column name 'b': {}", output);
    assert!(output.contains("Int32") || output.contains("int32"), "output should contain Int32: {}", output);
    assert!(output.contains("Float64") || output.contains("float64"), "output should contain Float64: {}", output);
}

#[test]
fn test_display_large_dataframe_still_omits_values() {
    let n = 30;
    let ids: Vec<i32> = (1000..1000 + n).collect();
    let vals: Vec<f64> = (0..n).map(|i| 5000.0 + i as f64 * 1.25).collect();
    let df = DataFrame::new(
        vec!["id".to_string(), "val".to_string()],
        vec![Column::from_slice(&ids).unwrap(), Column::from_slice(&vals).unwrap()],
    )
    .unwrap();

    let output = format!("{}", df);

    assert!(output.contains("30 rows"), "output should mention 30 rows: {}", output);
    assert!(!output.contains("1000"), "output should not contain row values: {}", output);
    assert!(!output.contains("5000"), "output should not contain row values: {}", output);
}

#[test]
fn test_display_nullable_column() {
    let df = DataFrame::new(
        vec!["a".to_string()],
        vec![Column::from_slice_nullable(&[1i32, 0, 3], &[true, false, true]).unwrap()],
    )
    .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("nullable"), "output should indicate nullable metadata: {}", output);
    assert!(output.contains("nulls=1"), "output should indicate null count metadata: {}", output);
}

#[test]
fn test_display_bool_column() {
    let df = DataFrame::new(vec!["flag".to_string()], vec![Column::from_slice(&[true, false, true]).unwrap()]).unwrap();

    let output = format!("{}", df);
    assert!(output.contains("flag"), "output should contain column name: {}", output);
    assert!(output.contains("Bool8") || output.contains("bool8"), "output should contain Bool8 type: {}", output);
    assert!(!output.contains("true"), "output should not contain boolean values: {}", output);
    assert!(!output.contains("false"), "output should not contain boolean values: {}", output);
}

#[test]
fn test_display_string_column_reports_dtype_without_materializing_values() {
    let df =
        DataFrame::new(vec!["city".to_string()], vec![Column::from_strings(&["Tokyo", "Paris", "Taipei"]).unwrap()])
            .unwrap();

    let output = format!("{}", df);
    assert!(output.contains("city"), "output should contain column name: {}", output);
    assert!(
        output.contains("String") || output.contains("STRING") || output.contains("string"),
        "output should contain String type: {}",
        output
    );
    assert!(!output.contains("Tokyo"), "output should not contain string values: {}", output);
    assert!(!output.contains("Paris"), "output should not contain string values: {}", output);
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

#[test]
fn test_debug_produces_output() {
    let df = DataFrame::new(
        vec!["id".to_string(), "score".to_string()],
        vec![Column::from_slice(&[1i32, 2, 3]).unwrap(), Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap()],
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
        vec![Column::from_slice(&[1i32]).unwrap(), Column::from_slice(&[1.0f64]).unwrap()],
    )
    .unwrap();

    let output = format!("{:?}", df);

    // Debug should contain type information.
    assert!(output.contains("Int32") || output.contains("int32"), "debug output should contain Int32: {}", output);
    assert!(
        output.contains("Float64") || output.contains("float64"),
        "debug output should contain Float64: {}",
        output
    );
}

#[test]
fn test_debug_contains_row_count() {
    let df = DataFrame::new(vec!["a".to_string()], vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()]).unwrap();

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
