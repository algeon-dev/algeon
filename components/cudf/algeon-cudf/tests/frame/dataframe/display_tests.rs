//! Smoke tests for DataFrame Display and Debug formatting.

use crate::common::{column_from_slice, column_from_slice_nullable, column_from_strs};

#[test]
fn test_display_smoke_reports_schema_metadata_without_materializing_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "city".to_string()],
        vec![
            column_from_slice_nullable(&[101i32, 0, 303], &[true, false, true]).unwrap(),
            column_from_strs(&["tokyo", "paris", "taipei"]).unwrap(),
        ],
    )
    .unwrap();

    let output = format!("{}", df);

    assert!(output.contains("DataFrame"), "output should identify a DataFrame: {}", output);
    assert!(output.contains("2 columns"), "output should contain column count: {}", output);
    assert!(output.contains("3 rows"), "output should contain row count: {}", output);
    assert!(output.contains("id"), "output should contain the id column name: {}", output);
    assert!(output.contains("city"), "output should contain the city column name: {}", output);
    assert!(
        output.contains("Int32") || output.contains("int32") || output.contains("INT32"),
        "output should contain Int32 schema metadata: {}",
        output
    );
    assert!(
        output.contains("String") || output.contains("string") || output.contains("STRING"),
        "output should contain String schema metadata: {}",
        output
    );
    assert!(output.contains("nullable"), "output should contain nullability metadata: {}", output);
    assert!(output.contains("nulls=1"), "output should contain null-count metadata: {}", output);
    assert!(!output.contains("101"), "output should not materialize integer payloads: {}", output);
    assert!(!output.contains("tokyo"), "output should not materialize string payloads: {}", output);
}

#[test]
fn test_debug_smoke_reports_schema_metadata_without_materializing_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "score".to_string()],
        vec![column_from_slice(&[7i32, 8, 9]).unwrap(), column_from_slice(&[11.25f64, 22.5, 33.75]).unwrap()],
    )
    .unwrap();

    let output = format!("{:?}", df);

    assert!(output.contains("DataFrame"), "debug output should identify a DataFrame: {}", output);
    assert!(output.contains("id"), "debug output should contain the id column name: {}", output);
    assert!(output.contains("score"), "debug output should contain the score column name: {}", output);
    assert!(
        output.contains("2") || output.contains("num_columns"),
        "debug output should contain column metadata: {}",
        output
    );
    assert!(
        output.contains("3") || output.contains("num_rows"),
        "debug output should contain row metadata: {}",
        output
    );
    assert!(
        output.contains("Int32") || output.contains("int32") || output.contains("INT32"),
        "debug output should contain Int32 schema metadata: {}",
        output
    );
    assert!(
        output.contains("Float64") || output.contains("float64") || output.contains("FLOAT64"),
        "debug output should contain Float64 schema metadata: {}",
        output
    );
    assert!(!output.contains("7"), "debug output should not materialize integer payloads: {}", output);
    assert!(!output.contains("11.25"), "debug output should not materialize float payloads: {}", output);
}
