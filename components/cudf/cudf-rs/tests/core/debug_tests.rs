//! Tests for metadata-only Debug formatting on core types.

use cudf::core::{Column, Scalar, Table};

#[test]
fn test_column_debug_reports_metadata_without_values() {
    let column = Column::from_slice_nullable(&[101i32, 0, 303], &[true, false, true]).unwrap();

    let output = format!("{:?}", column);

    assert!(output.contains("Column"), "debug output should contain the type name: {}", output);
    assert!(
        output.contains("Int32") || output.contains("int32") || output.contains("INT32"),
        "debug output should contain the dtype: {}",
        output
    );
    assert!(output.contains("len"), "debug output should contain len metadata: {}", output);
    assert!(output.contains("nullable"), "debug output should contain nullable metadata: {}", output);
    assert!(output.contains("null_count"), "debug output should contain null count metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
    assert!(!output.contains("101"), "debug output should not materialize payload values: {}", output);
    assert!(!output.contains("303"), "debug output should not materialize payload values: {}", output);
}

#[test]
fn test_column_view_debug_reports_metadata_without_values() {
    let column = Column::from_strs(&["tokyo", "paris", "taipei"]).unwrap();
    let view = column.view().unwrap();

    let output = format!("{:?}", view);

    assert!(output.contains("ColumnView"), "debug output should contain the type name: {}", output);
    assert!(
        output.contains("String") || output.contains("string") || output.contains("STRING"),
        "debug output should contain the dtype: {}",
        output
    );
    assert!(output.contains("len"), "debug output should contain len metadata: {}", output);
    assert!(output.contains("nullable"), "debug output should contain nullable metadata: {}", output);
    assert!(output.contains("null_count"), "debug output should contain null count metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
    assert!(!output.contains("tokyo"), "debug output should not materialize payload values: {}", output);
    assert!(!output.contains("paris"), "debug output should not materialize payload values: {}", output);
}

#[test]
fn test_table_debug_reports_schema_without_values() {
    let table = Table::from_columns(vec![
        Column::from_slice(&[11i32, 22, 33]).unwrap(),
        Column::from_strs(&["alpha", "beta", "gamma"]).unwrap(),
    ])
    .unwrap();

    let output = format!("{:?}", table);

    assert!(output.contains("Table"), "debug output should contain the type name: {}", output);
    assert!(output.contains("num_columns"), "debug output should contain column count metadata: {}", output);
    assert!(output.contains("num_rows"), "debug output should contain row count metadata: {}", output);
    assert!(output.contains("columns"), "debug output should contain schema metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
    assert!(!output.contains("alpha"), "debug output should not materialize string payloads: {}", output);
    assert!(!output.contains("beta"), "debug output should not materialize string payloads: {}", output);
}

#[test]
fn test_table_view_debug_reports_schema_without_values() {
    let table = Table::from_columns(vec![
        Column::from_slice(&[1.25f64, 2.5, 3.75]).unwrap(),
        Column::from_slice_nullable(&[true, false, true], &[true, false, true]).unwrap(),
    ])
    .unwrap();
    let view = table.view().unwrap();

    let output = format!("{:?}", view);

    assert!(output.contains("TableView"), "debug output should contain the type name: {}", output);
    assert!(output.contains("num_columns"), "debug output should contain column count metadata: {}", output);
    assert!(output.contains("num_rows"), "debug output should contain row count metadata: {}", output);
    assert!(output.contains("columns"), "debug output should contain schema metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
}

#[test]
fn test_scalar_debug_reports_type_and_validity_without_value() {
    let scalar = Scalar::from_str("secret-debug-payload").unwrap();

    let output = format!("{:?}", scalar);

    assert!(output.contains("Scalar"), "debug output should contain the type name: {}", output);
    assert!(
        output.contains("String") || output.contains("string") || output.contains("STRING"),
        "debug output should contain the dtype: {}",
        output
    );
    assert!(output.contains("valid"), "debug output should contain validity metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
    assert!(
        !output.contains("secret-debug-payload"),
        "debug output should not materialize scalar payload values: {}",
        output
    );
}
