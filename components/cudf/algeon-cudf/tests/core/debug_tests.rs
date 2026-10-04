//! Smoke tests for metadata-only Debug formatting on core types.

use cudf::core::Table;

use crate::common::{column_from_slice, column_from_slice_nullable, column_from_strs, scalar};

#[test]
fn test_column_debug_smoke_report_metadata_without_values() {
    let column = column_from_slice_nullable(&[101i32, 0, 303], &[true, false, true]).unwrap();

    let column_output = format!("{:?}", column);

    assert!(column_output.contains("Column"), "debug output should identify Column: {}", column_output);
    assert!(
        column_output.contains("Int32") || column_output.contains("int32") || column_output.contains("INT32"),
        "debug output should contain Int32 metadata: {}",
        column_output
    );
    assert!(column_output.contains("len"), "debug output should contain length metadata: {}", column_output);
    assert!(column_output.contains("nullable"), "debug output should contain nullability metadata: {}", column_output);
    assert!(
        column_output.contains("execution_domain"),
        "debug output should contain execution-domain metadata: {}",
        column_output
    );
    assert!(
        !column_output.contains("[Some(101), None, Some(303)]"),
        "debug output should not materialize optional host vectors: {}",
        column_output
    );
    assert!(
        !column_output.contains("[101, 0, 303]"),
        "debug output should not materialize plain host vectors: {}",
        column_output
    );
}

#[test]
fn test_table_debug_smoke_report_schema_without_values() {
    let table = Table::from_columns(vec![
        column_from_slice(&[11i32, 22, 33]).unwrap(),
        column_from_strs(&["alpha", "beta", "gamma"]).unwrap(),
    ])
    .unwrap();

    let table_output = format!("{:?}", table);

    assert!(table_output.contains("Table"), "debug output should identify Table: {}", table_output);
    assert!(table_output.contains("num_columns"), "debug output should contain column metadata: {}", table_output);
    assert!(table_output.contains("num_rows"), "debug output should contain row metadata: {}", table_output);
    assert!(table_output.contains("columns"), "debug output should contain schema metadata: {}", table_output);
    assert!(
        table_output.contains("execution_domain"),
        "debug output should contain execution-domain metadata: {}",
        table_output
    );
    assert!(!table_output.contains("alpha"), "debug output should not materialize table payloads: {}", table_output);
}

#[test]
fn test_scalar_debug_smoke_reports_type_and_validity_without_value() {
    let scalar = scalar("secret-debug-payload").unwrap();

    let output = format!("{:?}", scalar);

    assert!(output.contains("Scalar"), "debug output should identify Scalar: {}", output);
    assert!(
        output.contains("String") || output.contains("string") || output.contains("STRING"),
        "debug output should contain String metadata: {}",
        output
    );
    assert!(output.contains("valid"), "debug output should contain validity metadata: {}", output);
    assert!(output.contains("execution_domain"), "debug output should contain execution-domain metadata: {}", output);
    assert!(
        !output.contains("secret-debug-payload"),
        "debug output should not materialize scalar payloads: {}",
        output
    );
}
