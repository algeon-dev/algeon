//! Tests for metadata-only Debug formatting on DataFrame values.

use crate::common::{column_from_slice, column_from_strs};

#[test]
fn test_dataframe_debug_reports_table_metadata_without_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "city".to_string()],
        vec![column_from_slice(&[7i32, 8, 9]).unwrap(), column_from_strs(&["tokyo", "paris", "taipei"]).unwrap()],
    )
    .unwrap();

    let output = format!("{:?}", df);

    assert!(output.contains("DataFrame"), "debug output should contain the high-level type name: {}", output);
    assert!(output.contains("num_columns"), "debug output should contain column count metadata: {}", output);
    assert!(output.contains("num_rows"), "debug output should contain row count metadata: {}", output);
    assert!(output.contains("columns"), "debug output should contain schema metadata: {}", output);
    assert!(!output.contains("tokyo"), "debug output should not materialize payload values: {}", output);
    assert!(!output.contains("paris"), "debug output should not materialize payload values: {}", output);
}
