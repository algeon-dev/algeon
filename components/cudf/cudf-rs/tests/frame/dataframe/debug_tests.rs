//! Tests for metadata-only Debug formatting on DataFrame low-level views.

use crate::common::{column_from_slice, column_from_strs};

#[test]
fn test_dataframe_view_debug_reports_table_metadata_without_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "city".to_string()],
        vec![column_from_slice(&[7i32, 8, 9]).unwrap(), column_from_strs(&["tokyo", "paris", "taipei"]).unwrap()],
    )
    .unwrap();

    let view = df.view().unwrap();
    let output = format!("{:?}", view);

    assert!(output.contains("TableView"), "debug output should contain the low-level view type name: {}", output);
    assert!(output.contains("num_columns"), "debug output should contain column count metadata: {}", output);
    assert!(output.contains("num_rows"), "debug output should contain row count metadata: {}", output);
    assert!(output.contains("columns"), "debug output should contain schema metadata: {}", output);
    assert!(!output.contains("tokyo"), "debug output should not materialize payload values: {}", output);
    assert!(!output.contains("paris"), "debug output should not materialize payload values: {}", output);
}
