//! Tests for metadata-only Debug formatting on DataFrame bridge types.

use cudf::core::Column;

#[test]
fn test_dataframe_table_view_debug_reports_table_metadata_without_values() {
    let df = crate::common::dataframe_from_columns(
        vec!["id".to_string(), "city".to_string()],
        vec![Column::from_slice(&[7i32, 8, 9]).unwrap(), Column::from_strs(&["tokyo", "paris", "taipei"]).unwrap()],
    )
    .unwrap();

    let bridge = df.table_view().unwrap();
    let output = format!("{:?}", bridge);

    assert!(output.contains("DataFrameTableView"), "debug output should contain the bridge type name: {}", output);
    assert!(output.contains("num_columns"), "debug output should contain column count metadata: {}", output);
    assert!(output.contains("num_rows"), "debug output should contain row count metadata: {}", output);
    assert!(output.contains("columns"), "debug output should contain schema metadata: {}", output);
    assert!(!output.contains("tokyo"), "debug output should not materialize payload values: {}", output);
    assert!(!output.contains("paris"), "debug output should not materialize payload values: {}", output);
}
