//! Smoke tests for Series Display and Debug formatting.

#[test]
fn test_display_smoke_reports_series_metadata_without_materializing_values() {
    let series = crate::common::nullable_series("score", &[111i32, 0, 333], &[true, false, true]).unwrap();

    let output = format!("{}", series);

    assert!(output.contains("Series"), "output should identify a Series: {}", output);
    assert!(output.contains("name=score"), "output should contain the series name: {}", output);
    assert!(
        output.contains("dtype=Int32") || output.contains("dtype=int32") || output.contains("dtype=INT32"),
        "output should contain Int32 metadata: {}",
        output
    );
    assert!(output.contains("len=3"), "output should contain the series length: {}", output);
    assert!(output.contains("nullable"), "output should contain nullability metadata: {}", output);
    assert!(output.contains("nulls=1"), "output should contain null-count metadata: {}", output);
    assert!(!output.contains("111"), "output should not materialize integer payloads: {}", output);
    assert!(!output.contains("333"), "output should not materialize integer payloads: {}", output);
}

#[test]
fn test_debug_smoke_reports_series_metadata_without_materializing_values() {
    let series = crate::common::series("city", &["tokyo", "paris", "taipei"]).unwrap();

    let output = format!("{:?}", series);

    assert!(output.contains("Series"), "debug output should identify a Series: {}", output);
    assert!(output.contains("city"), "debug output should contain the series name: {}", output);
    assert!(
        output.contains("String") || output.contains("string") || output.contains("STRING"),
        "debug output should contain String metadata: {}",
        output
    );
    assert!(output.contains("len"), "debug output should contain length metadata: {}", output);
    assert!(!output.contains("tokyo"), "debug output should not materialize string payloads: {}", output);
    assert!(!output.contains("paris"), "debug output should not materialize string payloads: {}", output);
}
