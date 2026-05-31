//! Tests that verify the crate root re-exports the façade types correctly.
//!
//! These tests mirror the crate-level doc example to ensure it compiles and
//! works as documented. All tests require a GPU.

// ---------------------------------------------------------------------------
// Crate root re-exports: DataFrame, Series, and supporting types
// ---------------------------------------------------------------------------

#[test]
fn test_crate_root_reexports_series() {
    // Series should be importable directly from the crate root.
    let s = cudf::Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    assert_eq!(s.name(), Some("ids"));
    assert_eq!(s.len().unwrap(), 3);
}

#[test]
fn test_crate_root_reexports_dataframe() {
    // DataFrame should be importable directly from the crate root.
    let ids = cudf::Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = cudf::Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = cudf::DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();
    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);
}

#[test]
fn test_doc_example_facade_workflow() {
    // This mirrors the planned crate-level doc example: construct a DataFrame
    // from Series, retrieve a Series back, and inspect metadata.
    use cudf::{DataFrame, Series};

    let ids = Series::new::<i32>("ids", &[1, 2, 3]).unwrap();
    let scores = Series::new::<f64>("scores", &[10.0, 20.0, 30.0]).unwrap();

    let df = DataFrame::from_columns([("id", ids), ("score", scores)]).unwrap();

    assert_eq!(df.num_columns(), 2);
    assert_eq!(df.num_rows().unwrap(), 3);

    // Retrieve a column as a Series.
    let score_series = df.series("score").unwrap();
    assert_eq!(score_series.name(), Some("score"));
    assert_eq!(score_series.len().unwrap(), 3);
}

#[test]
fn test_lowlevel_types_still_reexported() {
    // Ensure the low-level types remain accessible from the crate root.
    let col = cudf::Column::from_slice(&[1i32, 2, 3]).unwrap();
    let view = col.view().unwrap();
    assert_eq!(view.len().unwrap(), 3);

    let _dt: cudf::DataType = view.data_type().unwrap();
    let _tid: cudf::TypeId = _dt.id();
}
