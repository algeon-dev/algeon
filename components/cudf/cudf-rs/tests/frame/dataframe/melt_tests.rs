//! Tests for DataFrame::melt() (unpivot: wide → long format).
//!
//! All tests require a GPU.

use cudf::{DataFrame, frame::dataframe::MeltOptions};

use crate::common::assert_missing_column_error;

// ===========================================================================
// Basic melt behavior
// ===========================================================================

#[test]
fn test_melt_single_value_var() {
    // df: A=[a,b,c], B=[1,3,5], C=[2,4,6]
    // melt(id_vars=["A"], value_vars=["B"])
    // → A=[a,b,c], variable=["B","B","B"], value=[1,3,5]
    let a = crate::common::series("A", &["a", "b", "c"]).unwrap();
    let b = crate::common::series::<i32>("B", &[1, 3, 5]).unwrap();
    let c = crate::common::series::<i32>("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.num_columns(), 3);

    let var_col = result.series("variable").unwrap();
    assert_eq!(var_col.to_strings_strict().unwrap(), vec!["B", "B", "B"]);

    let val_col = result.series("value").unwrap();
    assert_eq!(val_col.to_vec_strict::<i32>().unwrap(), vec![1, 3, 5]);

    let id_col = result.series("A").unwrap();
    assert_eq!(id_col.to_strings_strict().unwrap(), vec!["a", "b", "c"]);
}

#[test]
fn test_melt_multiple_value_vars() {
    // melt(id_vars=["A"], value_vars=["B", "C"])
    // → A repeated 2x, variable=["B","B","B","C","C","C"], value=[1,3,5,2,4,6]
    let a = crate::common::series("A", &["a", "b", "c"]).unwrap();
    let b = crate::common::series::<i32>("B", &[1, 3, 5]).unwrap();
    let c = crate::common::series::<i32>("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 6);
    assert_eq!(result.num_columns(), 3);

    let var_col = result.series("variable").unwrap();
    assert_eq!(var_col.to_strings_strict().unwrap(), vec!["B", "B", "B", "C", "C", "C"]);

    let val_col = result.series("value").unwrap();
    assert_eq!(val_col.to_vec_strict::<i32>().unwrap(), vec![1, 3, 5, 2, 4, 6]);
}

// ===========================================================================
// Custom variable/value names
// ===========================================================================

#[test]
fn test_melt_custom_var_value_names() {
    let a = crate::common::series("A", &["x"]).unwrap();
    let b = crate::common::series::<i32>("B", &[10]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b)]).unwrap();

    let opts =
        MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).var_name("my_var").value_name("my_val").build();
    let result = df.melt(opts).unwrap();

    assert!(result.contains("my_var"));
    assert!(result.contains("my_val"));
    assert!(!result.contains("variable"));
    assert!(!result.contains("value"));
}

// ===========================================================================
// Default value_vars = all columns not in id_vars
// ===========================================================================

#[test]
fn test_melt_default_value_vars() {
    let a = crate::common::series("A", &["x", "y"]).unwrap();
    let b = crate::common::series::<i32>("B", &[1, 2]).unwrap();
    let c = crate::common::series::<i32>("C", &[3, 4]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b), ("C", c)]).unwrap();

    // Only id_vars specified → value_vars = [B, C]
    let opts = MeltOptions::builder().id_vars(&["A"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 4); // 2 rows × 2 value_vars
    assert_eq!(result.num_columns(), 3); // A, variable, value
}

// ===========================================================================
// No id_vars (all columns become value_vars)
// ===========================================================================

#[test]
fn test_melt_no_id_vars() {
    let b = crate::common::series::<i32>("B", &[1, 2]).unwrap();
    let c = crate::common::series::<i32>("C", &[3, 4]).unwrap();
    let df = DataFrame::from_columns([("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().value_vars(&["B", "C"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.num_columns(), 2); // variable, value (no id columns)
}

// ===========================================================================
// Null preservation
// ===========================================================================

#[test]
fn test_melt_preserves_nulls_in_values() {
    let a = crate::common::series("A", &["x", "y"]).unwrap();
    let b = crate::common::nullable_series::<i32>("B", &[1, 0], &[true, false]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).build();
    let result = df.melt(opts).unwrap();

    let val_col = result.series("value").unwrap();
    assert_eq!(val_col.null_count(), 1);
    let vals = val_col.to_optional_vec::<i32>().unwrap();
    assert_eq!(vals, vec![Some(1), None]);
}

// ===========================================================================
// Edge cases
// ===========================================================================

#[test]
fn test_melt_empty_dataframe() {
    let a = crate::common::series::<&str>("A", &[]).unwrap();
    let b = crate::common::series::<i32>("B", &[]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.num_columns(), 3);
}

#[test]
fn test_melt_all_columns_as_id_vars_returns_empty_long_frame() {
    let a = crate::common::series("A", &["x", "y"]).unwrap();
    let b = crate::common::series::<i32>("B", &[1, 2]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A", "B"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.num_columns(), 4);
    assert_eq!(result.series("A").unwrap().to_strings_strict().unwrap(), Vec::<String>::new());
    assert_eq!(result.series("B").unwrap().to_vec_strict::<i32>().unwrap(), Vec::<i32>::new());
    assert_eq!(result.series("variable").unwrap().to_strings_strict().unwrap(), Vec::<String>::new());
    assert_eq!(result.series("value").unwrap().to_vec_strict::<i32>().unwrap(), Vec::<i32>::new());
}

#[test]
fn test_melt_single_row() {
    let a = crate::common::series("A", &["x"]).unwrap();
    let b = crate::common::series::<i32>("B", &[10]).unwrap();
    let c = crate::common::series::<i32>("C", &[20]).unwrap();
    let df = DataFrame::from_columns([("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_melt_missing_id_var_returns_error() {
    let b = crate::common::series::<i32>("B", &[1]).unwrap();
    let df = DataFrame::from_columns([("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["nonexistent"]).value_vars(&["B"]).build();
    let err = df.melt(opts).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}

#[test]
fn test_melt_missing_value_var_returns_error() {
    let a = crate::common::series::<i32>("A", &[1]).unwrap();
    let df = DataFrame::from_columns([("A", a)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["nonexistent"]).build();
    let err = df.melt(opts).unwrap_err();
    assert_missing_column_error(&err, "nonexistent");
}
