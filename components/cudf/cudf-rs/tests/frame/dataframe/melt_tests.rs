//! Tests for DataFrame::melt() (unpivot: wide → long format).
//!
//! All tests require a GPU.

use cudf::{DataFrame, MeltOptions, Series};

// ===========================================================================
// Basic melt behavior
// ===========================================================================

#[test]
fn test_melt_single_value_var() {
    // df: A=[a,b,c], B=[1,3,5], C=[2,4,6]
    // melt(id_vars=["A"], value_vars=["B"])
    // → A=[a,b,c], variable=["B","B","B"], value=[1,3,5]
    let a = Series::from_strings("A", &["a", "b", "c"]).unwrap();
    let b = Series::new::<i32>("B", &[1, 3, 5]).unwrap();
    let c = Series::new::<i32>("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.num_columns(), 3);

    let var_col = result.series("variable").unwrap();
    assert_eq!(var_col.to_strings().unwrap(), vec!["B", "B", "B"]);

    let val_col = result.series("value").unwrap();
    assert_eq!(val_col.to_vec::<i32>().unwrap(), vec![1, 3, 5]);

    let id_col = result.series("A").unwrap();
    assert_eq!(id_col.to_strings().unwrap(), vec!["a", "b", "c"]);
}

#[test]
fn test_melt_multiple_value_vars() {
    // melt(id_vars=["A"], value_vars=["B", "C"])
    // → A repeated 2x, variable=["B","B","B","C","C","C"], value=[1,3,5,2,4,6]
    let a = Series::from_strings("A", &["a", "b", "c"]).unwrap();
    let b = Series::new::<i32>("B", &[1, 3, 5]).unwrap();
    let c = Series::new::<i32>("C", &[2, 4, 6]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 6);
    assert_eq!(result.num_columns(), 3);

    let var_col = result.series("variable").unwrap();
    assert_eq!(var_col.to_strings().unwrap(), vec!["B", "B", "B", "C", "C", "C"]);

    let val_col = result.series("value").unwrap();
    assert_eq!(val_col.to_vec::<i32>().unwrap(), vec![1, 3, 5, 2, 4, 6]);
}

// ===========================================================================
// Custom variable/value names
// ===========================================================================

#[test]
fn test_melt_custom_var_value_names() {
    let a = Series::from_strings("A", &["x"]).unwrap();
    let b = Series::new::<i32>("B", &[10]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b)]).unwrap();

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
    let a = Series::from_strings("A", &["x", "y"]).unwrap();
    let b = Series::new::<i32>("B", &[1, 2]).unwrap();
    let c = Series::new::<i32>("C", &[3, 4]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b), ("C", c)]).unwrap();

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
    let b = Series::new::<i32>("B", &[1, 2]).unwrap();
    let c = Series::new::<i32>("C", &[3, 4]).unwrap();
    let df = DataFrame::from_series(vec![("B", b), ("C", c)]).unwrap();

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
    let a = Series::from_strings("A", &["x", "y"]).unwrap();
    let b = Series::new_nullable::<i32>("B", &[1, 0], &[true, false]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b)]).unwrap();

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
    let a = Series::from_strings("A", &[]).unwrap();
    let b = Series::new::<i32>("B", &[]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 0);
    assert_eq!(result.num_columns(), 3);
}

#[test]
fn test_melt_single_row() {
    let a = Series::from_strings("A", &["x"]).unwrap();
    let b = Series::new::<i32>("B", &[10]).unwrap();
    let c = Series::new::<i32>("C", &[20]).unwrap();
    let df = DataFrame::from_series(vec![("A", a), ("B", b), ("C", c)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["B", "C"]).build();
    let result = df.melt(opts).unwrap();

    assert_eq!(result.num_rows(), 2);
}

#[test]
fn test_melt_missing_id_var_returns_error() {
    let b = Series::new::<i32>("B", &[1]).unwrap();
    let df = DataFrame::from_series(vec![("B", b)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["nonexistent"]).value_vars(&["B"]).build();
    let result = df.melt(opts);
    assert!(result.is_err());
}

#[test]
fn test_melt_missing_value_var_returns_error() {
    let a = Series::new::<i32>("A", &[1]).unwrap();
    let df = DataFrame::from_series(vec![("A", a)]).unwrap();

    let opts = MeltOptions::builder().id_vars(&["A"]).value_vars(&["nonexistent"]).build();
    let result = df.melt(opts);
    assert!(result.is_err());
}
