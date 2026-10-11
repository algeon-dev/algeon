//! Tests verifying that Arc-based column sharing works correctly.
//!
//! These tests demonstrate that zero-copy operations (clone, select, rename,
//! with_column) share GPU memory via `Arc<Column>` rather than deep-copying.
//!
//! Sharing is verified by comparing device pointers: two `ColumnView`s that
//! originate from the same `Arc<Column>` will report the same `as_ptr()`.

use cudf::{DataFrame, FrameMemoryEffect};

use crate::common::{column_from_slice, dataframe_device_ptr, dataframe_from_columns};

/// Helper: build a 3-row, 3-column DataFrame with distinct column types.
fn sample_df() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let flags = column_from_slice(&[true, false, true]).unwrap();
    dataframe_from_columns(vec!["id".to_string(), "score".to_string(), "flag".to_string()], vec![ids, scores, flags])
        .unwrap()
}

// ---------------------------------------------------------------------------
// clone() produces shared columns
// ---------------------------------------------------------------------------

fn test_clone_shares_all_columns() {
    let df = sample_df();
    let copied = df.clone();

    // Both DataFrames should point to the same GPU memory for every column.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copied, "id"));
    assert_eq!(dataframe_device_ptr(&df, "score"), dataframe_device_ptr(&copied, "score"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&copied, "flag"));
}

fn test_clone_both_dfs_independently_readable() {
    let df = sample_df();
    let copied = df.clone();

    // Both DataFrames should be independently readable.
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(copied.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);

    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(copied.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);

    assert_eq!(df.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
    assert_eq!(copied.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
}

fn test_clone_preserves_schema() {
    let df = sample_df();
    let copied = df.clone();

    assert_eq!(df.num_columns(), copied.num_columns());
    assert_eq!(df.num_rows(), copied.num_rows());
    assert_eq!(df.column_names(), copied.column_names());
}

// ---------------------------------------------------------------------------
// select() shares subset of columns
// ---------------------------------------------------------------------------

fn test_select_shares_selected_columns() {
    let df = sample_df();
    let selected = df.select(["id", "flag"]).unwrap();

    // Selected columns should point to the same GPU memory.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&selected, "id"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&selected, "flag"));
}

fn test_select_original_still_usable() {
    let df = sample_df();
    let selected = df.select(["score"]).unwrap();

    // Original DataFrame still has all columns and correct data.
    assert_eq!(df.num_columns(), 3);
    assert_eq!(df.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(df.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);

    // Selected DataFrame has only the chosen column.
    assert_eq!(selected.num_columns(), 1);
    assert_eq!(selected.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

fn test_select_reorder_shares_columns() {
    let df = sample_df();
    let reordered = df.select(["flag", "id"]).unwrap();

    // Even after reordering, the GPU memory is shared.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&reordered, "id"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&reordered, "flag"));
}

fn test_select_memory_effect_reports_retained_owner_aliases() {
    let df = sample_df();
    let selected = df.select(["flag", "id"]).unwrap();
    let effect = df.select_memory_effect(["flag", "id"]).unwrap();

    assert!(effect.is_alias_of_input());
    let FrameMemoryEffect::AliasOfInput { input, output_columns } = effect else {
        panic!("select should report an alias-of-input memory effect");
    };

    assert_eq!(input, df.memory_identity());
    assert_eq!(output_columns.len(), 2);
    assert_eq!(output_columns[0].input_position, 2);
    assert_eq!(output_columns[0].input_name, "flag");
    assert_eq!(output_columns[0].output_position, 0);
    assert_eq!(output_columns[0].output_name, "flag");
    assert_eq!(output_columns[1].input_position, 0);
    assert_eq!(output_columns[1].input_name, "id");
    assert_eq!(output_columns[1].output_position, 1);
    assert_eq!(output_columns[1].output_name, "id");

    let selected_identity = selected.memory_identity();
    assert_eq!(output_columns[0].identity, selected_identity.columns[0].identity);
    assert_eq!(output_columns[1].identity, selected_identity.columns[1].identity);
}

fn test_empty_select_memory_effect_reports_zero_column_alias() {
    let df = sample_df();
    let effect = df.select_memory_effect(std::iter::empty::<&str>()).unwrap();

    let FrameMemoryEffect::AliasOfInput { input, output_columns } = effect else {
        panic!("empty select should still be a schema-only alias operation");
    };
    assert_eq!(input, df.memory_identity());
    assert!(output_columns.is_empty());
}

fn test_select_as_allows_multiple_output_aliases_for_one_owner() {
    let df = sample_df();
    let selected = df.select_as([("score", "sum_score"), ("score", "min_score")]).unwrap();
    let effect = df.select_as_memory_effect([("score", "sum_score"), ("score", "min_score")]).unwrap();

    assert_eq!(selected.column_names(), &["sum_score", "min_score"]);
    assert_eq!(dataframe_device_ptr(&selected, "sum_score"), dataframe_device_ptr(&selected, "min_score"));
    let FrameMemoryEffect::AliasOfInput { output_columns, .. } = effect else {
        panic!("select_as should report an alias-of-input memory effect");
    };
    assert_eq!(output_columns[0].input_position, output_columns[1].input_position);
    assert_eq!(output_columns[0].identity, output_columns[1].identity);
    assert_eq!(output_columns[0].output_name, "sum_score");
    assert_eq!(output_columns[1].output_name, "min_score");
}

fn test_select_as_rejects_duplicate_output_names() {
    let df = sample_df();
    let error = df.select_as([("id", "duplicate"), ("score", "duplicate")]).expect_err("output names remain unique");
    assert_eq!(error.detail_value("duplicate_name"), Some("duplicate"));
}

// ---------------------------------------------------------------------------
// rename() shares all column data
// ---------------------------------------------------------------------------

fn test_rename_shares_all_column_data() {
    let df = sample_df();
    let renamed = df.rename_column("id", "identifier").unwrap();

    // The renamed column should share the same GPU memory.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&renamed, "identifier"));

    // Other columns should also be shared.
    assert_eq!(dataframe_device_ptr(&df, "score"), dataframe_device_ptr(&renamed, "score"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&renamed, "flag"));
}

fn test_rename_original_unchanged() {
    let df = sample_df();
    let renamed = df.rename_column("score", "points").unwrap();

    // Original still has "score", renamed has "points".
    assert_eq!(df.column_names(), &["id", "score", "flag"]);
    assert_eq!(renamed.column_names(), &["id", "points", "flag"]);

    // Data is the same.
    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(renamed.series("points").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// with_column() only allocates for the new/replaced column
// ---------------------------------------------------------------------------

fn test_with_column_unchanged_columns_shared() {
    let df = sample_df();
    let new_col = column_from_slice(&[100i32, 200, 300]).unwrap();
    let result = df.with_column("extra", new_col).unwrap();

    // Existing columns should share GPU memory.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&result, "id"));
    assert_eq!(dataframe_device_ptr(&df, "score"), dataframe_device_ptr(&result, "score"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&result, "flag"));
}

fn test_with_column_replaced_column_has_new_allocation() {
    let df = sample_df();
    let original_score_ptr = dataframe_device_ptr(&df, "score");

    let new_score = column_from_slice(&[99.0f64, 88.0, 77.0]).unwrap();
    let result = df.with_column("score", new_score).unwrap();

    // The replaced column should have a DIFFERENT device pointer (new GPU allocation).
    assert_ne!(original_score_ptr, dataframe_device_ptr(&result, "score"));

    // Unchanged columns should still be shared.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&result, "id"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&result, "flag"));
}

fn test_with_column_replaced_column_has_correct_data() {
    let df = sample_df();
    let new_score = column_from_slice(&[99.0f64, 88.0, 77.0]).unwrap();
    let result = df.with_column("score", new_score).unwrap();

    assert_eq!(result.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![99.0, 88.0, 77.0]);

    // Original is unaffected.
    assert_eq!(df.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
}

// ---------------------------------------------------------------------------
// Multiple clone() calls don't OOM (zero-cost sharing)
// ---------------------------------------------------------------------------

fn test_many_clones_are_zero_cost() {
    let df = sample_df();

    // Create 1000 clones. Since clone() is just Arc::clone, this should use
    // virtually no additional GPU memory.
    let copies: Vec<DataFrame> = (0..1000).map(|_| df.clone()).collect();

    // All copies should be independently readable with correct data.
    for (i, c) in copies.iter().enumerate() {
        assert_eq!(c.num_columns(), 3, "copy {} has wrong column count", i);
        assert_eq!(c.num_rows(), 3, "copy {} has wrong row count", i);
    }

    // Spot-check a few copies share the same GPU pointers.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copies[0], "id"));
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copies[500], "id"));
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copies[999], "id"));
}

fn test_many_clones_readable_after_original_dropped() {
    let copies: Vec<DataFrame> = {
        let df = sample_df();
        (0..100).map(|_| df.clone()).collect()
        // df is dropped here
    };

    // All copies should still be readable (Arc refcount keeps Column alive).
    for c in &copies {
        assert_eq!(c.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 2, 3]);
        assert_eq!(c.series("score").unwrap().to_vec_strict::<f64>().unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(c.series("flag").unwrap().to_vec_strict::<bool>().unwrap(), vec![true, false, true]);
    }
}

// ---------------------------------------------------------------------------
// Combined operations preserve sharing where expected
// ---------------------------------------------------------------------------

fn test_clone_then_select_shares_columns() {
    let df = sample_df();
    let copied = df.clone();
    let selected = copied.select(["id"]).unwrap();

    // All three DataFrames share the "id" column.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copied, "id"));
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&selected, "id"));
}

fn test_rename_then_select_shares_columns() {
    let df = sample_df();
    let renamed = df.rename_column("id", "identifier").unwrap();
    let selected = renamed.select(["identifier", "flag"]).unwrap();

    // The "identifier" column in selected is the same as "id" in original.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&selected, "identifier"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&selected, "flag"));
}

fn test_with_column_then_clone_shares_new_and_old() {
    let df = sample_df();
    let new_col = column_from_slice(&[100i32, 200, 300]).unwrap();
    let with_extra = df.with_column("extra", new_col).unwrap();
    let copied = with_extra.clone();

    // Original columns are shared across all three DataFrames.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&with_extra, "id"));
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&copied, "id"));

    // The new column is shared between with_extra and copied.
    assert_eq!(dataframe_device_ptr(&with_extra, "extra"), dataframe_device_ptr(&copied, "extra"));
}

fn test_chained_renames_share_all_data() {
    let df = sample_df();
    let r1 = df.rename_column("id", "a").unwrap();
    let r2 = r1.rename_column("score", "b").unwrap();
    let r3 = r2.rename_column("flag", "c").unwrap();

    // Despite three renames, all columns share the same GPU data.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&r3, "a"));
    assert_eq!(dataframe_device_ptr(&df, "score"), dataframe_device_ptr(&r3, "b"));
    assert_eq!(dataframe_device_ptr(&df, "flag"), dataframe_device_ptr(&r3, "c"));
}

fn test_select_subset_then_with_column_shares_retained() {
    let df = sample_df();
    let selected = df.select(["id", "score"]).unwrap();
    let new_flag = column_from_slice(&[false, true, false]).unwrap();
    let result = selected.with_column("new_flag", new_flag).unwrap();

    // "id" and "score" are shared all the way through.
    assert_eq!(dataframe_device_ptr(&df, "id"), dataframe_device_ptr(&result, "id"));
    assert_eq!(dataframe_device_ptr(&df, "score"), dataframe_device_ptr(&result, "score"));
}

pub(super) const CASES: &[(&str, fn())] = &[
    ("arc_sharing_tests::test_clone_shares_all_columns", test_clone_shares_all_columns),
    ("arc_sharing_tests::test_clone_both_dfs_independently_readable", test_clone_both_dfs_independently_readable),
    ("arc_sharing_tests::test_clone_preserves_schema", test_clone_preserves_schema),
    ("arc_sharing_tests::test_select_shares_selected_columns", test_select_shares_selected_columns),
    ("arc_sharing_tests::test_select_original_still_usable", test_select_original_still_usable),
    ("arc_sharing_tests::test_select_reorder_shares_columns", test_select_reorder_shares_columns),
    (
        "arc_sharing_tests::test_select_memory_effect_reports_retained_owner_aliases",
        test_select_memory_effect_reports_retained_owner_aliases,
    ),
    (
        "arc_sharing_tests::test_empty_select_memory_effect_reports_zero_column_alias",
        test_empty_select_memory_effect_reports_zero_column_alias,
    ),
    (
        "arc_sharing_tests::test_select_as_allows_multiple_output_aliases_for_one_owner",
        test_select_as_allows_multiple_output_aliases_for_one_owner,
    ),
    ("arc_sharing_tests::test_select_as_rejects_duplicate_output_names", test_select_as_rejects_duplicate_output_names),
    ("arc_sharing_tests::test_rename_shares_all_column_data", test_rename_shares_all_column_data),
    ("arc_sharing_tests::test_rename_original_unchanged", test_rename_original_unchanged),
    ("arc_sharing_tests::test_with_column_unchanged_columns_shared", test_with_column_unchanged_columns_shared),
    (
        "arc_sharing_tests::test_with_column_replaced_column_has_new_allocation",
        test_with_column_replaced_column_has_new_allocation,
    ),
    (
        "arc_sharing_tests::test_with_column_replaced_column_has_correct_data",
        test_with_column_replaced_column_has_correct_data,
    ),
    ("arc_sharing_tests::test_many_clones_are_zero_cost", test_many_clones_are_zero_cost),
    (
        "arc_sharing_tests::test_many_clones_readable_after_original_dropped",
        test_many_clones_readable_after_original_dropped,
    ),
    ("arc_sharing_tests::test_clone_then_select_shares_columns", test_clone_then_select_shares_columns),
    ("arc_sharing_tests::test_rename_then_select_shares_columns", test_rename_then_select_shares_columns),
    (
        "arc_sharing_tests::test_with_column_then_clone_shares_new_and_old",
        test_with_column_then_clone_shares_new_and_old,
    ),
    ("arc_sharing_tests::test_chained_renames_share_all_data", test_chained_renames_share_all_data),
    (
        "arc_sharing_tests::test_select_subset_then_with_column_shares_retained",
        test_select_subset_then_with_column_shares_retained,
    ),
];
