//! Tests for `DataFrame::merge` façade and `MergeOptions` / `JoinType`.

use cudf::{
    column::Column,
    dataframe::DataFrame,
    error::ErrorKind,
    join::{JoinType, MergeOptions},
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn left_df() -> DataFrame {
    // id: [1, 2, 3], score: [10, 20, 30]
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    // id: [2, 3, 4], value: [200, 300, 400]
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

// ---------------------------------------------------------------------------
// MergeOptions builder
// ---------------------------------------------------------------------------

#[test]
fn test_merge_options_defaults() {
    let opts = MergeOptions::new();
    assert_eq!(opts.join_type(), JoinType::Inner);
    assert_eq!(opts.suffix_left(), "_left");
    assert_eq!(opts.suffix_right(), "_right");
}

#[test]
fn test_merge_options_builder_on() {
    let opts = MergeOptions::new().on(&["id"]);
    let on = opts.on_keys().unwrap();
    assert_eq!(on.len(), 1);
    assert_eq!(on[0], "id");
}

#[test]
fn test_merge_options_builder_left_right_on() {
    let opts = MergeOptions::new().left_on(&["left_id"]).right_on(&["right_id"]);
    let lo = opts.left_on_keys().unwrap();
    let ro = opts.right_on_keys().unwrap();
    assert_eq!(lo.len(), 1);
    assert_eq!(lo[0], "left_id");
    assert_eq!(ro.len(), 1);
    assert_eq!(ro[0], "right_id");
}

#[test]
fn test_merge_options_builder_how() {
    assert_eq!(MergeOptions::new().how(JoinType::Left).join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_inner().join_type(), JoinType::Inner);
    assert_eq!(MergeOptions::new().how_left().join_type(), JoinType::Left);
    assert_eq!(MergeOptions::new().how_full().join_type(), JoinType::Full);
}

#[test]
fn test_merge_options_builder_suffixes() {
    let opts = MergeOptions::new().suffixes("_x", "_y");
    assert_eq!(opts.suffix_left(), "_x");
    assert_eq!(opts.suffix_right(), "_y");
}

// ---------------------------------------------------------------------------
// Inner merge — same key name (on)
// ---------------------------------------------------------------------------

#[test]
fn test_inner_merge_same_key() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(&["id"])).unwrap();

    // Inner: matches on 2 and 3 → 2 rows.
    assert_eq!(result.num_rows().unwrap(), 2);
    // Schema: shared key + left payload + right payload.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    let mut ids = result.column("id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

// ---------------------------------------------------------------------------
// Left merge — different key names (left_on / right_on)
// ---------------------------------------------------------------------------

#[test]
fn test_left_merge_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result =
        left.merge(&right, MergeOptions::new().how_left().left_on(&["left_id"]).right_on(&["right_id"])).unwrap();

    // Left: all 3 left rows preserved.
    assert_eq!(result.num_rows().unwrap(), 3);
    // With left_on/right_on, both key columns are preserved in MVP.
    assert!(result.column("left_id").is_ok());
    assert!(result.column("right_id").is_ok());
}

// ---------------------------------------------------------------------------
// Suffix handling for overlapping payload names
// ---------------------------------------------------------------------------

#[test]
fn test_merge_suffix_overlapping_payload() {
    // Both sides have "score" as payload.
    let left = left_df();
    let right_ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = Column::from_slice(&[200i32, 300, 400]).unwrap();
    let right = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![right_ids, right_scores]).unwrap();

    let result = left.merge(&right, MergeOptions::new().on(&["id"])).unwrap();

    // Default suffixes: _left, _right for conflicting payloads.
    assert_eq!(result.column_names(), vec!["id", "score_left", "score_right"]);
}

#[test]
fn test_merge_custom_suffixes() {
    let left = left_df();
    let right_ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let right_scores = Column::from_slice(&[200i32, 300, 400]).unwrap();
    let right = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![right_ids, right_scores]).unwrap();

    let result = left.merge(&right, MergeOptions::new().on(&["id"]).suffixes("_x", "_y")).unwrap();

    assert_eq!(result.column_names(), vec!["id", "score_x", "score_y"]);
}

#[test]
fn test_merge_no_suffix_when_no_conflict() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().on(&["id"])).unwrap();

    // No conflict: "score" and "value" are distinct.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// Full merge
// ---------------------------------------------------------------------------

#[test]
fn test_full_merge() {
    let left = left_df();
    let right = right_df();

    let result = left.merge(&right, MergeOptions::new().how_full().on(&["id"])).unwrap();

    // Full: keys 1,2,3,4 → 4 rows.
    assert_eq!(result.num_rows().unwrap(), 4);
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// Invalid key errors
// ---------------------------------------------------------------------------

#[test]
fn test_merge_missing_on_key_in_left() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(&["nonexistent"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::merge");
}

#[test]
fn test_merge_missing_on_key_in_right() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().on(&["score"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_merge_left_on_without_right_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().left_on(&["id"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::merge");
}

#[test]
fn test_merge_right_on_without_left_on() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new().right_on(&["id"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::merge");
}

#[test]
fn test_merge_left_on_right_on_length_mismatch() {
    let left = left_df_asym();
    let right = right_df_asym();

    let err =
        left.merge(&right, MergeOptions::new().left_on(&["left_id"]).right_on(&["right_id", "value"])).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::merge");
}

#[test]
fn test_merge_no_keys_specified() {
    let left = left_df();
    let right = right_df();

    let err = left.merge(&right, MergeOptions::new()).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::merge");
}

// ---------------------------------------------------------------------------
// Multi-key merge
// ---------------------------------------------------------------------------

#[test]
fn test_merge_multi_key_inner() {
    let lk0 = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = Column::from_slice(&[10i32, 20, 10]).unwrap();
    let lp = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let left = DataFrame::new(vec!["a".to_string(), "b".to_string(), "val".to_string()], vec![lk0, lk1, lp]).unwrap();

    let rk0 = Column::from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = Column::from_slice(&[10i32, 10, 20]).unwrap();
    let rp = Column::from_slice(&[1000i32, 2000, 3000]).unwrap();
    let right = DataFrame::new(vec!["a".to_string(), "b".to_string(), "rval".to_string()], vec![rk0, rk1, rp]).unwrap();

    let result = left.merge(&right, MergeOptions::new().on(&["a", "b"])).unwrap();

    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(result.num_rows().unwrap(), 2);
    assert_eq!(result.column_names(), vec!["a", "b", "val", "rval"]);
}

// ---------------------------------------------------------------------------
// NullEquality option
// ---------------------------------------------------------------------------

#[test]
fn test_merge_null_equality_unequal() {
    // With null_equality(Unequal), null keys should NOT match.
    let left_ids = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let left_vals = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let left = DataFrame::new(vec!["id".to_string(), "val".to_string()], vec![left_ids, left_vals]).unwrap();

    let right_ids = Column::from_slice_nullable(&[1i32, 2, 3], &[true, false, true]).unwrap();
    let right_vals = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let right = DataFrame::new(vec!["id".to_string(), "rval".to_string()], vec![right_ids, right_vals]).unwrap();

    use cudf::join::NullEquality;

    // With Equal: null == null matches → 3 rows.
    let eq_result = left.merge(&right, MergeOptions::new().on(&["id"]).null_equality(NullEquality::Equal)).unwrap();
    assert_eq!(eq_result.num_rows().unwrap(), 3);

    // With Unequal: null != null → only rows with id=1 and id=3 match → 2 rows.
    let ne_result = left.merge(&right, MergeOptions::new().on(&["id"]).null_equality(NullEquality::Unequal)).unwrap();
    assert_eq!(ne_result.num_rows().unwrap(), 2);
}
