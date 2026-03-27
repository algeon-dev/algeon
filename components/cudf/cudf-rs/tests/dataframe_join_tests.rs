#![allow(deprecated)] // Tests exercise deprecated DataFrame::column()
//! Tests for DataFrame joins.

use cudf::{column::Column, dataframe::DataFrame, error::ErrorKind};

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

// ---------------------------------------------------------------------------
// inner_join
// ---------------------------------------------------------------------------

#[test]
fn test_inner_join_single_key() {
    let left = left_df();
    let right = right_df();

    let result = left.inner_join(&right, &["id"]).unwrap();

    // Inner join on id: matches are 2 and 3.
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.num_columns(), 3);

    // Merged schema: left keys + left payload + right payload.
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);

    // Verify matched data (sort for deterministic comparison).
    let ids = result.column("id").unwrap().to_vec::<i32>().unwrap();
    let scores = result.column("score").unwrap().to_vec::<i32>().unwrap();
    let values = result.column("value").unwrap().to_vec::<i32>().unwrap();

    let mut triples: Vec<(i32, i32, i32)> =
        ids.into_iter().zip(scores.into_iter().zip(values)).map(|(k, (s, v))| (k, s, v)).collect();
    triples.sort_by_key(|&(k, _, _)| k);
    assert_eq!(triples, vec![(2, 20, 200), (3, 30, 300)]);
}

#[test]
fn test_inner_join_multi_key() {
    let lk0 = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = Column::from_slice(&[10i32, 20, 10]).unwrap();
    let lp = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let left = DataFrame::new(vec!["a".to_string(), "b".to_string(), "val".to_string()], vec![lk0, lk1, lp]).unwrap();

    let rk0 = Column::from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = Column::from_slice(&[10i32, 10, 20]).unwrap();
    let rp = Column::from_slice(&[1000i32, 2000, 3000]).unwrap();
    let right = DataFrame::new(vec!["a".to_string(), "b".to_string(), "rval".to_string()], vec![rk0, rk1, rp]).unwrap();

    let result = left.inner_join(&right, &["a", "b"]).unwrap();

    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.column_names(), vec!["a", "b", "val", "rval"]);
}

// ---------------------------------------------------------------------------
// left_join
// ---------------------------------------------------------------------------

#[test]
fn test_left_join_preserves_all_left_rows() {
    let left = left_df();
    let right = right_df();

    let result = left.left_join(&right, &["id"]).unwrap();

    // Left join: all 3 left rows preserved.
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// full_join
// ---------------------------------------------------------------------------

#[test]
fn test_full_join_preserves_all_rows() {
    let left = left_df();
    let right = right_df();

    let result = left.full_join(&right, &["id"]).unwrap();

    // Full join: keys 1,2,3,4 → 4 rows total.
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["id", "score", "value"]);
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_join_nonexistent_key_in_left() {
    let left = left_df();
    let right = right_df();

    let result = left.inner_join(&right, &["nonexistent"]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

#[test]
fn test_join_nonexistent_key_in_right() {
    let left = left_df();

    // Right DataFrame without the "id" column.
    let c1 = Column::from_slice(&[1i32, 2]).unwrap();
    let right = DataFrame::new(vec!["other".to_string()], vec![c1]).unwrap();

    let result = left.inner_join(&right, &["id"]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
}

// ---------------------------------------------------------------------------
// Schema naming
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Asymmetric join keys (left_on / right_on)
// ---------------------------------------------------------------------------

fn left_df_asym() -> DataFrame {
    // left_id: [1, 2, 3], score: [10, 20, 30]
    let ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let scores = Column::from_slice(&[10i32, 20, 30]).unwrap();
    DataFrame::new(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    // right_id: [2, 3, 4], value: [200, 300, 400]
    let ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let values = Column::from_slice(&[200i32, 300, 400]).unwrap();
    DataFrame::new(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

#[test]
fn test_inner_join_on_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.inner_join_on(&right, &["left_id"], &["right_id"]).unwrap();

    // Inner join: matches on 2 and 3 → 2 rows.
    assert_eq!(result.num_rows(), 2);

    // Schema: left key name + left payload + right payload.
    assert_eq!(result.column_names(), vec!["left_id", "score", "value"]);

    let mut ids = result.column("left_id").unwrap().to_vec::<i32>().unwrap();
    ids.sort();
    assert_eq!(ids, vec![2, 3]);
}

#[test]
fn test_left_join_on_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.left_join_on(&right, &["left_id"], &["right_id"]).unwrap();

    // All 3 left rows preserved.
    assert_eq!(result.num_rows(), 3);
    assert_eq!(result.column_names(), vec!["left_id", "score", "value"]);
}

#[test]
fn test_full_join_on_asymmetric_keys() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.full_join_on(&right, &["left_id"], &["right_id"]).unwrap();

    // Full join: keys 1, 2, 3, 4 → 4 rows.
    assert_eq!(result.num_rows(), 4);
    assert_eq!(result.column_names(), vec!["left_id", "score", "value"]);
}

#[test]
fn test_join_on_multi_key_asymmetric() {
    let lk0 = Column::from_slice(&[1i32, 1, 2]).unwrap();
    let lk1 = Column::from_slice(&[10i32, 20, 10]).unwrap();
    let lp = Column::from_slice(&[100i32, 200, 300]).unwrap();
    let left = DataFrame::new(vec!["la".to_string(), "lb".to_string(), "val".to_string()], vec![lk0, lk1, lp]).unwrap();

    let rk0 = Column::from_slice(&[1i32, 2, 2]).unwrap();
    let rk1 = Column::from_slice(&[10i32, 10, 20]).unwrap();
    let rp = Column::from_slice(&[1000i32, 2000, 3000]).unwrap();
    let right =
        DataFrame::new(vec!["ra".to_string(), "rb".to_string(), "rval".to_string()], vec![rk0, rk1, rp]).unwrap();

    let result = left.inner_join_on(&right, &["la", "lb"], &["ra", "rb"]).unwrap();

    // Matches: (1,10) and (2,10) → 2 rows.
    assert_eq!(result.num_rows(), 2);
    // Output uses left key names.
    assert_eq!(result.column_names(), vec!["la", "lb", "val", "rval"]);
}

#[test]
fn test_join_on_mismatched_key_counts() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.inner_join_on(&right, &["left_id"], &["right_id", "value"]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::inner_join_on");
}

#[test]
fn test_join_on_left_key_missing() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.inner_join_on(&right, &["nonexistent"], &["right_id"]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::inner_join_on");
}

#[test]
fn test_join_on_right_key_missing() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.inner_join_on(&right, &["left_id"], &["nonexistent"]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "dataframe::inner_join_on");
}

#[test]
fn test_join_on_schema_uses_left_names() {
    let left = left_df_asym();
    let right = right_df_asym();

    let result = left.inner_join_on(&right, &["left_id"], &["right_id"]).unwrap();

    // Key column should use the left name "left_id", not the right name "right_id".
    assert!(result.column("left_id").is_ok());
    assert!(result.column("right_id").is_err());
}

#[test]
fn test_join_on_right_payload_collision() {
    // Left: left_id, score
    let left = left_df_asym();

    // Right: right_id, score (payload "score" collides with left payload).
    let rids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let rscores = Column::from_slice(&[200i32, 300, 400]).unwrap();
    let right = DataFrame::new(vec!["right_id".to_string(), "score".to_string()], vec![rids, rscores]).unwrap();

    let result = left.inner_join_on(&right, &["left_id"], &["right_id"]).unwrap();

    assert_eq!(result.column_names(), vec!["left_id", "score", "score_right"]);
}

#[test]
fn test_join_result_schema_collision_suffixes_right_payload() {
    let left_ids = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let left_score = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let left = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![left_ids, left_score]).unwrap();

    // Right has payload column named "score" too (collides with left).
    let right_ids = Column::from_slice(&[2i32, 3, 4]).unwrap();
    let right_score = Column::from_slice(&[200i32, 300, 400]).unwrap();
    let right = DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![right_ids, right_score]).unwrap();

    let result = left.inner_join(&right, &["id"]).unwrap();

    // Right payload should be suffixed.
    assert_eq!(result.column_names(), vec!["id", "score", "score_right"]);
}
