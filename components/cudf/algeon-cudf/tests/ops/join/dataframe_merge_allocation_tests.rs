use cudf::{DataFrame, MergeOptions};

use crate::{allocation::allocation_count_delta, common::column_from_slice};

fn left_df() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df() -> DataFrame {
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

fn left_df_asym() -> DataFrame {
    let ids = column_from_slice(&[1i32, 2, 3]).unwrap();
    let scores = column_from_slice(&[10i32, 20, 30]).unwrap();
    crate::common::dataframe_from_columns(vec!["left_id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}

fn right_df_asym() -> DataFrame {
    let ids = column_from_slice(&[2i32, 3, 4]).unwrap();
    let values = column_from_slice(&[200i32, 300, 400]).unwrap();
    crate::common::dataframe_from_columns(vec!["right_id".to_string(), "value".to_string()], vec![ids, values]).unwrap()
}

#[test]
fn merge_same_key_allocates_no_more_than_merge_split() {
    let left = left_df();
    let right = right_df();
    let options = MergeOptions::new().on(["id"]);

    let merge_allocations = allocation_count_delta(|| {
        let result = left.merge(&right, options.clone()).unwrap();
        assert_eq!(result.column_names(), vec!["id", "score", "value"]);
        assert_eq!(result.num_rows(), 2);
        let _ = result.series("id").unwrap().to_vec_strict::<i32>().unwrap();
        result
    });

    let split_allocations = allocation_count_delta(|| {
        let result = left.merge_split(&right, options.clone()).unwrap();
        assert_eq!(result.left.column_names(), vec!["id", "score"]);
        assert_eq!(result.right.column_names(), vec!["id", "value"]);
        assert_eq!(result.left.num_rows(), 2);
        assert_eq!(result.right.num_rows(), 2);
        let _ = result.left.series("id").unwrap().to_vec_strict::<i32>().unwrap();
        let _ = result.right.series("id").unwrap().to_vec_strict::<i32>().unwrap();
        result
    });

    assert!(
        merge_allocations <= split_allocations,
        "merge() should not allocate more than merge_split() for an inner join on shared keys",
    );
}

#[test]
fn merge_asymmetric_keys_allocates_no_more_than_merge_split() {
    let left = left_df_asym();
    let right = right_df_asym();
    let options = MergeOptions::new().how_left().left_on(["left_id"]).right_on(["right_id"]);

    let merge_allocations = allocation_count_delta(|| {
        let result = left.merge(&right, options.clone()).unwrap();
        assert_eq!(result.column_names(), vec!["left_id", "score", "right_id", "value"]);
        assert_eq!(result.num_rows(), 3);
        let _ = result.series("left_id").unwrap().to_vec_strict::<i32>().unwrap();
        result
    });

    let split_allocations = allocation_count_delta(|| {
        let result = left.merge_split(&right, options.clone()).unwrap();
        assert_eq!(result.left.column_names(), vec!["left_id", "score"]);
        assert_eq!(result.right.column_names(), vec!["right_id", "value"]);
        assert_eq!(result.left.num_rows(), 3);
        assert_eq!(result.right.num_rows(), 3);
        let _ = result.left.series("left_id").unwrap().to_vec_strict::<i32>().unwrap();
        let _ = result.right.series("right_id").unwrap().to_optional_vec::<i32>().unwrap();
        result
    });

    assert!(
        merge_allocations <= split_allocations,
        "merge() should not allocate more than merge_split() for asymmetric non-full joins",
    );
}
