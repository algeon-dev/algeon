use cudf::{
    Column, DataFrame, SortOptions,
    ops::sorting::{self, NullOrder, Order},
};

use crate::allocation::allocation_count_delta;

fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[2i32, 1, 2, 1]).unwrap();
    let scores = Column::from_slice(&[20i32, 10, 40, 30]).unwrap();
    let tags = Column::from_slice(&[200i32, 100, 400, 300]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string(), "tag".to_string()], vec![ids, scores, tags]).unwrap()
}

#[test]
fn subset_key_sort_values_allocates_no_more_than_low_level_borrowed_sort_by_key() {
    let df = sample_df();

    let high_level_allocations = allocation_count_delta(|| {
        let result = df.sort_values(&["id", "score"], SortOptions::new().with_descending(vec![false, true])).unwrap();

        assert_eq!(result.column_names(), vec!["id", "score", "tag"]);
        assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![30, 10, 40, 20]);
        assert_eq!(result.series_view("tag").unwrap().to_vec::<i32>().unwrap(), vec![300, 100, 400, 200]);
        result
    });

    let low_level_allocations = allocation_count_delta(|| {
        let values = df.temp_table_view().unwrap();
        let keys = values.table().select_columns(&[0, 1]).unwrap();

        let result_table = sorting::sort_by_key(
            values.table(),
            &keys,
            Some(&[Order::Ascending, Order::Descending]),
            Some(&[NullOrder::After, NullOrder::After]),
        )
        .unwrap();

        let result =
            DataFrame::from_table(result_table, vec!["id".to_string(), "score".to_string(), "tag".to_string()])
                .unwrap();

        assert_eq!(result.column_names(), vec!["id", "score", "tag"]);
        assert_eq!(result.series_view("id").unwrap().to_vec::<i32>().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(result.series_view("score").unwrap().to_vec::<i32>().unwrap(), vec![30, 10, 40, 20]);
        assert_eq!(result.series_view("tag").unwrap().to_vec::<i32>().unwrap(), vec![300, 100, 400, 200]);
        result
    });

    assert_eq!(
        high_level_allocations, low_level_allocations,
        "subset-key sort_values() should not allocate more than the equivalent low-level borrowed sort_by_key",
    );
}
