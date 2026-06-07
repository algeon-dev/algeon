use cudf::{
    Agg, AggSpec, DataFrame,
    core::{Column, TableView},
    ops::groupby::{AggregationRequest, NullPolicy, groupby_aggregate},
};

use crate::allocation::allocation_count_delta;

fn sample_df() -> DataFrame {
    let keys = Column::from_slice(&[1i32, 1, 2, 2]).unwrap();
    let a = Column::from_slice(&[10i64, 20, 30, 40]).unwrap();
    let b = Column::from_slice(&[1i64, 2, 3, 4]).unwrap();
    crate::common::dataframe_from_columns(vec!["key".to_string(), "a".to_string(), "b".to_string()], vec![keys, a, b])
        .unwrap()
}

#[test]
fn groupby_agg_allocates_no_more_than_low_level_borrowed_primitive() {
    let df = sample_df();

    let high_level_allocations = allocation_count_delta(|| {
        let result = df
            .groupby(["key"])
            .agg(&[AggSpec::new("a", Agg::Sum), AggSpec::new("b", Agg::Sum), AggSpec::new("a", Agg::Min)])
            .unwrap();

        assert_eq!(result.column_names(), vec!["key", "a_sum", "b_sum", "a_min"]);
        assert_eq!(result.num_rows(), 2);
        let _ = result.series("key").unwrap().to_vec_strict::<i32>().unwrap();
        result
    });

    let low_level_allocations = allocation_count_delta(|| {
        let key_series = df.series("key").unwrap();
        let key_cv = key_series.as_column_view().unwrap();
        let key_table = TableView::from_column_views(&[&key_cv]).unwrap();

        let a_series = df.series("a").unwrap();
        let b_series = df.series("b").unwrap();
        let a_cv = a_series.as_column_view().unwrap();
        let b_cv = b_series.as_column_view().unwrap();
        let a_aggs = [Agg::Sum, Agg::Min];
        let b_aggs = [Agg::Sum];
        let requests = [
            AggregationRequest { values: &a_cv, aggregations: &a_aggs },
            AggregationRequest { values: &b_cv, aggregations: &b_aggs },
        ];

        let result = groupby_aggregate(&key_table, &requests, NullPolicy::Exclude).unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].len(), 2);
        assert_eq!(result.results[1].len(), 1);
        {
            let key_view = result.keys.view().unwrap().column(0).unwrap();
            let _ = key_view.to_vec::<i32>().unwrap();
        }
        result
    });

    assert_eq!(
        high_level_allocations, low_level_allocations,
        "high-level groupby should not allocate more than the equivalent low-level borrowed primitive",
    );
}
