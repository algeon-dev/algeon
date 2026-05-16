#[test]
fn workload_modules_remain_available_at_the_crate_root() {
    assert_eq!(datafusion_cudf::tpch::TPCH_QUERY_COUNT, datafusion_cudf::TPCH_QUERY_COUNT);
    assert_eq!(datafusion_cudf::tpcds::TPCDS_QUERY_COUNT, datafusion_cudf::TPCDS_QUERY_COUNT);
    assert_eq!(datafusion_cudf::tpch::TPCH_QUERIES.len(), datafusion_cudf::TPCH_QUERIES.len());
    assert_eq!(datafusion_cudf::tpcds::TPCDS_QUERIES.len(), datafusion_cudf::TPCDS_QUERIES.len());
}
