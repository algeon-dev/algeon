#[test]
fn workload_modules_remain_available_at_the_crate_root() {
    assert_eq!(datafusion_nexus::tpch::TPCH_QUERY_COUNT, datafusion_nexus::TPCH_QUERY_COUNT);
    assert_eq!(datafusion_nexus::tpcds::TPCDS_QUERY_COUNT, datafusion_nexus::TPCDS_QUERY_COUNT);
    assert_eq!(datafusion_nexus::tpch::TPCH_QUERIES.len(), datafusion_nexus::TPCH_QUERIES.len());
    assert_eq!(datafusion_nexus::tpcds::TPCDS_QUERIES.len(), datafusion_nexus::TPCDS_QUERIES.len());
}
