use datafusion_nexus::table_format::iceberg::{
    IcebergTableFormatPlanner, NexusIcebergTableProvider, ResolvedTable, SourceAccessPolicy, TableIdentifier,
};

#[test]
fn public_iceberg_adaptation_requires_a_resolved_table() {
    let policy = SourceAccessPolicy::s3_default_chain(Some("us-east-1".to_string()));
    assert_eq!(policy.s3_access().region, Some("us-east-1"));

    let identifier = TableIdentifier::new(vec!["analytics".to_string()], "orders").unwrap();
    assert_eq!(identifier.display_name(), "analytics.orders");

    let _provider_constructor: fn(ResolvedTable) -> _ = NexusIcebergTableProvider::try_new_resolved;
    let _planner_constructor: fn(ResolvedTable) -> _ = IcebergTableFormatPlanner::new;
}
