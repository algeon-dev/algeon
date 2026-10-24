use datafusion::execution::object_store::ObjectStoreUrl;

use super::split_location;

#[test]
fn s3_family_locations_share_the_s3_bucket_url_and_other_schemes_are_refused() {
    for scheme in ["s3", "s3a", "s3n"] {
        let (url, path) = split_location(&format!("{scheme}://bucket/warehouse/a.parquet")).unwrap();
        assert_eq!(url, ObjectStoreUrl::parse("s3://bucket").unwrap());
        assert_eq!(path.as_ref(), "warehouse/a.parquet");
    }
    let (url, path) = split_location("file:///tmp/warehouse/a.parquet").unwrap();
    assert_eq!(url, ObjectStoreUrl::local_filesystem());
    assert_eq!(path.as_ref(), "tmp/warehouse/a.parquet");
    for location in ["gs://bucket/a.parquet", "s3:///a.parquet", "/tmp/a.parquet", "abfss://c@a/x.parquet"] {
        assert!(split_location(location).is_none(), "{location}");
    }
}
