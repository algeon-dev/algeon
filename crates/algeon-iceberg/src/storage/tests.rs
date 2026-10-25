use std::collections::HashMap;

use algeon_datafusion::native_exec::S3SourceStorageProfile;

use super::{file_io_properties, mismatched_property};

fn properties(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(key, value)| (key.to_string(), value.to_string())).collect()
}

#[test]
fn the_storage_profile_becomes_matching_file_io_properties() {
    let storage = S3SourceStorageProfile::default_chain()
        .with_endpoint("http://store:9000")
        .with_region("eu-west-2")
        .with_path_style_access(true)
        .with_unsigned_requests();
    let derived = file_io_properties(&storage);
    assert_eq!(
        derived,
        properties(&[
            ("s3.endpoint", "http://store:9000"),
            ("s3.region", "eu-west-2"),
            ("s3.path-style-access", "true"),
            ("s3.allow-anonymous", "true"),
        ])
    );
    assert_eq!(mismatched_property(&derived, &storage), None);
}

#[test]
fn the_endpoint_must_match_while_region_and_addressing_are_compared_when_both_are_set() {
    let storage = S3SourceStorageProfile::default_chain().with_region("eu-west-2");
    assert_eq!(
        mismatched_property(&properties(&[("s3.endpoint", "http://other:9000")]), &storage),
        Some("s3.endpoint")
    );
    assert_eq!(mismatched_property(&properties(&[("client.region", "us-east-1")]), &storage), Some("client.region"));
    assert_eq!(
        mismatched_property(&properties(&[("client.region", "eu-west-2"), ("s3.region", "us-east-1")]), &storage),
        None
    );
    assert_eq!(mismatched_property(&properties(&[("s3.path-style-access", "true")]), &storage), None);

    let path_style =
        S3SourceStorageProfile::default_chain().with_endpoint("http://store:9000/").with_path_style_access(true);
    assert_eq!(mismatched_property(&properties(&[("s3.endpoint", "http://store:9000")]), &path_style), None);
    assert_eq!(
        mismatched_property(
            &properties(&[("s3.endpoint", "http://store:9000"), ("s3.path-style-access", "false")]),
            &path_style
        ),
        Some("s3.path-style-access")
    );
    assert_eq!(mismatched_property(&HashMap::new(), &path_style), Some("s3.endpoint"));
}
