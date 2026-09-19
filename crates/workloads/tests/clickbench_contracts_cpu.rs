use std::{fs::File, path::Path};

use arrow::{
    array::Int32Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use workloads::clickbench::{
    CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE, ClickbenchLayout,
    clickbench_query_ids, describe_clickbench_dataset, discover_clickbench_layout,
};

#[test]
fn clickbench_query_catalog_is_stable() {
    assert_eq!(CLICKBENCH_QUERY_COUNT, 43);
    assert_eq!(CLICKBENCH_QUERIES.len(), 43);
    let expected: Vec<String> = (1..=43).map(|number| format!("q{number:02}")).collect();
    assert_eq!(clickbench_query_ids().collect::<Vec<_>>(), expected.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(
        CLICKBENCH_QUERIES.iter().map(|query| query.id).collect::<Vec<_>>(),
        expected.iter().map(String::as_str).collect::<Vec<_>>()
    );
    for query in CLICKBENCH_QUERIES {
        assert!(!query.sql.trim().is_empty(), "{} SQL is empty", query.id);
        assert!(!query.sql.trim_end().ends_with(';'), "{} SQL has a trailing semicolon", query.id);
        assert!(query.sql.contains("hits"), "{} SQL does not reference hits", query.id);
    }
    assert_eq!(CLICKBENCH_TABLE, "hits");
    assert_eq!(CLICKBENCH_RAW_TABLE, "hits_raw");
}

#[test]
fn discover_and_describe_single_fixture() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("hits.parquet");
    write_tiny_parquet(&file, &[1, 2, 3]);

    assert_eq!(discover_clickbench_layout(temp.path()).unwrap(), ClickbenchLayout::Single { file: file.clone() });
    let metadata = describe_clickbench_dataset(temp.path(), "test", "generated tiny fixture").unwrap();
    assert_eq!(metadata.layout_label(), "single");
    assert_eq!(metadata.file_count, 1);
    assert_eq!(metadata.total_rows, 3);
    assert!(metadata.total_bytes > 0);
    assert_eq!(metadata.row_group_rows.iter().sum::<u64>(), metadata.total_rows);
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert_eq!(metadata.content_fingerprint, None);
}

#[test]
fn discover_and_describe_partitioned_fixture() {
    let temp = tempfile::tempdir().unwrap();
    write_tiny_parquet(&temp.path().join("hits_10.parquet"), &[10]);
    write_tiny_parquet(&temp.path().join("hits_2.parquet"), &[2, 3]);
    std::fs::write(temp.path().join("notes.parquet"), b"not a shard").unwrap();

    let layout = discover_clickbench_layout(temp.path()).unwrap();
    let ClickbenchLayout::Partitioned { dir, files } = &layout else {
        panic!("expected partitioned layout, got {layout:?}");
    };
    assert_eq!(dir, temp.path());
    assert_eq!(
        files.iter().map(|file| file.file_name().unwrap().to_str().unwrap()).collect::<Vec<_>>(),
        vec!["hits_2.parquet", "hits_10.parquet"]
    );
    let metadata = describe_clickbench_dataset(temp.path(), "test", "generated tiny fixture").unwrap();
    assert_eq!(metadata.layout_label(), "partitioned");
    assert_eq!(metadata.file_count, 2);
    assert_eq!(metadata.total_rows, 3);
    assert!(metadata.total_bytes > 0);
}

#[test]
fn discover_clickbench_layout_rejects_missing_root() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing");
    let error = discover_clickbench_layout(&missing).expect_err("missing root must fail");
    assert!(error.to_string().contains("ClickBench"));
}

fn write_tiny_parquet(path: &Path, values: &[i32]) {
    let schema = std::sync::Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(Int32Array::from(values.to_vec()))]).unwrap();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}
