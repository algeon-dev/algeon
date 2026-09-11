use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, Float64Array, Int32Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::SessionContext;
use parquet::{
    arrow::ArrowWriter,
    file::{metadata::KeyValue, properties::WriterProperties},
};
use serde_json::json;
use workloads::graph::{
    DBLP_DATASET_ID, DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY, DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY,
    DBLP_TABLES, describe_canonical_dblp_dataset, describe_dblp_dataset, discover_dblp_fixture_layout,
    register_dblp_tables,
};

#[test]
fn generic_dblp_discovery_validates_parquet_and_projects_serde_metadata() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, None);

    let layout = discover_dblp_fixture_layout(&root.join("parquet")).unwrap();
    assert_eq!(layout.dataset_root, root);
    assert_eq!(layout.parquet_root, root.join("parquet"));

    let metadata = describe_dblp_dataset(
        &root,
        "sample",
        "fixture/fixture.sh graph dblp-ingest --source-json /data/dblp.v12.json --out-root /data/demo",
    )
    .unwrap();
    assert_eq!(metadata.table_count, DBLP_TABLES.len());
    assert_eq!(metadata.file_count, DBLP_TABLES.len());
    assert_eq!(metadata.tables.iter().map(|table| table.table.as_str()).collect::<Vec<_>>(), DBLP_TABLES);
    assert_eq!(metadata.total_rows, 16);
    assert_eq!(metadata.source_content_sha256, "a".repeat(64));

    let projected = serde_json::to_value(metadata.to_neutral_metadata()).unwrap();
    assert_eq!(projected["workload"], "dblp_aminer_citation_network");
    assert_eq!(projected["scale_factor"], "sample");
    assert!(projected["metadata_fingerprint"].as_str().unwrap().starts_with("manifest-sha256:"));
    assert!(projected["content_fingerprint"].as_str().unwrap().starts_with("source-sha256:"));
}

#[test]
fn generic_dblp_discovery_rejects_stale_parquet_metadata_and_missing_source_fingerprint() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, None);

    let manifest = root.join("manifest.json");
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["bytes"] = json!(999_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("byte size"), "{error}");

    write_dblp_fixture(&root, None);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["num_rows"] = json!(999_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet row count"), "{error}");

    write_dblp_fixture(&root, None);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["num_row_groups"] = json!(2_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet row-group count"), "{error}");

    write_dblp_fixture(&root, None);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["schema"] = json!(["not_value"]);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet schema"), "{error}");

    write_dblp_fixture(&root, None);
    let source = root.join("source.json");
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    document["source_json"]["sha256"] = serde_json::Value::Null;
    fs::write(&source, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("SHA-256"), "{error}");
}

#[test]
fn generic_dblp_discovery_preserves_multi_graph_slice_identity_and_requires_matching_source_metadata() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, None);

    let slice = json!({
        "schema_version": 1,
        "partition": "source_hash_splitmix64_v1",
        "slice_count": 8,
        "slice_index": 3,
        "hash_seed": 20260804_u64,
        "edge_rowset_sha256": "b".repeat(64)
    });
    let manifest_path = root.join("manifest.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["multi_graph_slice"] = slice.clone();
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    let source_path = root.join("source.json");
    let mut source: serde_json::Value = serde_json::from_slice(&fs::read(&source_path).unwrap()).unwrap();
    source["derived_fixture"] = slice;
    fs::write(&source_path, serde_json::to_vec_pretty(&source).unwrap()).unwrap();

    let metadata = describe_dblp_dataset(&root, "v12_slice_03", "prepared").unwrap();
    let identity = metadata.multi_graph_slice.expect("slice fixture must expose its rowset identity");
    assert_eq!(identity.partition, "source_hash_splitmix64_v1");
    assert_eq!(identity.slice_count, 8);
    assert_eq!(identity.slice_index, 3);
    assert_eq!(identity.hash_seed, 20_260_804);
    assert_eq!(identity.edge_rowset_sha256, "b".repeat(64));

    let mut source: serde_json::Value = serde_json::from_slice(&fs::read(&source_path).unwrap()).unwrap();
    source["derived_fixture"]["slice_index"] = json!(4);
    fs::write(&source_path, serde_json::to_vec_pretty(&source).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "v12_slice_03", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match manifest multi_graph_slice"), "{error}");
}

#[test]
fn canonical_dblp_discovery_rejects_a_self_consistent_sample_fixture() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, Some("3"));

    let error = describe_canonical_dblp_dataset(&root, "v12", "prepared").unwrap_err();
    assert!(error.to_string().contains("full canonical DBLP V12"), "{error}");
}

#[test]
fn canonical_dblp_discovery_requires_edge_footer_vertex_bounds_before_using_the_fixture() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, None);

    let error = describe_canonical_dblp_dataset(&root, "v12", "prepared").unwrap_err();
    assert!(error.to_string().contains("missing required footer key algeon.graph.vertex_upper_bound.v1"), "{error}");
}

#[test]
fn generic_dblp_discovery_requires_complete_simple_directed_edge_footer() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");

    write_dblp_fixture_with_edge_topology(&root, None, None);
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(
        error.to_string().contains("missing required footer key algeon.graph.simple_directed_edge_topology.v1"),
        "{error}"
    );

    write_dblp_fixture_with_edge_topology(
        &root,
        None,
        Some(r#"{"source_column_index":0,"destination_column_index":2}"#),
    );
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("must bind src/dst physical columns (1, 2)"), "{error}");
}

#[tokio::test]
async fn dblp_registration_binds_every_required_table_under_its_fixture_name() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root, None);

    let ctx = SessionContext::new();
    let layout = register_dblp_tables(&ctx, &root).await.unwrap();
    assert_eq!(layout.parquet_root, root.join("parquet"));
    for table in DBLP_TABLES {
        let rows = ctx.sql(&format!("SELECT COUNT(*) FROM {table}")).await.unwrap().collect().await.unwrap();
        assert_eq!(rows.len(), 1, "{table}");
    }

    let missing = temporary.path().join("absent");
    assert!(register_dblp_tables(&SessionContext::new(), &missing).await.is_err());
}

fn write_dblp_fixture(root: &Path, graph_vertex_upper_bound: Option<&str>) {
    write_dblp_fixture_with_edge_topology(
        root,
        graph_vertex_upper_bound,
        Some(r#"{"source_column_index":1,"destination_column_index":2}"#),
    );
}

fn write_dblp_fixture_with_edge_topology(
    root: &Path,
    graph_vertex_upper_bound: Option<&str>,
    edge_topology: Option<&str>,
) {
    let parquet = root.join("parquet");
    fs::create_dir_all(&parquet).unwrap();
    let rows = |table: &str| -> usize {
        match table {
            "edges_by_src" | "edges_by_dst" => 2,
            "profiles_by_paper_id" => 3,
            "paper_authors" => 4,
            "paper_fos" => 5,
            _ => unreachable!("fixed DBLP table contract"),
        }
    };
    for &table in DBLP_TABLES {
        let footer = matches!(table, "edges_by_src" | "edges_by_dst").then_some(graph_vertex_upper_bound).flatten();
        let topology = matches!(table, "edges_by_src" | "edges_by_dst").then_some(edge_topology).flatten();
        write_tiny_parquet(&parquet.join(format!("{table}.parquet")), table, rows(table), footer, topology);
    }
    let entry = |table: &str| {
        json!({
            "bytes": fs::metadata(parquet.join(format!("{table}.parquet"))).unwrap().len(),
            "num_rows": rows(table),
            "num_row_groups": 1,
            "schema": tiny_table_schema_names(table)
        })
    };
    let manifest = json!({
        "schema_version": 1,
        "generator": "fixture/graph/dblp_ingest.py",
        "dataset": {"dataset_id": DBLP_DATASET_ID, "source": "test source"},
        "max_papers": 3,
        "counts": {"papers": 3, "edges": 2, "author_rows": 4, "fos_rows": 5},
        "edge_layouts": {
            "src": {"parquet": entry("edges_by_src")},
            "dst": {"parquet": entry("edges_by_dst")}
        },
        "profiles": entry("profiles_by_paper_id"),
        "paper_authors": entry("paper_authors"),
        "paper_fos": entry("paper_fos")
    });
    let source = json!({
        "schema_version": 1,
        "dataset_id": DBLP_DATASET_ID,
        "source": "test source",
        "source_json": {"path": "/data/dblp.v12.json", "bytes": 1, "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
    });
    fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    fs::write(root.join("source.json"), serde_json::to_vec_pretty(&source).unwrap()).unwrap();
}

fn tiny_table_schema_names(table: &str) -> Vec<&'static str> {
    if matches!(table, "edges_by_src" | "edges_by_dst") {
        vec!["edge_id", "src", "dst", "weight", "src_bucket", "dst_bucket"]
    } else {
        vec!["value"]
    }
}

fn write_tiny_parquet(
    path: &Path,
    table: &str,
    rows: usize,
    graph_vertex_upper_bound: Option<&str>,
    edge_topology: Option<&str>,
) {
    let is_edge_table = matches!(table, "edges_by_src" | "edges_by_dst");
    let schema = Arc::new(Schema::new(if is_edge_table {
        vec![
            Field::new("edge_id", DataType::Int64, false),
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("weight", DataType::Float64, false),
            Field::new("src_bucket", DataType::Int32, false),
            Field::new("dst_bucket", DataType::Int32, false),
        ]
    } else {
        vec![Field::new("value", DataType::Int64, false)]
    }));
    let values = (0..i64::try_from(rows).unwrap()).collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = if is_edge_table {
        vec![
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Int64Array::from(values.clone())),
            Arc::new(Float64Array::from(vec![1.0; rows])),
            Arc::new(Int32Array::from(vec![0; rows])),
            Arc::new(Int32Array::from(vec![0; rows])),
        ]
    } else {
        vec![Arc::new(Int64Array::from(values))]
    };
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    let mut footer_metadata = Vec::new();
    if let Some(value) = graph_vertex_upper_bound {
        footer_metadata.push(KeyValue::new(DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY.to_string(), value.to_string()));
    }
    if let Some(value) = edge_topology {
        footer_metadata
            .push(KeyValue::new(DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY.to_string(), value.to_string()));
    }
    let properties = WriterProperties::builder()
        .set_key_value_metadata((!footer_metadata.is_empty()).then_some(footer_metadata))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}
