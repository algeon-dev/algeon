use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
};

use arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::prelude::SessionContext;
use nexus_workloads::graph::{
    DBLP_DATASET_ID, DBLP_TABLES, describe_canonical_dblp_dataset, describe_dblp_dataset, discover_dblp_fixture_layout,
    register_dblp_tables,
};
use parquet::arrow::ArrowWriter;
use serde_json::json;

#[test]
fn generic_dblp_discovery_validates_parquet_and_projects_serde_metadata() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root);

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
    write_dblp_fixture(&root);

    let manifest = root.join("manifest.json");
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["bytes"] = json!(999_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("byte size"), "{error}");

    write_dblp_fixture(&root);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["num_rows"] = json!(999_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet row count"), "{error}");

    write_dblp_fixture(&root);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["num_row_groups"] = json!(2_u64);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet row-group count"), "{error}");

    write_dblp_fixture(&root);
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    document["profiles"]["schema"] = json!(["not_value"]);
    fs::write(&manifest, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("does not match Parquet schema"), "{error}");

    write_dblp_fixture(&root);
    let source = root.join("source.json");
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    document["source_json"]["sha256"] = serde_json::Value::Null;
    fs::write(&source, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let error = describe_dblp_dataset(&root, "sample", "prepared").unwrap_err();
    assert!(error.to_string().contains("SHA-256"), "{error}");
}

#[test]
fn canonical_dblp_discovery_rejects_a_self_consistent_sample_fixture() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root);

    let error = describe_canonical_dblp_dataset(&root, "v12", "prepared").unwrap_err();
    assert!(error.to_string().contains("full canonical DBLP V12"), "{error}");
}

#[tokio::test]
async fn dblp_registration_binds_every_required_table_under_its_fixture_name() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("citation_network").join("v1");
    write_dblp_fixture(&root);

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

fn write_dblp_fixture(root: &Path) {
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
        write_tiny_parquet(&parquet.join(format!("{table}.parquet")), rows(table));
    }
    let entry = |table: &str| {
        json!({
            "bytes": fs::metadata(parquet.join(format!("{table}.parquet"))).unwrap().len(),
            "num_rows": rows(table),
            "num_row_groups": 1,
            "schema": ["value"]
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

fn write_tiny_parquet(path: &Path, rows: usize) {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
    let values = (0..i64::try_from(rows).unwrap()).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(values))]).unwrap();
    let mut writer = ArrowWriter::try_new(File::create(path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}
