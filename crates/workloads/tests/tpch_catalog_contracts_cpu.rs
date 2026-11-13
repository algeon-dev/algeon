//! The TPC-H catalog is the SQL files under `src/tpch/queries`, with qgen's
//! substitution values for the dataset's scale factor, which comes from the
//! data and must agree with any declared scale label. A dataset's content
//! fingerprint covers its Parquet footers and generation manifest.

use std::{fs, path::Path};

use workloads::tpch::{
    TPCH_QUERIES, TPCH_QUERY_COUNT, TpchScaleFactor, describe_tpch_dataset,
    describe_tpch_dataset_with_content_fingerprint, read_tpch_scale_factor, write_tiny_tpch_parquet_dataset,
};

fn scale(label: &str) -> TpchScaleFactor {
    TpchScaleFactor::from_label(label).unwrap().expect("label declares a scale")
}

#[test]
fn each_catalog_query_is_the_file_named_by_its_id() {
    assert_eq!(TPCH_QUERIES.len(), TPCH_QUERY_COUNT);
    let queries = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tpch/queries");
    for query in TPCH_QUERIES {
        let file = fs::read_to_string(queries.join(format!("{}.sql", query.id))).expect("catalog query file");
        assert_eq!(query.sql(scale("sf1")), file.replace("[FRACTION]", "0.0001000000"), "{}", query.id);
    }
    let files = fs::read_dir(&queries).expect("queries directory").count();
    assert_eq!(files, TPCH_QUERY_COUNT, "every query file belongs to the catalog");
}

/// qgen substitutes Q11's FRACTION as `0.0001 / SF` with ten decimal places,
/// rounding scales that do not divide it evenly.
#[test]
fn q11_fraction_is_qgen_value_for_the_scale() {
    let q11 = TPCH_QUERIES.iter().find(|query| query.id == "q11").unwrap();
    for (label, fraction) in [
        ("sf1", "0.0001000000"),
        ("sf100", "0.0000010000"),
        ("sf300", "0.0000003333"),
        ("sf500", "0.0000002000"),
        ("sf0.01", "0.0100000000"),
    ] {
        let sql = q11.sql(scale(label));
        assert!(sql.contains(&format!("SUM(ps_supplycost * ps_availqty) * {fraction}\n")), "{label}: {sql}");
    }
}

#[test]
fn declared_scale_that_disagrees_with_the_data_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_tpch_parquet_dataset(dir.path()).unwrap();
    assert_eq!(read_tpch_scale_factor(dir.path()).unwrap(), scale("sf0.0003"));
    assert_eq!(scale("tiny"), scale("sf0.0003"));

    for label in ["tiny", "sf0.0003", "unknown"] {
        let metadata = describe_tpch_dataset(dir.path(), label, "test", "test").unwrap();
        assert_eq!(metadata.scale().unwrap(), scale("sf0.0003"), "{label}");
    }
    let error = describe_tpch_dataset(dir.path(), "sf1", "test", "test").unwrap_err();
    assert!(error.to_string().contains("declared TPC-H scale factor sf1 disagrees with the data"), "{error}");
    assert!(TpchScaleFactor::from_label("sf1x").is_err());
    assert!(TpchScaleFactor::from_label("sf0.00001").is_err());
}

fn content_fingerprint(root: &Path) -> String {
    describe_tpch_dataset_with_content_fingerprint(root, "tiny", "test", "test", true)
        .unwrap()
        .content_fingerprint
        .expect("content fingerprint was requested")
}

/// Byte range of the serialized footer, from the length in the file's trailer.
fn footer_range(bytes: &[u8]) -> std::ops::Range<usize> {
    let trailer = bytes.len() - 8;
    let footer_len = u32::from_le_bytes(bytes[trailer..trailer + 4].try_into().unwrap()) as usize;
    trailer - footer_len..trailer
}

#[test]
fn content_fingerprint_hashes_footers_and_manifest_but_not_data_pages() {
    let dir = tempfile::tempdir().unwrap();
    write_tiny_tpch_parquet_dataset(dir.path()).unwrap();
    let lineitem = dir.path().join("lineitem.parquet");
    let original = content_fingerprint(dir.path());

    let mut bytes = fs::read(&lineitem).unwrap();
    let footer = footer_range(&bytes);
    let data_byte = 4 + (footer.start - 4) / 2;
    bytes[data_byte] ^= 0xff;
    fs::write(&lineitem, &bytes).unwrap();
    assert_eq!(content_fingerprint(dir.path()), original, "a byte outside the footer must not change the value");

    let created_by = bytes[footer.clone()].windows(10).position(|window| window == b"parquet-rs").unwrap();
    bytes[footer.start + created_by] = b'q';
    fs::write(&lineitem, &bytes).unwrap();
    let footer_changed = content_fingerprint(dir.path());
    assert_ne!(footer_changed, original, "a footer byte must change the value");

    let manifest = dir.path().join("_generation.json");
    fs::write(&manifest, r#"{"writer": "spark"}"#).unwrap();
    let with_manifest = content_fingerprint(dir.path());
    assert_ne!(with_manifest, footer_changed, "adding the manifest must change the value");
    fs::write(&manifest, r#"{"writer": "duckdb"}"#).unwrap();
    assert_ne!(content_fingerprint(dir.path()), with_manifest, "editing the manifest must change the value");
}
