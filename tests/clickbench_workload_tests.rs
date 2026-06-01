use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use datafusion_nexus::{
    CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE, ClickbenchLayout,
    DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR, DEFAULT_LOCAL_CLICKBENCH_SINGLE_DIR,
    DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET, clickbench_query_ids, describe_clickbench_dataset,
    discover_clickbench_layout,
};

#[test]
fn clickbench_query_count_matches_official_43_queries() {
    assert_eq!(CLICKBENCH_QUERY_COUNT, 43);
    assert_eq!(CLICKBENCH_QUERIES.len(), 43);
}

#[test]
fn clickbench_query_ids_are_q01_through_q43_in_order() {
    let expected: Vec<String> = (1..=43).map(|n| format!("q{n:02}")).collect();
    let actual: Vec<&'static str> = clickbench_query_ids().collect();
    assert_eq!(actual, expected.iter().map(String::as_str).collect::<Vec<_>>());
    let from_table: Vec<&'static str> = CLICKBENCH_QUERIES.iter().map(|query| query.id).collect();
    assert_eq!(from_table, expected.iter().map(String::as_str).collect::<Vec<_>>());
}

#[test]
fn clickbench_query_sql_strings_are_non_empty_and_have_no_trailing_semicolon() {
    for query in CLICKBENCH_QUERIES {
        let sql = query.sql;
        assert!(!sql.trim().is_empty(), "{} sql is empty", query.id);
        let trimmed = sql.trim_end();
        assert!(!trimmed.ends_with(';'), "{} sql has a trailing semicolon", query.id);
        assert!(sql.contains("hits"), "{} sql does not reference the hits table", query.id);
    }
}

#[test]
fn clickbench_table_names_are_stable() {
    assert_eq!(CLICKBENCH_TABLE, "hits");
    assert_eq!(CLICKBENCH_RAW_TABLE, "hits_raw");
}

/// Locate an in-repo single-file ClickBench fixture, accepting either the
/// flat `fixture/data/clickbench/hits.parquet` layout or the
/// `fixture/data/clickbench/single/hits.parquet` directory layout. Returns the
/// path to pass to `discover_clickbench_layout` (a file path for the flat
/// layout, a directory path for the `single/` layout) together with the
/// resolved `hits.parquet` file, or `None` when no fixture is present.
fn local_single_fixture() -> Option<(PathBuf, PathBuf)> {
    let flat = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET);
    if flat.is_file() {
        return Some((flat.clone(), flat));
    }
    let dir = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_SINGLE_DIR);
    let nested = dir.join("hits.parquet");
    if nested.is_file() {
        return Some((dir, nested));
    }
    None
}

#[test]
fn discover_clickbench_layout_recognizes_single_file() {
    let Some((root, expected_file)) = local_single_fixture() else {
        eprintln!("skipping: no single-file ClickBench fixture present");
        return;
    };
    let layout = discover_clickbench_layout(&root).expect("single-file layout");
    match layout {
        ClickbenchLayout::Single { file } => assert_eq!(file, expected_file),
        other => panic!("expected single layout, got {other:?}"),
    }
}

#[test]
fn describe_clickbench_dataset_reads_single_metadata_when_present() {
    let Some((root, _)) = local_single_fixture() else {
        eprintln!("skipping: no single-file ClickBench fixture present");
        return;
    };
    let metadata = describe_clickbench_dataset(&root, "test_local_single", "tests/clickbench_workload_tests.rs")
        .expect("describe single");
    assert_eq!(metadata.layout_label(), "single");
    assert_eq!(metadata.file_count, 1);
    assert!(metadata.total_rows > 0);
    assert!(metadata.total_bytes > 0);
    assert_eq!(metadata.row_group_rows.iter().sum::<u64>(), metadata.total_rows);
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.is_none());
}

/// True when `dir` contains at least one canonical `hits_*.parquet` shard.
/// The in-repo `partitioned/` directory may exist but be empty (the large
/// shards are gitignored and only present on machines that downloaded them),
/// so partitioned tests must check for shards, not just directory existence.
fn has_partitioned_shards(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        entry.file_name().to_str().map(|name| name.starts_with("hits_") && name.ends_with(".parquet")).unwrap_or(false)
    })
}

#[test]
fn discover_clickbench_layout_recognizes_partitioned_dir() {
    let dir = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR);
    if !has_partitioned_shards(&dir) {
        eprintln!("skipping: no partitioned shards under {}", dir.display());
        return;
    }
    let layout = discover_clickbench_layout(&dir).expect("partitioned layout");
    match &layout {
        ClickbenchLayout::Partitioned { dir: out_dir, files } => {
            assert_eq!(out_dir, &dir);
            assert!(!files.is_empty(), "partitioned layout has no shards");
            for file in files {
                let name = file.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                assert!(name.starts_with("hits_"), "unexpected shard name {name}");
                assert!(name.ends_with(".parquet"), "unexpected shard suffix {name}");
            }
        },
        other => panic!("expected partitioned layout, got {other:?}"),
    }
}

#[test]
fn discover_clickbench_layout_rejects_missing_root() {
    let missing = Path::new("/tmp/datafusion-nexus-clickbench-missing-root-do-not-create");
    let err = discover_clickbench_layout(missing).expect_err("missing root should error");
    let rendered = err.to_string();
    assert!(rendered.contains("clickbench") || rendered.contains("ClickBench"), "error message: {rendered}");
}

/// Stray `.parquet` files alongside the canonical `hits_*.parquet` shards
/// must never be admitted by discovery. Registration uses the discovered
/// file list, so this is what blocks DataFusion from silently widening the
/// scan to a leftover sample/export and producing rows that do not match
/// the metadata fingerprints.
#[test]
fn discover_clickbench_layout_ignores_stray_parquet_alongside_partitioned_shards() {
    let dir = unique_temp_dir("datafusion-nexus-clickbench-partitioned-stray");
    fs::write(dir.join("hits_0.parquet"), b"").unwrap();
    fs::write(dir.join("hits_1.parquet"), b"").unwrap();
    fs::write(dir.join("hits_extra_sample.parquet"), b"").unwrap();
    fs::write(dir.join("notes.parquet"), b"").unwrap();
    fs::write(dir.join("hits_0.parquet.tmp"), b"").unwrap();

    let layout = discover_clickbench_layout(&dir).expect("partitioned layout");
    let files = match &layout {
        ClickbenchLayout::Partitioned { dir: out_dir, files } => {
            assert_eq!(out_dir, &dir);
            files.clone()
        },
        other => panic!("expected partitioned layout, got {other:?}"),
    };

    let names: Vec<String> =
        files.iter().map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string()).collect();
    assert_eq!(names, vec!["hits_0.parquet", "hits_1.parquet"]);

    fs::remove_dir_all(dir).unwrap();
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let path = std::env::temp_dir().join(format!("{prefix}-{nanos}-{}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn describe_clickbench_dataset_reads_partitioned_metadata_when_present() {
    let dir = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR);
    if !has_partitioned_shards(&dir) {
        eprintln!("skipping: no partitioned shards under {}", dir.display());
        return;
    }
    let metadata = describe_clickbench_dataset(&dir, "test_local_partitioned", "tests/clickbench_workload_tests.rs")
        .expect("describe partitioned");
    assert_eq!(metadata.layout_label(), "partitioned");
    assert!(metadata.file_count > 0);
    assert!(metadata.total_rows > 0);
    assert!(metadata.total_bytes > 0);
    assert_eq!(metadata.row_group_rows.iter().sum::<u64>(), metadata.total_rows);
    assert!(metadata.metadata_fingerprint.starts_with("metadata-fnv1a64:"));
    assert!(metadata.content_fingerprint.is_none());
}
