//! The TPC-H catalog is the SQL files under `src/tpch/queries`, which Flight
//! stress profiles reference by query id.

use std::{fs, path::Path};

use workloads::tpch::{TPCH_QUERIES, TPCH_QUERY_COUNT};

#[test]
fn each_catalog_query_is_the_file_named_by_its_id() {
    assert_eq!(TPCH_QUERIES.len(), TPCH_QUERY_COUNT);
    let queries = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tpch/queries");
    for query in TPCH_QUERIES {
        let file = fs::read_to_string(queries.join(format!("{}.sql", query.id))).expect("catalog query file");
        assert_eq!(query.sql, file, "{}", query.id);
    }
    let files = fs::read_dir(&queries).expect("queries directory").count();
    assert_eq!(files, TPCH_QUERY_COUNT, "every query file belongs to the catalog");
}
