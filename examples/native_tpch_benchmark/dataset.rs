use super::*;

pub(super) async fn prepare_benchmark_dataset(
    config: &NativeTpchBenchmarkConfig,
    tiny_root: &Path,
) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    match config.dataset_mode {
        NativeTpchDatasetMode::Sf1External => prepare_sf1_external_dataset(),
        NativeTpchDatasetMode::AllowTiny => prepare_tpch_report_dataset(tiny_root).map_err(tpch_error),
    }
}

const SF1_TABLE_ROW_COUNTS: &[(&str, u64)] = &[
    ("customer", 150_000),
    ("lineitem", 6_001_215),
    ("nation", 25),
    ("orders", 1_500_000),
    ("part", 200_000),
    ("partsupp", 800_000),
    ("region", 5),
    ("supplier", 10_000),
];

const SF1_REQUIRED_COLUMNS: &[(&str, &[&str])] = &[
    (
        "customer",
        &["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"],
    ),
    (
        "lineitem",
        &[
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        ],
    ),
    ("nation", &["n_nationkey", "n_name", "n_regionkey", "n_comment"]),
    (
        "orders",
        &[
            "o_orderkey",
            "o_custkey",
            "o_orderstatus",
            "o_totalprice",
            "o_orderdate",
            "o_orderpriority",
            "o_clerk",
            "o_shippriority",
            "o_comment",
        ],
    ),
    (
        "part",
        &["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type", "p_size", "p_container", "p_retailprice", "p_comment"],
    ),
    ("partsupp", &["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"]),
    ("region", &["r_regionkey", "r_name", "r_comment"]),
    ("supplier", &["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]),
];

fn prepare_sf1_external_dataset() -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    let Some(root) = std::env::var_os(CUDF_TPCH_PARQUET_DIR_ENV) else {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode=sf1_external requires {CUDF_TPCH_PARQUET_DIR_ENV}; set CUDF_TPCH_BENCH_DATASET_MODE=allow_tiny for local tiny/dev runs"
        )));
    };

    let root = PathBuf::from(root);
    let scale_factor = std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
    let generation_command =
        std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
    let metadata = describe_tpch_dataset_with_content_fingerprint(
        &root,
        scale_factor,
        "external_parquet_root",
        generation_command,
        true,
    )
    .map_err(tpch_error)?;
    validate_sf1_external_metadata(&metadata)?;
    Ok((root, metadata))
}

pub(super) fn validate_sf1_external_metadata(metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    if metadata.scale_factor != "sf1" {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode=sf1_external requires {CUDF_TPCH_SCALE_FACTOR_ENV}=sf1; observed {}",
            metadata.scale_factor
        )));
    }
    if metadata.provenance != "external_parquet_root" {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode=sf1_external requires an external Parquet root; observed provenance {}",
            metadata.provenance
        )));
    }
    if metadata.content_fingerprint.is_none() {
        return Err(plan_error("native TPC-H benchmark dataset_mode=sf1_external requires a content fingerprint"));
    }
    if metadata.table_count != TPCH_TABLES.len() || metadata.tables.len() != TPCH_TABLES.len() {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode=sf1_external requires {} TPC-H tables; observed metadata table_count={} tables_len={}",
            TPCH_TABLES.len(),
            metadata.table_count,
            metadata.tables.len()
        )));
    }
    if metadata.file_count == 0 || metadata.total_bytes == 0 {
        return Err(plan_error("native TPC-H benchmark dataset_mode=sf1_external requires non-empty Parquet files"));
    }

    let expected_row_counts = SF1_TABLE_ROW_COUNTS.iter().copied().collect::<BTreeMap<_, _>>();
    let required_columns = SF1_REQUIRED_COLUMNS.iter().copied().collect::<BTreeMap<_, _>>();
    for &expected_table in TPCH_TABLES {
        let Some(table) = metadata.tables.iter().find(|table| table.table == expected_table) else {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode=sf1_external is missing table {expected_table}"
            )));
        };
        let expected_rows = expected_row_counts[expected_table];
        if table.row_count != expected_rows {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode=sf1_external requires SF1 row count {expected_table}={expected_rows}; observed {}",
                table.row_count
            )));
        }
        if table.file_count == 0 || table.byte_size == 0 || table.row_group_count == 0 {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode=sf1_external requires non-empty Parquet metadata for table {expected_table}"
            )));
        }
        if table.row_group_rows.iter().sum::<u64>() != table.row_count {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode=sf1_external has inconsistent row-group rows for table {expected_table}"
            )));
        }
        for column in required_columns[expected_table] {
            if !table.schema.contains(&format!("{column}:")) {
                return Err(plan_error(format!(
                    "native TPC-H benchmark dataset_mode=sf1_external table {expected_table} is missing required column {column}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) async fn register_tpch_tables(ctx: &SessionContext, metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    for table in &metadata.tables {
        let path = table
            .path
            .to_str()
            .ok_or_else(|| plan_error(format!("TPC-H table path is not valid UTF-8: {}", table.path.display())))?;
        ctx.register_parquet(&table.table, path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}
