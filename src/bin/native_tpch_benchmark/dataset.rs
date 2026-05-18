use super::*;

pub(super) async fn prepare_benchmark_dataset(
    config: &NativeTpchBenchmarkConfig,
    tiny_root: &Path,
) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    match config.dataset_mode {
        NativeTpchDatasetMode::External(scale_factor) => prepare_external_dataset(scale_factor),
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

const SF10_TABLE_ROW_COUNTS: &[(&str, u64)] = &[
    ("customer", 1_500_000),
    ("lineitem", 59_986_052),
    ("nation", 25),
    ("orders", 15_000_000),
    ("part", 2_000_000),
    ("partsupp", 8_000_000),
    ("region", 5),
    ("supplier", 100_000),
];

const SF100_TABLE_ROW_COUNTS: &[(&str, u64)] = &[
    ("customer", 15_000_000),
    ("lineitem", 600_037_902),
    ("nation", 25),
    ("orders", 150_000_000),
    ("part", 20_000_000),
    ("partsupp", 80_000_000),
    ("region", 5),
    ("supplier", 1_000_000),
];

const REQUIRED_COLUMNS: &[(&str, &[&str])] = &[
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

fn prepare_external_dataset(
    expected_scale_factor: NativeTpchScaleFactor,
) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    let dataset_mode = expected_scale_factor.dataset_mode_str();
    let (root, metadata) = if let Some(root) = std::env::var_os(CUDF_TPCH_PARQUET_DIR_ENV) {
        let root = PathBuf::from(root);
        let scale_factor = std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
        let generation_command = std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).map_err(|_| {
            plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} requires {CUDF_TPCH_GENERATION_COMMAND_ENV} so {} provenance is reproducible",
                expected_scale_factor.label()
            ))
        })?;
        let metadata = describe_tpch_dataset_with_content_fingerprint(
            &root,
            scale_factor,
            "external_parquet_root",
            generation_command,
            true,
        )
        .map_err(tpch_error)?;
        (root, metadata)
    } else {
        let default_root = default_root_for(expected_scale_factor);
        let Some((root, metadata)) =
            default_nvme_tpch_dataset_with_fingerprint(default_root, expected_scale_factor.as_str())
        else {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} requires {CUDF_TPCH_PARQUET_DIR_ENV} or a populated default fixture at {default_root}; set CUDF_TPCH_BENCH_DATASET_MODE=allow_tiny for local tiny/dev runs"
            )));
        };
        (root, metadata)
    };
    validate_external_metadata(&metadata, expected_scale_factor)?;
    Ok((root, metadata))
}

fn default_root_for(scale_factor: NativeTpchScaleFactor) -> &'static str {
    datafusion_cudf::default_tpch_parquet_root(match scale_factor {
        NativeTpchScaleFactor::Sf1 => datafusion_cudf::TpchDefaultScale::Sf1,
        NativeTpchScaleFactor::Sf10 => datafusion_cudf::TpchDefaultScale::Sf10,
        NativeTpchScaleFactor::Sf100 => datafusion_cudf::TpchDefaultScale::Sf100Optimized,
    })
}

#[cfg(test)]
pub(super) fn validate_sf1_external_metadata(metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    validate_external_metadata(metadata, NativeTpchScaleFactor::Sf1)
}

#[cfg(test)]
pub(super) fn validate_sf10_external_metadata(metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    validate_external_metadata(metadata, NativeTpchScaleFactor::Sf10)
}

pub(super) fn validate_external_metadata(
    metadata: &TpchDatasetMetadata,
    expected_scale_factor: NativeTpchScaleFactor,
) -> DataFusionResult<()> {
    let dataset_mode = expected_scale_factor.dataset_mode_str();
    if metadata.scale_factor != expected_scale_factor.as_str() {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires {CUDF_TPCH_SCALE_FACTOR_ENV}={}; observed {}",
            expected_scale_factor.as_str(),
            metadata.scale_factor
        )));
    }
    if metadata.provenance != "external_parquet_root" {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires an external Parquet root; observed provenance {}",
            metadata.provenance
        )));
    }
    if metadata.generation_command.trim().is_empty() || metadata.generation_command == "unknown_external" {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires a reproducible {CUDF_TPCH_GENERATION_COMMAND_ENV}; observed {}",
            metadata.generation_command
        )));
    }
    if metadata.content_fingerprint.is_none() {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires a content fingerprint"
        )));
    }
    if metadata.table_count != TPCH_TABLES.len() || metadata.tables.len() != TPCH_TABLES.len() {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires {} TPC-H tables; observed metadata table_count={} tables_len={}",
            TPCH_TABLES.len(),
            metadata.table_count,
            metadata.tables.len()
        )));
    }
    if metadata.file_count == 0 || metadata.total_bytes == 0 {
        return Err(plan_error(format!(
            "native TPC-H benchmark dataset_mode={dataset_mode} requires non-empty Parquet files"
        )));
    }

    let expected_row_counts =
        expected_table_row_counts(expected_scale_factor).iter().copied().collect::<BTreeMap<_, _>>();
    let required_columns = REQUIRED_COLUMNS.iter().copied().collect::<BTreeMap<_, _>>();
    for &expected_table in TPCH_TABLES {
        let Some(table) = metadata.tables.iter().find(|table| table.table == expected_table) else {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} is missing table {expected_table}"
            )));
        };
        let expected_rows = expected_row_counts[expected_table];
        if table.row_count != expected_rows {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} requires {} row count {expected_table}={expected_rows}; observed {}",
                expected_scale_factor.label(),
                table.row_count
            )));
        }
        if table.file_count == 0 || table.byte_size == 0 || table.row_group_count == 0 {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} requires non-empty Parquet metadata for table {expected_table}"
            )));
        }
        if table.row_group_rows.len() != table.row_group_count {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has inconsistent row_group_rows length for table {expected_table}; row_group_count={} row_group_rows_len={}",
                table.row_group_count,
                table.row_group_rows.len()
            )));
        }
        if table.row_group_bytes.len() != table.row_group_count {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has inconsistent row_group_bytes length for table {expected_table}; row_group_count={} row_group_bytes_len={}",
                table.row_group_count,
                table.row_group_bytes.len()
            )));
        }
        if table.row_group_rows.iter().sum::<u64>() != table.row_count {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has inconsistent row-group rows for table {expected_table}"
            )));
        }
        if table.row_group_rows.contains(&0) {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has zero-row row groups for table {expected_table}"
            )));
        }
        if table.row_group_bytes.contains(&0) {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has zero-byte row groups for table {expected_table}"
            )));
        }
        let row_group_bytes = table.row_group_bytes.iter().sum::<u64>();
        if row_group_bytes > table.byte_size {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} has row-group bytes greater than table bytes for table {expected_table}; row_group_bytes={row_group_bytes} table_bytes={}",
                table.byte_size
            )));
        }
        for column in required_columns[expected_table] {
            if !table.schema.contains(&format!("{column}:")) {
                return Err(plan_error(format!(
                    "native TPC-H benchmark dataset_mode={dataset_mode} table {expected_table} is missing required column {column}"
                )));
            }
        }
    }
    Ok(())
}

fn expected_table_row_counts(scale_factor: NativeTpchScaleFactor) -> &'static [(&'static str, u64)] {
    match scale_factor {
        NativeTpchScaleFactor::Sf1 => SF1_TABLE_ROW_COUNTS,
        NativeTpchScaleFactor::Sf10 => SF10_TABLE_ROW_COUNTS,
        NativeTpchScaleFactor::Sf100 => SF100_TABLE_ROW_COUNTS,
    }
}

pub(super) async fn register_tpch_tables(ctx: &SessionContext, metadata: &TpchDatasetMetadata) -> DataFusionResult<()> {
    for table in &metadata.tables {
        let layout = TpchTableLayout::from_metadata(table);
        register_tpch_table_layout(ctx, &table.table, &layout).await.map_err(tpch_error)?;
    }
    Ok(())
}
