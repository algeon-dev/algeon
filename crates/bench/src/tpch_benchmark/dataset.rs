use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use workloads::tpch::{TpchDefaultScale, TpchScaleFactor, default_tpch_parquet_root};

use super::{
    ALGEON_PARQUET_DIR_ENV, ALGEON_TPCH_GENERATION_COMMAND_ENV, ALGEON_TPCH_SCALE_FACTOR_ENV, DataFusionResult,
    NativeTpchBenchmarkConfig, NativeTpchDatasetMode, NativeTpchScaleFactor, SessionContext, TPCH_TABLES,
    TpchDatasetMetadata, TpchTableLayout, default_tpch_dataset_with_fingerprint,
    describe_tpch_dataset_with_content_fingerprint, plan_error, prepare_tpch_report_dataset,
    register_tpch_table_layout, tpch_error,
};

pub(super) async fn prepare_benchmark_dataset(
    config: &NativeTpchBenchmarkConfig,
    tiny_root: &Path,
) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    match config.dataset_mode {
        NativeTpchDatasetMode::External(scale_factor) => prepare_external_dataset(scale_factor),
        NativeTpchDatasetMode::AllowTiny => prepare_tpch_report_dataset(tiny_root).map_err(tpch_error),
    }
}

/// Per-scale table row counts, shared with `fixture/tpc-h/generate.py` so the
/// generator's verification and this validation accept the same data.
const EXPECTED_ROWS_JSON: &str = include_str!("../../../../fixture/tpc-h/expected_rows.json");

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

/// Where an external benchmark dataset is read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ExternalDatasetRoot {
    /// `ALGEON_PARQUET_DIR`, described with its declared scale and generation command.
    Explicit(PathBuf),
    /// The scale's in-repo default fixture root.
    Default(&'static str),
}

/// Resolves the dataset root for `scale_factor` from the `ALGEON_PARQUET_DIR`
/// value. A scale without an in-repo default root requires that variable.
pub(super) fn resolve_external_dataset_root(
    scale_factor: NativeTpchScaleFactor,
    parquet_dir: Option<PathBuf>,
) -> DataFusionResult<ExternalDatasetRoot> {
    if let Some(root) = parquet_dir {
        return Ok(ExternalDatasetRoot::Explicit(root));
    }
    default_root_for(scale_factor).map(ExternalDatasetRoot::Default).ok_or_else(|| {
        plan_error(format!(
            "native TPC-H benchmark dataset_mode={} has no default fixture root; set {ALGEON_PARQUET_DIR_ENV} to the {} Parquet root",
            scale_factor.dataset_mode_str(),
            scale_factor.label()
        ))
    })
}

fn prepare_external_dataset(
    expected_scale_factor: NativeTpchScaleFactor,
) -> DataFusionResult<(PathBuf, TpchDatasetMetadata)> {
    let dataset_mode = expected_scale_factor.dataset_mode_str();
    let parquet_dir = std::env::var_os(ALGEON_PARQUET_DIR_ENV).map(PathBuf::from);
    let (root, metadata) = match resolve_external_dataset_root(expected_scale_factor, parquet_dir)? {
        ExternalDatasetRoot::Explicit(root) => {
            let scale_factor = std::env::var(ALGEON_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
            let generation_command = std::env::var(ALGEON_TPCH_GENERATION_COMMAND_ENV).map_err(|_| {
                plan_error(format!(
                    "native TPC-H benchmark dataset_mode={dataset_mode} requires {ALGEON_TPCH_GENERATION_COMMAND_ENV} so {} provenance is reproducible",
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
        },
        ExternalDatasetRoot::Default(default_root) => {
            default_tpch_dataset_with_fingerprint(default_root, expected_scale_factor.as_str()).ok_or_else(|| {
                plan_error(format!(
                    "native TPC-H benchmark dataset_mode={dataset_mode} requires {ALGEON_PARQUET_DIR_ENV} or a populated default fixture at {default_root}; set ALGEON_TPCH_BENCH_DATASET_MODE=allow_tiny for local tiny/dev runs"
                ))
            })?
        },
    };
    validate_external_metadata(&metadata, expected_scale_factor)?;
    Ok((root, metadata))
}

fn default_root_for(scale_factor: NativeTpchScaleFactor) -> Option<&'static str> {
    let scale = match scale_factor {
        NativeTpchScaleFactor::Sf1 => TpchDefaultScale::Sf1,
        NativeTpchScaleFactor::Sf10 => TpchDefaultScale::Sf10,
        NativeTpchScaleFactor::Sf100 => TpchDefaultScale::Sf100Optimized,
        NativeTpchScaleFactor::Sf300 | NativeTpchScaleFactor::Sf500 | NativeTpchScaleFactor::Sf1000 => return None,
    };
    Some(default_tpch_parquet_root(scale))
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
            "native TPC-H benchmark dataset_mode={dataset_mode} requires {ALGEON_TPCH_SCALE_FACTOR_ENV}={}; observed {}",
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
            "native TPC-H benchmark dataset_mode={dataset_mode} requires a reproducible {ALGEON_TPCH_GENERATION_COMMAND_ENV}; observed {}",
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

    let expected_row_counts = expected_table_row_counts(expected_scale_factor)?;
    let required_columns = REQUIRED_COLUMNS.iter().copied().collect::<BTreeMap<_, _>>();
    for &expected_table in TPCH_TABLES {
        let Some(table) = metadata.tables.iter().find(|table| table.table == expected_table) else {
            return Err(plan_error(format!(
                "native TPC-H benchmark dataset_mode={dataset_mode} is missing table {expected_table}"
            )));
        };
        let Some(&expected_rows) = expected_row_counts.get(expected_table) else {
            return Err(plan_error(format!(
                "fixture/tpc-h/expected_rows.json has no {} row count for table {expected_table}",
                expected_scale_factor.label()
            )));
        };
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

/// Expected row count of each table at `scale_factor`, keyed by table name.
pub(super) fn expected_table_row_counts(
    scale_factor: NativeTpchScaleFactor,
) -> DataFusionResult<BTreeMap<String, u64>> {
    let mut scales = serde_json::from_str::<BTreeMap<String, BTreeMap<String, u64>>>(EXPECTED_ROWS_JSON)
        .map_err(|error| plan_error(format!("fixture/tpc-h/expected_rows.json is not valid: {error}")))?;
    scales.remove(scale_factor.as_str()).ok_or_else(|| {
        plan_error(format!("fixture/tpc-h/expected_rows.json has no {} row counts", scale_factor.label()))
    })
}

/// Registers the described tables and returns the scale factor of their data.
pub(super) async fn register_tpch_tables(
    ctx: &SessionContext,
    metadata: &TpchDatasetMetadata,
) -> DataFusionResult<TpchScaleFactor> {
    for table in &metadata.tables {
        let layout = TpchTableLayout::from_metadata(table);
        register_tpch_table_layout(ctx, &table.table, &layout).await.map_err(tpch_error)?;
    }
    metadata.scale().map_err(tpch_error)
}
