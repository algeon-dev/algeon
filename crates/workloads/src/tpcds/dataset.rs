use std::path::{Path, PathBuf};

use super::{WorkloadError, WorkloadResult, fixture::write_tiny_tpcds_parquet_dataset};
use crate::{
    ALGEON_PARQUET_DIR_ENV,
    metadata::{DatasetFingerprint, WorkloadDatasetMetadata},
    parquet_metadata,
};

/// Canonical TPC-DS table names in registration/report order.
pub const TPCDS_TABLES: &[&str] = &[
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
];
/// Reported scale-factor label; set this explicitly for non-SF1 fixtures.
pub const ALGEON_TPCDS_SCALE_FACTOR_ENV: &str = "ALGEON_TPCDS_SCALE_FACTOR";
/// Free-form provenance command recorded in dataset metadata.
pub const ALGEON_TPCDS_GENERATION_COMMAND_ENV: &str = "ALGEON_TPCDS_GENERATION_COMMAND";
/// Truthy flag requesting a content-fingerprint pass. Without it, only the cheaper
/// metadata fingerprint is recorded.
pub const ALGEON_TPCDS_CONTENT_FINGERPRINT_ENV: &str = "ALGEON_TPCDS_CONTENT_FINGERPRINT";

/// In-repo default Parquet root for the TPC-DS SF1 fixture.
pub const DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR: &str = "./fixture/tpc-ds/data/sf1/parquet";

/// Returns the in-repo default Parquet root for the TPC-DS SF1 fixture.
pub fn default_tpcds_sf1_parquet_root() -> &'static str {
    DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR
}

/// TPC-DS fixture metadata recorded in report headers.
///
/// `metadata_fingerprint` is derived from paths, schema, row counts, row-group
/// metadata, and byte sizes. `content_fingerprint`, when present, requires
/// reading Parquet contents and is intentionally opt-in for large fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsDatasetMetadata {
    /// Fixture root used for table discovery.
    pub root: PathBuf,
    /// Report label such as `sf1`, `sf10`, `sf100`, `tiny`, or `unknown`.
    pub scale_factor: String,
    /// Stable provenance label written to report headers.
    pub provenance: String,
    /// Free-form generation command or fixture description.
    pub generation_command: String,
    /// Number of discovered TPC-DS tables.
    pub table_count: usize,
    /// Total Parquet file count across all tables.
    pub file_count: usize,
    /// Total row count from Parquet metadata.
    pub total_rows: u64,
    /// Total compressed file bytes from filesystem metadata.
    pub total_bytes: u64,
    /// Stable metadata hash; changes with paths, schema, row counts, row groups, or sizes.
    pub metadata_fingerprint: String,
    /// Optional content hash; computing it reads Parquet contents.
    pub content_fingerprint: Option<String>,
    /// Per-table metadata in canonical TPC-DS table order.
    pub tables: Vec<TpcdsTableMetadata>,
}

/// Per-table TPC-DS fixture metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableMetadata {
    /// Logical TPC-DS table name.
    pub table: String,
    /// Registration anchor: a single file or table directory.
    pub path: PathBuf,
    /// Canonical Arrow schema string used by metadata fingerprints.
    pub schema: String,
    /// File metadata in deterministic path order.
    pub files: Vec<TpcdsTableFileMetadata>,
    /// Number of Parquet files for this table.
    pub file_count: usize,
    /// Total table row count from Parquet metadata.
    pub row_count: u64,
    /// Total table byte size from filesystem metadata.
    pub byte_size: u64,
    /// Total table row-group count.
    pub row_group_count: usize,
    /// Row counts per row group, flattened in file order.
    pub row_group_rows: Vec<u64>,
    /// Compressed bytes per row group, flattened in file order.
    pub row_group_bytes: Vec<u64>,
}

/// Per-file TPC-DS Parquet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableFileMetadata {
    /// Absolute or caller-relative Parquet path as discovered.
    pub path: PathBuf,
    /// Path relative to the dataset root, used in fingerprints.
    pub relative_path: String,
    /// Canonical Arrow schema string.
    pub schema: String,
    /// File byte size from filesystem metadata.
    pub byte_size: u64,
    /// Row count from Parquet row-group metadata.
    pub row_count: u64,
    /// Number of row groups in this file.
    pub row_group_count: usize,
    /// Row counts per row group in file order.
    pub row_group_rows: Vec<u64>,
    /// Compressed bytes per row group in file order.
    pub row_group_bytes: Vec<u64>,
}

impl TpcdsDatasetMetadata {
    /// Formats per-table row counts as stable `table=value` pairs.
    pub fn table_row_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_count)
    }

    /// Formats per-table byte sizes as stable `table=value` pairs.
    pub fn table_byte_sizes(&self) -> String {
        format_table_metadata(&self.tables, |table| table.byte_size)
    }

    /// Formats per-table file counts as stable `table=value` pairs.
    pub fn table_file_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.file_count as u64)
    }

    /// Formats per-table row-group counts as stable `table=value` pairs.
    pub fn table_row_group_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_group_count as u64)
    }

    /// Converts TPC-DS metadata into the shared report dataset header shape.
    pub fn to_neutral_metadata(&self) -> WorkloadDatasetMetadata {
        WorkloadDatasetMetadata {
            workload: "tpcds",
            root: self.root.clone(),
            scale_factor: Some(self.scale_factor.clone()),
            provenance: self.provenance.clone(),
            generation_command: self.generation_command.clone(),
            table_count: self.table_count,
            file_count: self.file_count,
            total_rows: self.total_rows,
            total_bytes: self.total_bytes,
            metadata_fingerprint: self.metadata_fingerprint.clone(),
            content_fingerprint: self.content_fingerprint.clone(),
            table_file_counts: self.table_file_counts(),
            table_row_counts: self.table_row_counts(),
            table_byte_sizes: self.table_byte_sizes(),
            table_row_group_counts: self.table_row_group_counts(),
        }
    }
}

/// Describes the generated tiny TPC-DS fixture under `root`.
pub fn tiny_tpcds_dataset_metadata(root: &Path) -> WorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset(root, "tiny", "generated_tiny", "algeon-datafusion::write_tiny_tpcds_parquet_dataset")
}

/// Describes an externally supplied TPC-DS Parquet fixture.
pub fn external_tpcds_dataset_metadata(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> WorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset(root, scale_factor, "external_parquet_root", generation_command)
}

/// Selects the TPC-DS fixture for report helpers.
///
/// `ALGEON_PARQUET_DIR` is strict: if set, that fixture must be complete.
/// When unset, the helper tries the implicit SF1 default and falls back to
/// generating tiny data under `tiny_root`. A truthy
/// `ALGEON_TPCDS_CONTENT_FINGERPRINT` requests the expensive content hash.
pub fn prepare_tpcds_report_dataset(tiny_root: &Path) -> WorkloadResult<(PathBuf, TpcdsDatasetMetadata)> {
    if let Ok(root) = std::env::var(ALGEON_PARQUET_DIR_ENV) {
        let root = PathBuf::from(root);
        let scale_factor = std::env::var(ALGEON_TPCDS_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
        let generation_command =
            std::env::var(ALGEON_TPCDS_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
        let include_content_fingerprint =
            std::env::var(ALGEON_TPCDS_CONTENT_FINGERPRINT_ENV).is_ok_and(|value| env_truthy(&value));
        let metadata = describe_tpcds_dataset_with_content_fingerprint(
            &root,
            scale_factor,
            "external_parquet_root",
            generation_command,
            include_content_fingerprint,
        )?;
        return Ok((root, metadata));
    }

    if let Some((root, metadata)) = try_default_local_tpcds_sf1_dataset() {
        return Ok((root, metadata));
    }

    write_tiny_tpcds_parquet_dataset(tiny_root)?;
    let metadata = tiny_tpcds_dataset_metadata(tiny_root)?;
    Ok((tiny_root.to_path_buf(), metadata))
}

/// Returns the in-repo default SF1 fixture when all tables are present, or
/// `None` if the directory is missing or incomplete.
fn try_default_local_tpcds_sf1_dataset() -> Option<(PathBuf, TpcdsDatasetMetadata)> {
    let root_str = default_tpcds_sf1_parquet_root();
    let root = PathBuf::from(root_str);
    if !root.is_dir() {
        return None;
    }
    for table in TPCDS_TABLES {
        if discover_tpcds_table_layout(&root, table).is_err() {
            return None;
        }
    }
    let generation_command = format!("local {root_str} parquet fixture (default)");
    let metadata = describe_tpcds_dataset(&root, "sf1", "external_parquet_root", generation_command).ok()?;
    Some((root, metadata))
}

/// Describes a complete TPC-DS fixture without reading Parquet contents.
pub fn describe_tpcds_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
) -> WorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset_with_content_fingerprint(root, scale_factor, provenance, generation_command, false)
}

/// Describes a complete TPC-DS fixture and optionally hashes file contents.
///
/// Metadata fingerprints are cheap and derived from Parquet metadata. Content
/// fingerprints scan table bytes and emit `content-fnv1a64:*` values.
pub fn describe_tpcds_dataset_with_content_fingerprint(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
    include_content_fingerprint: bool,
) -> WorkloadResult<TpcdsDatasetMetadata> {
    let scale_factor = scale_factor.into();
    let provenance = provenance.into();
    let generation_command = generation_command.into();
    let mut tables = Vec::with_capacity(TPCDS_TABLES.len());
    for table in TPCDS_TABLES {
        tables.push(read_tpcds_table_metadata(root, table)?);
    }

    let file_count = tables.iter().map(|table| table.file_count).sum();
    let total_rows = tables.iter().map(|table| table.row_count).sum();
    let total_bytes = tables.iter().map(|table| table.byte_size).sum();
    let metadata_fingerprint =
        tpcds_dataset_metadata_fingerprint(root, &scale_factor, &provenance, &generation_command, &tables);
    let content_fingerprint = include_content_fingerprint
        .then(|| {
            parquet_metadata::dataset_content_fingerprint(
                "algeon-datafusion-tpcds-dataset-content-v1",
                tables.iter().map(|table| {
                    (
                        table.table.as_str(),
                        table
                            .files
                            .iter()
                            .map(|file| (file.relative_path.as_str(), file.byte_size, file.path.as_path())),
                    )
                }),
            )
        })
        .transpose()?;
    Ok(TpcdsDatasetMetadata {
        root: root.to_path_buf(),
        scale_factor,
        provenance,
        generation_command,
        table_count: tables.len(),
        file_count,
        total_rows,
        total_bytes,
        metadata_fingerprint,
        content_fingerprint,
        tables,
    })
}

/// Physical layout for one TPC-DS table fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableLayout {
    /// Registration anchor: single file or table directory.
    pub anchor: PathBuf,
    /// Concrete Parquet files in deterministic order.
    pub files: Vec<PathBuf>,
}

impl TpcdsTableLayout {
    /// Builds a table layout from previously collected table metadata.
    pub fn from_metadata(table: &TpcdsTableMetadata) -> Self {
        Self { anchor: table.path.clone(), files: table.files.iter().map(|file| file.path.clone()).collect() }
    }
}

/// Discovers one TPC-DS table layout under a fixture root.
///
/// Accepted layouts are `<table>.parquet` or `<table>/` containing Parquet
/// files recursively. Missing or empty layouts are fixture errors.
pub fn discover_tpcds_table_layout(root: &Path, table: &str) -> WorkloadResult<TpcdsTableLayout> {
    let file_path = root.join(format!("{table}.parquet"));
    if file_path.is_file() {
        return Ok(TpcdsTableLayout { anchor: file_path.clone(), files: vec![file_path] });
    }

    let directory_path = root.join(table);
    if directory_path.is_dir() {
        let mut files = Vec::new();
        parquet_metadata::collect_parquet_files(&directory_path, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(WorkloadError::InvalidDataset(format!(
                "expected at least one Parquet file under {}",
                directory_path.display()
            )));
        }
        return Ok(TpcdsTableLayout { anchor: directory_path, files });
    }

    Err(WorkloadError::InvalidDataset(format!(
        "expected TPC-DS table {table} at {} or {}/ under {}",
        file_path.display(),
        directory_path.display(),
        root.display()
    )))
}

fn read_tpcds_table_metadata(root: &Path, table: &str) -> WorkloadResult<TpcdsTableMetadata> {
    let layout = discover_tpcds_table_layout(root, table)?;
    let mut files = Vec::with_capacity(layout.files.len());
    for file_path in &layout.files {
        files.push(read_tpcds_table_file_metadata(root, file_path)?);
    }

    let schema = files
        .first()
        .map(|file| file.schema.clone())
        .ok_or_else(|| WorkloadError::InvalidDataset(format!("expected at least one Parquet file for {table}")))?;
    if let Some(mismatched) = files.iter().find(|file| file.schema != schema) {
        return Err(WorkloadError::InvalidDataset(format!(
            "schema mismatch in TPC-DS table {table}: {} differs from first file",
            mismatched.path.display()
        )));
    }

    let file_count = files.len();
    let row_count = files.iter().map(|file| file.row_count).sum();
    let byte_size = files.iter().map(|file| file.byte_size).sum();
    let row_group_count = files.iter().map(|file| file.row_group_count).sum();
    let row_group_rows = files.iter().flat_map(|file| file.row_group_rows.iter().copied()).collect();
    let row_group_bytes = files.iter().flat_map(|file| file.row_group_bytes.iter().copied()).collect();
    Ok(TpcdsTableMetadata {
        table: table.to_string(),
        path: layout.anchor,
        schema,
        files,
        file_count,
        row_count,
        byte_size,
        row_group_count,
        row_group_rows,
        row_group_bytes,
    })
}

fn read_tpcds_table_file_metadata(root: &Path, path: &Path) -> WorkloadResult<TpcdsTableFileMetadata> {
    let metadata = parquet_metadata::read_file_metadata(root, path)?;
    Ok(TpcdsTableFileMetadata {
        path: metadata.path,
        relative_path: metadata.relative_path,
        schema: metadata.schema,
        byte_size: metadata.byte_size,
        row_count: metadata.row_count,
        row_group_count: metadata.row_group_count,
        row_group_rows: metadata.row_group_rows,
        row_group_bytes: metadata.row_group_bytes,
    })
}

fn format_table_metadata(tables: &[TpcdsTableMetadata], value: impl Fn(&TpcdsTableMetadata) -> u64) -> String {
    tables.iter().map(|table| format!("{}={}", table.table, value(table))).collect::<Vec<_>>().join(",")
}

fn tpcds_dataset_metadata_fingerprint(
    root: &Path,
    scale_factor: &str,
    provenance: &str,
    generation_command: &str,
    tables: &[TpcdsTableMetadata],
) -> String {
    let mut hasher = DatasetFingerprint::new();
    hasher.update_field("algeon-datafusion-tpcds-dataset-metadata-v1");
    hasher.update_field(scale_factor);
    hasher.update_field(provenance);
    hasher.update_field(generation_command);
    for table in tables {
        hasher.update_field(&table.table);
        hasher.update_field(&parquet_metadata::relative_dataset_path(root, &table.path));
        hasher.update_field(&table.schema);
        hasher.update_field(&table.file_count.to_string());
        hasher.update_field(&table.byte_size.to_string());
        hasher.update_field(&table.row_count.to_string());
        hasher.update_field(&table.row_group_count.to_string());
        for file in &table.files {
            hasher.update_field(&file.relative_path);
            hasher.update_field(&file.schema);
            hasher.update_field(&file.byte_size.to_string());
            hasher.update_field(&file.row_count.to_string());
            hasher.update_field(&file.row_group_count.to_string());
            for rows in &file.row_group_rows {
                hasher.update_field(&rows.to_string());
            }
            for bytes in &file.row_group_bytes {
                hasher.update_field(&bytes.to_string());
            }
        }
    }
    format!("metadata-fnv1a64:{:016x}", hasher.finish())
}

fn env_truthy(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}
