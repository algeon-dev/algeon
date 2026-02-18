use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use arrow::datatypes::Schema;
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    file::reader::{FileReader, SerializedFileReader},
};

use super::{TpcdsWorkloadError, TpcdsWorkloadResult, tiny::write_tiny_tpcds_parquet_dataset};

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
pub const CUDF_TPCDS_PARQUET_DIR_ENV: &str = "CUDF_TPCDS_PARQUET_DIR";
pub const CUDF_TPCDS_SCALE_FACTOR_ENV: &str = "CUDF_TPCDS_SCALE_FACTOR";
pub const CUDF_TPCDS_GENERATION_COMMAND_ENV: &str = "CUDF_TPCDS_GENERATION_COMMAND";
pub const CUDF_TPCDS_CONTENT_FINGERPRINT_ENV: &str = "CUDF_TPCDS_CONTENT_FINGERPRINT";

/// Default on-host NVMe Parquet root for the TPC-DS SF1 fixture, staged at
/// boot by `rsync-tpcds.service`. Used as an implicit fallback by
/// `prepare_tpcds_report_dataset` when `CUDF_TPCDS_PARQUET_DIR` is unset and
/// the host actually has the fixture present; absence is silent (the
/// preparation falls back to the tiny dataset).
pub const DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR: &str = "/opt/dlami/nvme/tpcds/sf1/parquet";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsDatasetMetadata {
    pub root: PathBuf,
    pub scale_factor: String,
    pub provenance: String,
    pub generation_command: String,
    pub table_count: usize,
    pub file_count: usize,
    pub total_rows: u64,
    pub total_bytes: u64,
    pub metadata_fingerprint: String,
    pub content_fingerprint: Option<String>,
    pub tables: Vec<TpcdsTableMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableMetadata {
    pub table: String,
    pub path: PathBuf,
    pub schema: String,
    pub files: Vec<TpcdsTableFileMetadata>,
    pub file_count: usize,
    pub row_count: u64,
    pub byte_size: u64,
    pub row_group_count: usize,
    pub row_group_rows: Vec<u64>,
    pub row_group_bytes: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableFileMetadata {
    pub path: PathBuf,
    pub relative_path: String,
    pub schema: String,
    pub byte_size: u64,
    pub row_count: u64,
    pub row_group_count: usize,
    pub row_group_rows: Vec<u64>,
    pub row_group_bytes: Vec<u64>,
}

impl TpcdsDatasetMetadata {
    pub fn table_row_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_count)
    }

    pub fn table_byte_sizes(&self) -> String {
        format_table_metadata(&self.tables, |table| table.byte_size)
    }

    pub fn table_file_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.file_count as u64)
    }

    pub fn table_row_group_counts(&self) -> String {
        format_table_metadata(&self.tables, |table| table.row_group_count as u64)
    }

    pub fn to_workload_dataset_metadata(&self) -> crate::native_report::WorkloadDatasetMetadata {
        crate::native_report::WorkloadDatasetMetadata {
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

pub fn tiny_tpcds_dataset_metadata(root: &Path) -> TpcdsWorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset(root, "tiny", "generated_tiny", "datafusion-cudf::write_tiny_tpcds_parquet_dataset")
}

pub fn external_tpcds_dataset_metadata(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpcdsWorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset(root, scale_factor, "external_parquet_root", generation_command)
}

pub fn prepare_tpcds_report_dataset(tiny_root: &Path) -> TpcdsWorkloadResult<(PathBuf, TpcdsDatasetMetadata)> {
    if let Ok(root) = std::env::var(CUDF_TPCDS_PARQUET_DIR_ENV) {
        let root = PathBuf::from(root);
        let scale_factor = std::env::var(CUDF_TPCDS_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
        let generation_command =
            std::env::var(CUDF_TPCDS_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
        let include_content_fingerprint =
            std::env::var(CUDF_TPCDS_CONTENT_FINGERPRINT_ENV).is_ok_and(|value| env_truthy(&value));
        let metadata = describe_tpcds_dataset_with_content_fingerprint(
            &root,
            scale_factor,
            "external_parquet_root",
            generation_command,
            include_content_fingerprint,
        )?;
        return Ok((root, metadata));
    }

    if let Some((root, metadata)) = try_default_nvme_tpcds_sf1_dataset() {
        return Ok((root, metadata));
    }

    write_tiny_tpcds_parquet_dataset(tiny_root)?;
    let metadata = tiny_tpcds_dataset_metadata(tiny_root)?;
    Ok((tiny_root.to_path_buf(), metadata))
}

/// Returns the default NVMe SF1 fixture if `DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR`
/// exists and every required TPC-DS table is discoverable there. A `None`
/// return covers both "the NVMe drive isn't staged here" and "the fixture is
/// incomplete" — both are equivalent to "no implicit external fixture
/// available, fall back to the tiny dataset".
fn try_default_nvme_tpcds_sf1_dataset() -> Option<(PathBuf, TpcdsDatasetMetadata)> {
    let root = PathBuf::from(DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR);
    if !root.is_dir() {
        return None;
    }
    for table in TPCDS_TABLES {
        if discover_tpcds_table_layout(&root, table).is_err() {
            return None;
        }
    }
    let generation_command = format!("local {DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR} parquet fixture (default)");
    let metadata = describe_tpcds_dataset(&root, "sf1", "external_parquet_root", generation_command).ok()?;
    Some((root, metadata))
}

pub fn describe_tpcds_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpcdsWorkloadResult<TpcdsDatasetMetadata> {
    describe_tpcds_dataset_with_content_fingerprint(root, scale_factor, provenance, generation_command, false)
}

pub fn describe_tpcds_dataset_with_content_fingerprint(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
    include_content_fingerprint: bool,
) -> TpcdsWorkloadResult<TpcdsDatasetMetadata> {
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
    let content_fingerprint =
        include_content_fingerprint.then(|| tpcds_dataset_content_fingerprint(&tables)).transpose()?;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpcdsTableLayout {
    pub anchor: PathBuf,
    pub files: Vec<PathBuf>,
}

impl TpcdsTableLayout {
    pub fn from_metadata(table: &TpcdsTableMetadata) -> Self {
        Self { anchor: table.path.clone(), files: table.files.iter().map(|file| file.path.clone()).collect() }
    }
}

pub fn discover_tpcds_table_layout(root: &Path, table: &str) -> TpcdsWorkloadResult<TpcdsTableLayout> {
    let file_path = root.join(format!("{table}.parquet"));
    if file_path.is_file() {
        return Ok(TpcdsTableLayout { anchor: file_path.clone(), files: vec![file_path] });
    }

    let directory_path = root.join(table);
    if directory_path.is_dir() {
        let mut files = Vec::new();
        collect_parquet_files(&directory_path, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(TpcdsWorkloadError::InvalidDataset(format!(
                "expected at least one Parquet file under {}",
                directory_path.display()
            )));
        }
        return Ok(TpcdsTableLayout { anchor: directory_path, files });
    }

    Err(TpcdsWorkloadError::InvalidDataset(format!(
        "expected TPC-DS table {table} at {} or {}/ under {}",
        file_path.display(),
        directory_path.display(),
        root.display()
    )))
}

fn read_tpcds_table_metadata(root: &Path, table: &str) -> TpcdsWorkloadResult<TpcdsTableMetadata> {
    let layout = discover_tpcds_table_layout(root, table)?;
    let mut files = Vec::with_capacity(layout.files.len());
    for file_path in &layout.files {
        files.push(read_tpcds_table_file_metadata(root, file_path)?);
    }

    let schema = files
        .first()
        .map(|file| file.schema.clone())
        .ok_or_else(|| TpcdsWorkloadError::InvalidDataset(format!("expected at least one Parquet file for {table}")))?;
    if let Some(mismatched) = files.iter().find(|file| file.schema != schema) {
        return Err(TpcdsWorkloadError::InvalidDataset(format!(
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

fn collect_parquet_files(path: &Path, files: &mut Vec<PathBuf>) -> TpcdsWorkloadResult<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_parquet_files(&entry_path, files)?;
        } else if file_type.is_file() && entry_path.extension().is_some_and(|extension| extension == "parquet") {
            files.push(entry_path);
        }
    }
    Ok(())
}

fn read_tpcds_table_file_metadata(root: &Path, path: &Path) -> TpcdsWorkloadResult<TpcdsTableFileMetadata> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(TpcdsWorkloadError::InvalidDataset(format!("expected {} to be a Parquet file", path.display())));
    }

    let byte_size = metadata.len();
    let arrow_schema = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.schema().clone();
    let schema = canonical_schema(arrow_schema.as_ref());
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let parquet_metadata = reader.metadata();
    let row_group_count = parquet_metadata.num_row_groups();
    let mut row_group_rows = Vec::with_capacity(row_group_count);
    let mut row_group_bytes = Vec::with_capacity(row_group_count);
    for row_group in parquet_metadata.row_groups() {
        let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
            TpcdsWorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        let bytes = u64::try_from(row_group.compressed_size()).map_err(|_| {
            TpcdsWorkloadError::InvalidDataset(format!(
                "negative compressed byte size in Parquet metadata for {}",
                path.display()
            ))
        })?;
        row_group_rows.push(rows);
        row_group_bytes.push(bytes);
    }
    let row_count = row_group_rows.iter().sum();
    Ok(TpcdsTableFileMetadata {
        path: path.to_path_buf(),
        relative_path: relative_dataset_path(root, path),
        schema,
        byte_size,
        row_count,
        row_group_count,
        row_group_rows,
        row_group_bytes,
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
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-cudf-tpcds-dataset-metadata-v1");
    hasher.update_field(scale_factor);
    hasher.update_field(provenance);
    hasher.update_field(generation_command);
    for table in tables {
        hasher.update_field(&table.table);
        hasher.update_field(&relative_dataset_path(root, &table.path));
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

fn tpcds_dataset_content_fingerprint(tables: &[TpcdsTableMetadata]) -> TpcdsWorkloadResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-cudf-tpcds-dataset-content-v1");
    let mut buffer = vec![0_u8; 1024 * 1024];
    for table in tables {
        hasher.update_field(&table.table);
        for file in &table.files {
            hasher.update_field(&file.relative_path);
            hasher.update_field(&file.byte_size.to_string());
            let mut reader = File::open(&file.path)?;
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update_bytes(&buffer[..bytes_read]);
            }
            hasher.update_bytes(&[0]);
        }
    }
    Ok(format!("content-fnv1a64:{:016x}", hasher.finish()))
}

fn canonical_schema(schema: &Schema) -> String {
    schema
        .fields()
        .iter()
        .map(|field| {
            format!(
                "{}:{:?}:nullable={}:metadata={:?}",
                field.name(),
                field.data_type(),
                field.is_nullable(),
                field.metadata()
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn relative_dataset_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative.iter().map(|part| part.to_string_lossy()).collect::<Vec<_>>().join("/")
}

fn env_truthy(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}

struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self { state: Self::OFFSET }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(self) -> u64 {
        self.state
    }
}
