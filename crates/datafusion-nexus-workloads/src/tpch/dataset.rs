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

use super::{TpchWorkloadError, TpchWorkloadResult, fixture::write_tiny_tpch_parquet_dataset};

/// Canonical TPC-H table names in fixture discovery order.
pub const TPCH_TABLES: &[&str] =
    &["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"];
/// Explicit Parquet fixture root. When unset, report helpers use the default
/// NVMe/local fallback chain before generating tiny data where allowed.
pub const CUDF_TPCH_PARQUET_DIR_ENV: &str = "CUDF_TPCH_PARQUET_DIR";
/// Reported scale-factor label; it does not change fixture discovery by itself.
pub const CUDF_TPCH_SCALE_FACTOR_ENV: &str = "CUDF_TPCH_SCALE_FACTOR";
/// Free-form provenance command recorded in dataset metadata.
pub const CUDF_TPCH_GENERATION_COMMAND_ENV: &str = "CUDF_TPCH_GENERATION_COMMAND";
/// Optional caller-supplied content fingerprint. When unset, only the cheaper
/// metadata fingerprint is recorded unless the caller explicitly requests a
/// content-fingerprint pass.
pub const CUDF_TPCH_CONTENT_FINGERPRINT_ENV: &str = "CUDF_TPCH_CONTENT_FINGERPRINT";

/// Root of the on-host NVMe mount where `rsync-tpch.service` /
/// `rsync-tpcds.service` stage the canonical fixtures. When this mount point
/// is missing (i.e. running on a dev host without the AWS DLAMI NVMe drive),
/// the dataset probes fall back to the in-repo `./fixture/data/...` roots
/// below. "Mount missing" is the only fallback trigger — a present mount with
/// an incomplete fixture is treated as a real fixture problem, not as
/// "fixture not available here".
pub const DEFAULT_NVME_MOUNT_ROOT: &str = "/opt/dlami/nvme";

/// Default on-host NVMe Parquet root for the TPC-H SF1 fixture, staged at
/// boot by `rsync-tpch.service`. Used as an implicit fallback by
/// `prepare_tpch_report_dataset` when `CUDF_TPCH_PARQUET_DIR` is unset and
/// the host actually has the fixture present; absence is silent (the
/// preparation falls back to the tiny dataset).
pub const DEFAULT_NVME_TPCH_SF1_PARQUET_DIR: &str = "/opt/dlami/nvme/tpch/sf1/parquet";

/// Default on-host NVMe Parquet root for the TPC-H SF10 fixture.
pub const DEFAULT_NVME_TPCH_SF10_PARQUET_DIR: &str = "/opt/dlami/nvme/tpch/sf10/parquet";

/// Default on-host NVMe Parquet root for the TPC-H SF100 `_optimized`
/// fixture (sibling-split `lineitem`, single-file other tables).
pub const DEFAULT_NVME_TPCH_SF100_OPTIMIZED_PARQUET_DIR: &str = "/opt/dlami/nvme/tpch/sf100_optimized/parquet";

/// In-repo fallback Parquet root for the TPC-H SF1 fixture, used when the
/// `DEFAULT_NVME_MOUNT_ROOT` mount point is not present on this host.
pub const DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR: &str = "./fixture/data/tpch/sf1/parquet";

/// In-repo fallback Parquet root for the TPC-H SF10 fixture.
pub const DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR: &str = "./fixture/data/tpch/sf10/parquet";

/// In-repo fallback Parquet root for the TPC-H SF100 `_optimized` fixture.
pub const DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR: &str = "./fixture/data/tpch/sf100_optimized/parquet";

/// True when `DEFAULT_NVME_MOUNT_ROOT` exists on this host. Used by the
/// dataset probes to decide between NVMe-staged fixtures and the in-repo
/// `./fixture/data/...` fallback roots.
pub fn default_nvme_mount_present() -> bool {
    Path::new(DEFAULT_NVME_MOUNT_ROOT).is_dir()
}

/// Returns the on-host default Parquet root for the given TPC-H scale, picking
/// the NVMe-staged path when `DEFAULT_NVME_MOUNT_ROOT` is mounted and the
/// in-repo `./fixture/data/...` path otherwise. The mount check is purely about
/// whether the NVMe drive is attached — fixture completeness is not consulted
/// here; the caller probes that separately.
pub fn default_tpch_parquet_root(scale: TpchDefaultScale) -> &'static str {
    if default_nvme_mount_present() { scale.nvme_root() } else { scale.local_root() }
}

/// TPC-H scale factors that have a canonical on-host default Parquet root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpchDefaultScale {
    Sf1,
    Sf10,
    Sf100Optimized,
}

impl TpchDefaultScale {
    pub const fn nvme_root(self) -> &'static str {
        match self {
            Self::Sf1 => DEFAULT_NVME_TPCH_SF1_PARQUET_DIR,
            Self::Sf10 => DEFAULT_NVME_TPCH_SF10_PARQUET_DIR,
            Self::Sf100Optimized => DEFAULT_NVME_TPCH_SF100_OPTIMIZED_PARQUET_DIR,
        }
    }

    pub const fn local_root(self) -> &'static str {
        match self {
            Self::Sf1 => DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR,
            Self::Sf10 => DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR,
            Self::Sf100Optimized => DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR,
        }
    }
}

/// TPC-H fixture metadata recorded in report headers.
///
/// `metadata_fingerprint` is derived from paths, schema, row counts, row-group
/// metadata, and byte sizes. `content_fingerprint`, when present, requires
/// reading Parquet contents and is intentionally opt-in for large fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpchDatasetMetadata {
    /// Fixture root used for table discovery.
    pub root: PathBuf,
    /// Report label such as `sf1`, `sf10`, `sf100`, `tiny`, or `unknown`.
    pub scale_factor: String,
    /// Stable provenance label written to report headers.
    pub provenance: String,
    /// Free-form generation command or fixture description.
    pub generation_command: String,
    /// Number of discovered TPC-H tables.
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
    /// Per-table metadata in canonical TPC-H table order.
    pub tables: Vec<TpchTableMetadata>,
}

/// Per-table TPC-H fixture metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpchTableMetadata {
    /// Logical TPC-H table name.
    pub table: String,
    /// Registration anchor: a single file, a table directory, or first sibling shard.
    pub path: PathBuf,
    /// Canonical Arrow schema string used by metadata fingerprints.
    pub schema: String,
    /// File metadata in deterministic path order.
    pub files: Vec<TpchTableFileMetadata>,
    pub file_count: usize,
    pub row_count: u64,
    pub byte_size: u64,
    pub row_group_count: usize,
    /// Row counts per row group, flattened in file order.
    pub row_group_rows: Vec<u64>,
    /// Compressed bytes per row group, flattened in file order.
    pub row_group_bytes: Vec<u64>,
}

/// Per-file TPC-H Parquet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TpchTableFileMetadata {
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
    pub row_group_count: usize,
    /// Row counts per row group in file order.
    pub row_group_rows: Vec<u64>,
    /// Compressed bytes per row group in file order.
    pub row_group_bytes: Vec<u64>,
}

impl TpchDatasetMetadata {
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

    pub fn to_neutral_metadata(&self) -> crate::metadata::WorkloadDatasetMetadata {
        crate::metadata::WorkloadDatasetMetadata {
            workload: "tpch",
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

pub fn tiny_tpch_dataset_metadata(root: &Path) -> TpchWorkloadResult<TpchDatasetMetadata> {
    describe_tpch_dataset(root, "tiny", "generated_tiny", "datafusion-nexus::write_tiny_tpch_parquet_dataset")
}

pub fn external_tpch_dataset_metadata(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpchWorkloadResult<TpchDatasetMetadata> {
    describe_tpch_dataset(root, scale_factor, "external_parquet_root", generation_command)
}

/// Selects the TPC-H fixture for report helpers.
///
/// `CUDF_TPCH_PARQUET_DIR` is strict: if set, that fixture must be complete.
/// When unset, the helper tries the implicit SF1 default and falls back to
/// generating tiny data under `tiny_root`. A truthy
/// `CUDF_TPCH_CONTENT_FINGERPRINT` requests the expensive content hash.
pub fn prepare_tpch_report_dataset(tiny_root: &Path) -> TpchWorkloadResult<(PathBuf, TpchDatasetMetadata)> {
    if let Ok(root) = std::env::var(CUDF_TPCH_PARQUET_DIR_ENV) {
        let root = PathBuf::from(root);
        let scale_factor = std::env::var(CUDF_TPCH_SCALE_FACTOR_ENV).unwrap_or_else(|_| "unknown".to_string());
        let generation_command =
            std::env::var(CUDF_TPCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
        let include_content_fingerprint =
            std::env::var(CUDF_TPCH_CONTENT_FINGERPRINT_ENV).is_ok_and(|value| env_truthy(&value));
        let metadata = describe_tpch_dataset_with_content_fingerprint(
            &root,
            scale_factor,
            "external_parquet_root",
            generation_command,
            include_content_fingerprint,
        )?;
        return Ok((root, metadata));
    }

    if let Some((root, metadata)) = try_default_nvme_tpch_sf1_dataset() {
        return Ok((root, metadata));
    }

    write_tiny_tpch_parquet_dataset(tiny_root)?;
    let metadata = tiny_tpch_dataset_metadata(tiny_root)?;
    Ok((tiny_root.to_path_buf(), metadata))
}

/// Returns the on-host default SF1 fixture when it is complete. The probe
/// picks between the NVMe-staged root and the in-repo `./fixture/data/...`
/// fallback based solely on whether `DEFAULT_NVME_MOUNT_ROOT` is mounted on
/// this host; fixture completeness is then checked at the chosen root. A
/// `None` return means "no implicit external fixture available, fall back to
/// the tiny dataset".
fn try_default_nvme_tpch_sf1_dataset() -> Option<(PathBuf, TpchDatasetMetadata)> {
    default_nvme_tpch_dataset(default_tpch_parquet_root(TpchDefaultScale::Sf1), "sf1")
}

/// Lower-level helper: probe an arbitrary on-host NVMe Parquet root for the
/// fixed TPC-H table set and, if all tables are discoverable, describe it as
/// an external fixture with a synthetic `generation_command`. Returns `None`
/// when the directory is missing or the fixture is incomplete.
pub fn default_nvme_tpch_dataset(root: &str, scale_factor: &str) -> Option<(PathBuf, TpchDatasetMetadata)> {
    let root_path = PathBuf::from(root);
    if !root_path.is_dir() {
        return None;
    }
    for table in TPCH_TABLES {
        if discover_tpch_table_layout(&root_path, table).is_err() {
            return None;
        }
    }
    let generation_command = format!("local {root} parquet fixture (default)");
    let metadata = describe_tpch_dataset(&root_path, scale_factor, "external_parquet_root", generation_command).ok()?;
    Some((root_path, metadata))
}

/// Same as `default_nvme_tpch_dataset` but includes a content fingerprint —
/// needed by the benchmark binary's strict-validation gate.
pub fn default_nvme_tpch_dataset_with_fingerprint(
    root: &str,
    scale_factor: &str,
) -> Option<(PathBuf, TpchDatasetMetadata)> {
    let root_path = PathBuf::from(root);
    if !root_path.is_dir() {
        return None;
    }
    for table in TPCH_TABLES {
        if discover_tpch_table_layout(&root_path, table).is_err() {
            return None;
        }
    }
    let generation_command = format!("local {root} parquet fixture (default)");
    let metadata = describe_tpch_dataset_with_content_fingerprint(
        &root_path,
        scale_factor,
        "external_parquet_root",
        generation_command,
        true,
    )
    .ok()?;
    Some((root_path, metadata))
}

/// Describes a complete TPC-H fixture without reading Parquet contents.
pub fn describe_tpch_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
) -> TpchWorkloadResult<TpchDatasetMetadata> {
    describe_tpch_dataset_with_content_fingerprint(root, scale_factor, provenance, generation_command, false)
}

/// Describes a complete TPC-H fixture and optionally hashes file contents.
///
/// Metadata fingerprints are cheap and derived from Parquet metadata. Content
/// fingerprints scan table bytes and use a stable `tpch-content-fnv1a64-v1`
/// report vocabulary.
pub fn describe_tpch_dataset_with_content_fingerprint(
    root: &Path,
    scale_factor: impl Into<String>,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
    include_content_fingerprint: bool,
) -> TpchWorkloadResult<TpchDatasetMetadata> {
    let scale_factor = scale_factor.into();
    let provenance = provenance.into();
    let generation_command = generation_command.into();
    let mut tables = Vec::with_capacity(TPCH_TABLES.len());
    for table in TPCH_TABLES {
        tables.push(read_tpch_table_metadata(root, table)?);
    }

    let file_count = tables.iter().map(|table| table.file_count).sum();
    let total_rows = tables.iter().map(|table| table.row_count).sum();
    let total_bytes = tables.iter().map(|table| table.byte_size).sum();
    let metadata_fingerprint =
        tpch_dataset_metadata_fingerprint(root, &scale_factor, &provenance, &generation_command, &tables);
    let content_fingerprint =
        include_content_fingerprint.then(|| tpch_dataset_content_fingerprint(&tables)).transpose()?;
    Ok(TpchDatasetMetadata {
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
pub struct TpchTableLayout {
    /// Registration anchor: single file, table directory, or first sibling shard.
    pub anchor: PathBuf,
    /// Concrete Parquet files in deterministic order.
    pub files: Vec<PathBuf>,
}

impl TpchTableLayout {
    pub fn from_metadata(table: &TpchTableMetadata) -> Self {
        Self { anchor: table.path.clone(), files: table.files.iter().map(|file| file.path.clone()).collect() }
    }
}

/// Discovers one TPC-H table layout under a fixture root.
///
/// Accepted layouts are `<table>.parquet`, `<table>/` containing Parquet
/// files recursively, or sibling shards named `<table>_NNNN.parquet`.
pub fn discover_tpch_table_layout(root: &Path, table: &str) -> TpchWorkloadResult<TpchTableLayout> {
    let file_path = root.join(format!("{table}.parquet"));
    if file_path.is_file() {
        return Ok(TpchTableLayout { anchor: file_path.clone(), files: vec![file_path] });
    }

    let directory_path = root.join(table);
    if directory_path.is_dir() {
        let mut files = Vec::new();
        collect_parquet_files(&directory_path, &mut files)?;
        files.sort();
        if files.is_empty() {
            return Err(TpchWorkloadError::InvalidDataset(format!(
                "expected at least one Parquet file under {}",
                directory_path.display()
            )));
        }
        return Ok(TpchTableLayout { anchor: directory_path, files });
    }

    let sibling_shards = discover_sibling_split_shards(root, table)?;
    if !sibling_shards.is_empty() {
        let anchor = sibling_shards.first().cloned().expect("non-empty sibling_shards guarantees a first element");
        return Ok(TpchTableLayout { anchor, files: sibling_shards });
    }

    Err(TpchWorkloadError::InvalidDataset(format!(
        "expected TPC-H table {table} at {}, {}/, or {}_NNNN.parquet siblings under {}",
        file_path.display(),
        directory_path.display(),
        table,
        root.display()
    )))
}

fn discover_sibling_split_shards(root: &Path, table: &str) -> TpchWorkloadResult<Vec<PathBuf>> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let prefix = format!("{table}_");
    let mut shards = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with(&prefix) || !name.ends_with(".parquet") {
            continue;
        }
        let shard_suffix = &name[prefix.len()..name.len() - ".parquet".len()];
        if shard_suffix.is_empty() || !shard_suffix.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        shards.push(path);
    }
    shards.sort();
    Ok(shards)
}

fn read_tpch_table_metadata(root: &Path, table: &str) -> TpchWorkloadResult<TpchTableMetadata> {
    let layout = discover_tpch_table_layout(root, table)?;
    let mut files = Vec::with_capacity(layout.files.len());
    for file_path in &layout.files {
        files.push(read_tpch_table_file_metadata(root, file_path)?);
    }

    let schema = files
        .first()
        .map(|file| file.schema.clone())
        .ok_or_else(|| TpchWorkloadError::InvalidDataset(format!("expected at least one Parquet file for {table}")))?;
    if let Some(mismatched) = files.iter().find(|file| file.schema != schema) {
        return Err(TpchWorkloadError::InvalidDataset(format!(
            "schema mismatch in TPC-H table {table}: {} differs from first file",
            mismatched.path.display()
        )));
    }

    let file_count = files.len();
    let row_count = files.iter().map(|file| file.row_count).sum();
    let byte_size = files.iter().map(|file| file.byte_size).sum();
    let row_group_count = files.iter().map(|file| file.row_group_count).sum();
    let row_group_rows = files.iter().flat_map(|file| file.row_group_rows.iter().copied()).collect();
    let row_group_bytes = files.iter().flat_map(|file| file.row_group_bytes.iter().copied()).collect();
    Ok(TpchTableMetadata {
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

fn collect_parquet_files(path: &Path, files: &mut Vec<PathBuf>) -> TpchWorkloadResult<()> {
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

fn read_tpch_table_file_metadata(root: &Path, path: &Path) -> TpchWorkloadResult<TpchTableFileMetadata> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(TpchWorkloadError::InvalidDataset(format!("expected {} to be a Parquet file", path.display())));
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
            TpchWorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        let bytes = u64::try_from(row_group.compressed_size()).map_err(|_| {
            TpchWorkloadError::InvalidDataset(format!(
                "negative compressed byte size in Parquet metadata for {}",
                path.display()
            ))
        })?;
        row_group_rows.push(rows);
        row_group_bytes.push(bytes);
    }
    let row_count = row_group_rows.iter().sum();
    Ok(TpchTableFileMetadata {
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

fn format_table_metadata(tables: &[TpchTableMetadata], value: impl Fn(&TpchTableMetadata) -> u64) -> String {
    tables.iter().map(|table| format!("{}={}", table.table, value(table))).collect::<Vec<_>>().join(",")
}

fn tpch_dataset_metadata_fingerprint(
    root: &Path,
    scale_factor: &str,
    provenance: &str,
    generation_command: &str,
    tables: &[TpchTableMetadata],
) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-tpch-dataset-metadata-v2");
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

fn tpch_dataset_content_fingerprint(tables: &[TpchTableMetadata]) -> TpchWorkloadResult<String> {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("datafusion-nexus-tpch-dataset-content-v1");
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
