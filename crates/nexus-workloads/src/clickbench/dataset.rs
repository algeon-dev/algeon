use std::{
    ffi::OsStr,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use arrow::datatypes::Schema;
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    file::reader::{FileReader, SerializedFileReader},
};

use super::{WorkloadError, WorkloadResult};
use crate::{NEXUS_PARQUET_DIR_ENV, metadata::DatasetFingerprint};

/// The single logical table exposed by the ClickBench fixture. Upstream uses
/// `hits` as the canonical view name; we register the raw Parquet under
/// [`CLICKBENCH_RAW_TABLE`] and surface the cast `EventDate` projection through
/// `hits` so the upstream queries.sql works verbatim.
pub const CLICKBENCH_TABLE: &str = "hits";

/// Raw table name used for the Parquet listing before the `hits` view applies
/// the canonical `EventDate` cast. Mirrors `ClickBench/datafusion/create.sql`.
pub const CLICKBENCH_RAW_TABLE: &str = "hits_raw";

/// Free-form provenance command recorded in dataset metadata.
pub const NEXUS_CLICKBENCH_GENERATION_COMMAND_ENV: &str = "NEXUS_CLICKBENCH_GENERATION_COMMAND";
/// Truthy flag requesting a content-fingerprint pass.
///
/// Accepted truthy values are `1`, `true`, `yes`, and `on`. The env value is
/// not used as the fingerprint; it only decides whether to scan file bytes.
pub const NEXUS_CLICKBENCH_CONTENT_FINGERPRINT_ENV: &str = "NEXUS_CLICKBENCH_CONTENT_FINGERPRINT";

/// In-repo default for the single-file ClickBench fixture. Used when no env
/// var is set and the partitioned variants are unavailable.
pub const DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET: &str = "./fixture/clickbench/data/hits.parquet";

/// In-repo default directory for the single-file ClickBench fixture, i.e. a
/// directory containing `hits.parquet`. Discovery resolves this to a
/// [`ClickbenchLayout::Single`].
pub const DEFAULT_LOCAL_CLICKBENCH_SINGLE_DIR: &str = "./fixture/clickbench/data/single";

/// In-repo default for the partitioned ClickBench fixture (100 shards).
pub const DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR: &str = "./fixture/clickbench/data/partitioned";

/// Discovered ClickBench fixture layout.
///
/// Mixed single-file and shard layouts are rejected so reports cannot silently
/// benchmark a different fixture than the caller intended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClickbenchLayout {
    /// One canonical Parquet file, usually `hits.parquet`.
    Single {
        /// Path to the canonical ClickBench Parquet file.
        file: PathBuf,
    },
    /// Canonical `hits_<N>.parquet` shards under one directory.
    ///
    /// Discovery admits only numeric-suffix shards, sorts them by suffix, and
    /// ignores unrelated Parquet files in the same directory.
    Partitioned {
        /// Directory containing the shard files.
        dir: PathBuf,
        /// Sorted shard paths included in the fixture.
        files: Vec<PathBuf>,
    },
}

impl ClickbenchLayout {
    /// Path used to register the table in DataFusion. Single layouts point at
    /// the canonical `hits.parquet`; partitioned layouts point at the
    /// containing directory (DataFusion's `register_parquet` handles directory
    /// roots when every file shares a schema).
    pub fn register_path(&self) -> &Path {
        match self {
            Self::Single { file } => file,
            Self::Partitioned { dir, .. } => dir,
        }
    }

    /// Concrete Parquet files in deterministic order.
    pub fn files(&self) -> Vec<&Path> {
        match self {
            Self::Single { file } => vec![file.as_path()],
            Self::Partitioned { files, .. } => files.iter().map(PathBuf::as_path).collect(),
        }
    }

    /// Number of concrete Parquet files in the layout.
    pub fn file_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Partitioned { files, .. } => files.len(),
        }
    }

    /// Stable report label: `single` or `partitioned`.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Single { .. } => "single",
            Self::Partitioned { .. } => "partitioned",
        }
    }
}

/// ClickBench fixture metadata recorded in report headers.
///
/// `metadata_fingerprint` is derived from schema, row counts, row-group
/// metadata, and byte sizes. `content_fingerprint`, when present, requires
/// reading Parquet contents and is intentionally opt-in for the 100M-row
/// fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClickbenchDatasetMetadata {
    /// Fixture root passed by the caller or selected by fallback discovery.
    pub root: PathBuf,
    /// Discovered single-file or partitioned layout.
    pub layout: ClickbenchLayout,
    /// Stable provenance label written to report headers.
    pub provenance: String,
    /// Free-form generation command or fixture description.
    pub generation_command: String,
    /// Total Parquet file count.
    pub file_count: usize,
    /// Total row count from Parquet metadata.
    pub total_rows: u64,
    /// Total file bytes from filesystem metadata.
    pub total_bytes: u64,
    /// Canonical Arrow debug schema string used by metadata fingerprints.
    pub schema: String,
    /// Total row-group count across files.
    pub row_group_count: usize,
    /// Row counts per row group, flattened in file order.
    pub row_group_rows: Vec<u64>,
    /// Parquet compressed bytes per row group, flattened in file order.
    pub row_group_bytes: Vec<u64>,
    /// Stable `metadata-fnv1a64:*` hash over layout, schema, row groups, sizes, and provenance.
    pub metadata_fingerprint: String,
    /// Optional `content-fnv1a64:*` hash; computing it reads Parquet contents.
    pub content_fingerprint: Option<String>,
}

impl ClickbenchDatasetMetadata {
    /// Stable layout label projected as the shared `dataset_scale_factor`.
    pub fn layout_label(&self) -> &'static str {
        self.layout.label()
    }

    /// Converts ClickBench metadata into shared report-header metadata.
    pub fn to_neutral_metadata(&self) -> crate::metadata::WorkloadDatasetMetadata {
        crate::metadata::WorkloadDatasetMetadata {
            workload: "clickbench",
            root: self.root.clone(),
            scale_factor: Some(self.layout_label().to_string()),
            provenance: self.provenance.clone(),
            generation_command: self.generation_command.clone(),
            table_count: 1,
            file_count: self.file_count,
            total_rows: self.total_rows,
            total_bytes: self.total_bytes,
            metadata_fingerprint: self.metadata_fingerprint.clone(),
            content_fingerprint: self.content_fingerprint.clone(),
            table_file_counts: format!("{}={}", CLICKBENCH_TABLE, self.file_count),
            table_row_counts: format!("{}={}", CLICKBENCH_TABLE, self.total_rows),
            table_byte_sizes: format!("{}={}", CLICKBENCH_TABLE, self.total_bytes),
            table_row_group_counts: format!("{}={}", CLICKBENCH_TABLE, self.row_group_count),
        }
    }
}

/// Discover a ClickBench layout under `root`. Accepts:
///
/// 1. A file path ending in `.parquet` (e.g. `fixture/clickbench/data/hits.parquet`).
/// 2. A directory containing `hits.parquet` (single layout).
/// 3. A directory containing `hits_*.parquet` (partitioned at this level).
/// 4. A directory containing a `partitioned/` subdir with `hits_*.parquet`.
///
/// Mixed layouts (both a `hits.parquet` and `hits_*.parquet` shards at the
/// same level) error out — that combination has no upstream meaning and is
/// almost always a stale-fixture bug.
pub fn discover_clickbench_layout(root: &Path) -> WorkloadResult<ClickbenchLayout> {
    if root.is_file() {
        ensure_parquet_extension(root)?;
        return Ok(ClickbenchLayout::Single { file: root.to_path_buf() });
    }
    if !root.is_dir() {
        return Err(WorkloadError::InvalidDataset(format!(
            "expected ClickBench fixture at {} (must be a Parquet file or a directory containing one)",
            root.display()
        )));
    }

    let single = root.join("hits.parquet");
    let shards = collect_hits_shards(root)?;

    match (single.is_file(), shards.is_empty()) {
        (true, true) => Ok(ClickbenchLayout::Single { file: single }),
        (false, false) => Ok(ClickbenchLayout::Partitioned { dir: root.to_path_buf(), files: shards }),
        (true, false) => Err(WorkloadError::InvalidDataset(format!(
            "ambiguous ClickBench fixture under {}: both hits.parquet and hits_*.parquet present",
            root.display()
        ))),
        (false, true) => {
            let nested = root.join("partitioned");
            if nested.is_dir() {
                let shards = collect_hits_shards(&nested)?;
                if shards.is_empty() {
                    return Err(WorkloadError::InvalidDataset(format!(
                        "expected hits_*.parquet shards under {}",
                        nested.display()
                    )));
                }
                return Ok(ClickbenchLayout::Partitioned { dir: nested, files: shards });
            }
            Err(WorkloadError::InvalidDataset(format!(
                "no ClickBench fixture found under {}: expected hits.parquet, hits_*.parquet, or partitioned/hits_*.parquet",
                root.display()
            )))
        },
    }
}

fn ensure_parquet_extension(path: &Path) -> WorkloadResult<()> {
    if path.extension() == Some(OsStr::new("parquet")) {
        Ok(())
    } else {
        Err(WorkloadError::InvalidDataset(format!(
            "expected ClickBench Parquet file at {} (got non-.parquet extension)",
            path.display()
        )))
    }
}

/// Collect canonical partitioned shards. The upstream
/// `download-hits-parquet-partitioned` script always names shards
/// `hits_<u32>.parquet`, so we admit only that pattern and silently drop any
/// other Parquet file in the directory — e.g. user samples, user-managed exports,
/// or `.parquet.tmp` half-writes. Anything that does not match is ignored
/// rather than rejected, since `partitioned/` directories are routinely
/// shared with unrelated data on staging volumes.
fn collect_hits_shards(dir: &Path) -> WorkloadResult<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        if let Some(index) = shard_index(&path) {
            shards.push((index, path));
        }
    }
    shards.sort_by_key(|(index, _)| *index);
    Ok(shards.into_iter().map(|(_, path)| path).collect())
}

fn shard_index(path: &Path) -> Option<u32> {
    let name = path.file_name().and_then(OsStr::to_str)?;
    let stem = name.strip_suffix(".parquet")?;
    let rest = stem.strip_prefix("hits_")?;
    rest.parse::<u32>().ok()
}

/// Build a [`ClickbenchDatasetMetadata`] for the layout rooted at `root`,
/// labelling it with the supplied provenance and generation command but
/// **without** scanning the underlying file bytes.
pub fn describe_clickbench_dataset(
    root: &Path,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
) -> WorkloadResult<ClickbenchDatasetMetadata> {
    describe_clickbench_dataset_with_content_fingerprint(root, provenance, generation_command, false)
}

/// Describes a ClickBench fixture and optionally hashes file contents.
///
/// Metadata fingerprints are cheap and derived from Parquet metadata. All file
/// schemas must match. Content fingerprints scan the full fixture, using paths
/// relative to the caller-supplied `root`.
pub fn describe_clickbench_dataset_with_content_fingerprint(
    root: &Path,
    provenance: impl Into<String>,
    generation_command: impl Into<String>,
    include_content_fingerprint: bool,
) -> WorkloadResult<ClickbenchDatasetMetadata> {
    let layout = discover_clickbench_layout(root)?;
    let mut files_meta = Vec::with_capacity(layout.file_count());
    for path in layout.files() {
        files_meta.push(read_parquet_file_metadata(root, path)?);
    }

    let schema = files_meta
        .first()
        .map(|f| f.schema.clone())
        .ok_or_else(|| WorkloadError::InvalidDataset("ClickBench fixture has no Parquet files".to_string()))?;
    if let Some(mismatched) = files_meta.iter().find(|f| f.schema != schema) {
        return Err(WorkloadError::InvalidDataset(format!(
            "schema mismatch in ClickBench fixture: {} differs from first shard",
            mismatched.path.display()
        )));
    }

    let file_count = files_meta.len();
    let total_rows = files_meta.iter().map(|f| f.row_count).sum();
    let total_bytes = files_meta.iter().map(|f| f.byte_size).sum();
    let row_group_count = files_meta.iter().map(|f| f.row_group_count).sum();
    let row_group_rows = files_meta.iter().flat_map(|f| f.row_group_rows.iter().copied()).collect();
    let row_group_bytes = files_meta.iter().flat_map(|f| f.row_group_bytes.iter().copied()).collect();

    let provenance = provenance.into();
    let generation_command = generation_command.into();
    let metadata_fingerprint =
        clickbench_dataset_metadata_fingerprint(root, &layout, &provenance, &generation_command, &schema, &files_meta);
    let content_fingerprint =
        include_content_fingerprint.then(|| clickbench_dataset_content_fingerprint(&files_meta)).transpose()?;

    Ok(ClickbenchDatasetMetadata {
        root: root.to_path_buf(),
        layout,
        provenance,
        generation_command,
        file_count,
        total_rows,
        total_bytes,
        schema,
        row_group_count,
        row_group_rows,
        row_group_bytes,
        metadata_fingerprint,
        content_fingerprint,
    })
}

/// Describes an explicitly supplied external ClickBench fixture.
pub fn external_clickbench_dataset_metadata(
    root: &Path,
    generation_command: impl Into<String>,
) -> WorkloadResult<ClickbenchDatasetMetadata> {
    describe_clickbench_dataset(root, "external_parquet_root", generation_command)
}

/// Discover the ClickBench fixture path for reports/benchmarks. Resolution
/// order:
///
/// 1. `NEXUS_PARQUET_DIR` env var (file or directory).
/// 2. Local partitioned default (`./fixture/clickbench/data/partitioned`).
/// 3. Local single-file default (`./fixture/clickbench/data/hits.parquet`).
///
/// The env-var fixture is strict: if set, it must describe a valid complete
/// ClickBench layout, and its generation command defaults to
/// `unknown_external`. Invalid implicit local candidates are skipped
/// during fallback. A truthy `NEXUS_CLICKBENCH_CONTENT_FINGERPRINT` requests the
/// expensive content hash. Returns `Ok(None)` when no implicit fixture exists.
pub fn prepare_clickbench_report_dataset() -> WorkloadResult<Option<(PathBuf, ClickbenchDatasetMetadata)>> {
    if let Ok(root) = std::env::var(NEXUS_PARQUET_DIR_ENV) {
        let root_path = PathBuf::from(&root);
        let generation_command =
            std::env::var(NEXUS_CLICKBENCH_GENERATION_COMMAND_ENV).unwrap_or_else(|_| "unknown_external".to_string());
        let include_content_fingerprint =
            std::env::var(NEXUS_CLICKBENCH_CONTENT_FINGERPRINT_ENV).is_ok_and(|value| env_truthy(&value));
        let metadata = describe_clickbench_dataset_with_content_fingerprint(
            &root_path,
            "external_parquet_root",
            generation_command,
            include_content_fingerprint,
        )?;
        return Ok(Some((root_path, metadata)));
    }

    let local_partitioned = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR);
    if local_partitioned.is_dir()
        && let Ok(metadata) = describe_clickbench_dataset(
            &local_partitioned,
            "external_parquet_root",
            format!("local {DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR} parquet fixture (default)"),
        )
    {
        return Ok(Some((local_partitioned, metadata)));
    }

    let local_single = PathBuf::from(DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET);
    if local_single.is_file()
        && let Ok(metadata) = describe_clickbench_dataset(
            &local_single,
            "external_parquet_root",
            format!("local {DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET} parquet fixture (default)"),
        )
    {
        return Ok(Some((local_single, metadata)));
    }

    Ok(None)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClickbenchFileMetadata {
    path: PathBuf,
    relative_path: String,
    schema: String,
    byte_size: u64,
    row_count: u64,
    row_group_count: usize,
    row_group_rows: Vec<u64>,
    row_group_bytes: Vec<u64>,
}

fn read_parquet_file_metadata(root: &Path, path: &Path) -> WorkloadResult<ClickbenchFileMetadata> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(WorkloadError::InvalidDataset(format!("expected {} to be a Parquet file", path.display())));
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
            WorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        let bytes = u64::try_from(row_group.compressed_size()).map_err(|_| {
            WorkloadError::InvalidDataset(format!(
                "negative compressed byte size in Parquet metadata for {}",
                path.display()
            ))
        })?;
        row_group_rows.push(rows);
        row_group_bytes.push(bytes);
    }
    let row_count = row_group_rows.iter().sum();
    Ok(ClickbenchFileMetadata {
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

fn clickbench_dataset_metadata_fingerprint(
    root: &Path,
    layout: &ClickbenchLayout,
    provenance: &str,
    generation_command: &str,
    schema: &str,
    files: &[ClickbenchFileMetadata],
) -> String {
    let mut hasher = DatasetFingerprint::new();
    hasher.update_field("datafusion-nexus-clickbench-dataset-metadata-v1");
    hasher.update_field(layout.label());
    hasher.update_field(&relative_dataset_path(root, layout.register_path()));
    hasher.update_field(provenance);
    hasher.update_field(generation_command);
    hasher.update_field(schema);
    for file in files {
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
    format!("metadata-fnv1a64:{:016x}", hasher.finish())
}

fn clickbench_dataset_content_fingerprint(files: &[ClickbenchFileMetadata]) -> WorkloadResult<String> {
    let mut hasher = DatasetFingerprint::new();
    hasher.update_field("datafusion-nexus-clickbench-dataset-content-v1");
    let mut buffer = vec![0_u8; 1024 * 1024];
    for file in files {
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
    let relative = path.strip_prefix(root).expect("discovered dataset path must be under root");
    relative.iter().map(|part| part.to_string_lossy()).collect::<Vec<_>>().join("/")
}

fn env_truthy(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
}
