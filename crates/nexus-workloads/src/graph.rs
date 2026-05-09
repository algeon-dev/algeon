//! Read-only discovery for the canonical DBLP / AMiner graph fixture.
//!
//! The generator owns conversion. This module deliberately never invokes it:
//! a production-validation preflight must only verify the already prepared
//! Parquet layout and its generator-written provenance documents.

use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    fs::{self, File},
    io::Read as _,
    path::{Path, PathBuf},
};

use arrow::error::ArrowError;
use datafusion::{
    error::DataFusionError,
    prelude::{ParquetReadOptions, SessionContext},
};
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    errors::ParquetError,
    file::reader::{FileReader, SerializedFileReader},
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::metadata::WorkloadDatasetMetadata;

/// Dataset identifier written by `fixture/graph/dblp_ingest.py`.
pub const DBLP_DATASET_ID: &str = "citation-network";
/// Generator manifest/source format accepted by this discovery contract.
pub const DBLP_FIXTURE_SCHEMA_VERSION: u64 = 1;
/// Canonical DBLP / AMiner Citation Network V12 paper count.
pub const DBLP_EXPECTED_PAPERS: u64 = 4_894_081;
/// Canonical DBLP / AMiner Citation Network V12 directed edge count.
pub const DBLP_EXPECTED_EDGES: u64 = 45_564_149;
/// Canonical DBLP / AMiner Citation Network V12 paper-author row count.
pub const DBLP_EXPECTED_AUTHOR_ROWS: u64 = 14_934_850;
/// Canonical DBLP / AMiner Citation Network V12 paper-field-of-study row count.
pub const DBLP_EXPECTED_FOS_ROWS: u64 = 45_029_892;
/// Parquet-footer key carrying the full edge rowset's graph vertex upper
/// bound. The fixture generator writes it only for a complete DBLP V12
/// fixture; prefix samples cannot safely make this claim.
pub const DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY: &str = "nexus.graph.vertex_upper_bound.v1";
/// Parquet-footer key certifying that one complete DBLP edge file is a simple
/// directed relation over the declared physical source/destination columns.
/// The execution adapter may use this source-bound fact to omit Python edge
/// normalization only when the exact complete file rowset remains intact.
pub const DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY: &str = "nexus.graph.simple_directed_edge_topology.v1";
/// Manifest contract for a DBLP-derived source-hash graph slice.
pub const DBLP_MULTI_GRAPH_SLICE_PARTITION: &str = "source_hash_splitmix64_v1";
/// Version of the DBLP multi-graph slice manifest contract.
pub const DBLP_MULTI_GRAPH_SLICE_SCHEMA_VERSION: u64 = 1;
/// Required Parquet tables in deterministic artifact/report order.
pub const DBLP_TABLES: &[&str] =
    &["edges_by_src", "edges_by_dst", "profiles_by_paper_id", "paper_authors", "paper_fos"];

/// Read-only DBLP fixture discovery error.
#[derive(Debug)]
pub enum GraphDatasetError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    InvalidDataset(String),
    /// A fixture path that cannot be expressed as UTF-8 for table registration.
    NonUtf8Path(PathBuf),
}

impl Display for GraphDatasetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Arrow(error) => write!(formatter, "Arrow error: {error}"),
            Self::Parquet(error) => write!(formatter, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(formatter, "DataFusion error: {error}"),
            Self::InvalidDataset(message) => write!(formatter, "invalid DBLP graph dataset: {message}"),
            Self::NonUtf8Path(path) => write!(formatter, "non-UTF-8 DBLP fixture path: {}", path.display()),
        }
    }
}

impl StdError for GraphDatasetError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::DataFusion(error) => Some(error),
            Self::InvalidDataset(_) | Self::NonUtf8Path(_) => None,
        }
    }
}

impl From<std::io::Error> for GraphDatasetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for GraphDatasetError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for GraphDatasetError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for GraphDatasetError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

pub type GraphDatasetResult<T> = Result<T, GraphDatasetError>;

/// Fully resolved DBLP fixture layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DblpFixtureLayout {
    /// Directory containing `manifest.json`, `source.json`, and `parquet/`.
    pub dataset_root: PathBuf,
    /// Directory containing the five required Parquet files.
    pub parquet_root: PathBuf,
    pub manifest_path: PathBuf,
    pub source_path: PathBuf,
}

/// Metadata for one DBLP Parquet table, as verified against `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DblpTableMetadata {
    pub table: String,
    pub path: PathBuf,
    pub relative_path: String,
    pub byte_size: u64,
    pub row_count: u64,
    pub row_group_count: u64,
    pub schema: Vec<String>,
}

/// Generator-attested identity for one independently discoverable DBLP graph
/// slice. The hash is over the logical source-sorted edge rows, not Parquet
/// bytes, so it identifies the graph rowset across writer implementations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DblpMultiGraphSliceMetadata {
    pub partition: String,
    pub slice_count: u64,
    pub slice_index: u64,
    pub hash_seed: u64,
    pub edge_rowset_sha256: String,
}

/// Typed DBLP fixture metadata. It mirrors the TPC-H dataset metadata shape
/// while retaining the graph fixture's source and generated-manifest SHA-256
/// identities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DblpDatasetMetadata {
    /// The selected Parquet root, whether supplied directly or resolved from a dataset root.
    pub root: PathBuf,
    pub dataset_root: PathBuf,
    pub scale_factor: String,
    pub provenance: String,
    pub generation_command: String,
    pub table_count: usize,
    pub file_count: usize,
    pub total_rows: u64,
    pub total_bytes: u64,
    /// SHA-256 identity of the generated `manifest.json`.
    pub manifest_sha256: String,
    /// SHA-256 identity of the generated `source.json`.
    pub source_metadata_sha256: String,
    /// SHA-256 of the original source JSON, recorded by the fixture generator.
    pub source_content_sha256: String,
    /// Present only for a DBLP-derived independently discoverable graph slice.
    pub multi_graph_slice: Option<DblpMultiGraphSliceMetadata>,
    /// Per-table layout metadata in [`DBLP_TABLES`] order.
    pub tables: Vec<DblpTableMetadata>,
}

impl DblpDatasetMetadata {
    /// Projects graph-specific metadata into the shared serde artifact surface.
    ///
    /// `content_fingerprint` is the generator-attested SHA-256 of the raw
    /// DBLP source. Relational fixtures use a separately named FNV-1a64
    /// fingerprint, so callers must preserve the algorithm-qualified values.
    pub fn to_neutral_metadata(&self) -> WorkloadDatasetMetadata {
        WorkloadDatasetMetadata {
            workload: "dblp_aminer_citation_network",
            root: self.root.clone(),
            scale_factor: Some(self.scale_factor.clone()),
            provenance: self.provenance.clone(),
            generation_command: self.generation_command.clone(),
            table_count: self.table_count,
            file_count: self.file_count,
            total_rows: self.total_rows,
            total_bytes: self.total_bytes,
            metadata_fingerprint: format!("manifest-sha256:{}", self.manifest_sha256),
            content_fingerprint: Some(format!("source-sha256:{}", self.source_content_sha256)),
            table_file_counts: format_table_metadata(&self.tables, |_| 1),
            table_row_counts: format_table_metadata(&self.tables, |table| table.row_count),
            table_byte_sizes: format_table_metadata(&self.tables, |table| table.byte_size),
            table_row_group_counts: format_table_metadata(&self.tables, |table| table.row_group_count),
        }
    }
}

/// Finds and validates the DBLP fixture layout rooted either at
/// `<out-root>/citation_network/v1` or at its `parquet/` subdirectory.
pub fn discover_dblp_fixture_layout(root: &Path) -> GraphDatasetResult<DblpFixtureLayout> {
    let (dataset_root, parquet_root) = if root.join("parquet").is_dir() {
        (root.to_path_buf(), root.join("parquet"))
    } else if root.file_name().is_some_and(|name| name == "parquet") {
        let dataset_root = root.parent().ok_or_else(|| {
            GraphDatasetError::InvalidDataset(format!("Parquet root {} has no dataset parent", root.display()))
        })?;
        (dataset_root.to_path_buf(), root.to_path_buf())
    } else {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "expected DBLP dataset root with parquet/ or its parquet directory at {}",
            root.display()
        )));
    };

    if !dataset_root.is_dir() {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "dataset root {} is not a directory",
            dataset_root.display()
        )));
    }
    if !parquet_root.is_dir() {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "Parquet root {} is not a directory",
            parquet_root.display()
        )));
    }

    let manifest_path = dataset_root.join("manifest.json");
    let source_path = dataset_root.join("source.json");
    for (label, path) in [("manifest", &manifest_path), ("source metadata", &source_path)] {
        if !path.is_file() {
            return Err(GraphDatasetError::InvalidDataset(format!("missing {label} file {}", path.display())));
        }
    }
    for table in DBLP_TABLES {
        let path = parquet_root.join(format!("{table}.parquet"));
        if !path.is_file() {
            return Err(GraphDatasetError::InvalidDataset(format!("missing required Parquet file {}", path.display())));
        }
    }

    Ok(DblpFixtureLayout { dataset_root, parquet_root, manifest_path, source_path })
}

/// Registers the five [`DBLP_TABLES`] Parquet files under their fixture names.
///
/// Each table is exactly one file, so this needs no listing-table fan-in. The
/// returned layout lets a caller record the resolved roots it registered from.
pub async fn register_dblp_tables(ctx: &SessionContext, root: &Path) -> GraphDatasetResult<DblpFixtureLayout> {
    let layout = discover_dblp_fixture_layout(root)?;
    for table in DBLP_TABLES {
        let path = layout.parquet_root.join(format!("{table}.parquet"));
        let path_str = path.to_str().ok_or_else(|| GraphDatasetError::NonUtf8Path(path.clone()))?;
        ctx.register_parquet(*table, path_str, ParquetReadOptions::default()).await?;
    }
    Ok(layout)
}

/// Describes a prepared DBLP fixture without invoking the fixture generator.
///
/// This general discovery contract accepts samples as well as the production
/// fixture. It verifies the complete layout and generator metadata against the
/// actual Parquet file metadata. Use [`describe_canonical_dblp_dataset`] for
/// production preflight, which additionally requires the complete V12 corpus.
pub fn describe_dblp_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> GraphDatasetResult<DblpDatasetMetadata> {
    describe_dblp_dataset_with_contract(root, scale_factor.into(), generation_command.into(), false)
}

/// Describes the full canonical DBLP / AMiner Citation Network V12 fixture.
///
/// In addition to [`describe_dblp_dataset`]'s read-only Parquet validation,
/// this rejects generated samples and requires the canonical paper, edge,
/// author, and field-of-study row counts.
pub fn describe_canonical_dblp_dataset(
    root: &Path,
    scale_factor: impl Into<String>,
    generation_command: impl Into<String>,
) -> GraphDatasetResult<DblpDatasetMetadata> {
    describe_dblp_dataset_with_contract(root, scale_factor.into(), generation_command.into(), true)
}

fn describe_dblp_dataset_with_contract(
    root: &Path,
    scale_factor: String,
    generation_command: String,
    require_canonical_v12: bool,
) -> GraphDatasetResult<DblpDatasetMetadata> {
    let layout = discover_dblp_fixture_layout(root)?;
    let manifest = read_json_document(&layout.manifest_path, "manifest")?;
    let source = read_json_document(&layout.source_path, "source metadata")?;
    let provenance = validate_manifest(&manifest)?;
    let multi_graph_slice = parse_multi_graph_slice(&manifest)?;
    if require_canonical_v12 {
        if multi_graph_slice.is_some() {
            return Err(GraphDatasetError::InvalidDataset(
                "P6 preflight requires the canonical DBLP V12 fixture, not a derived multi-graph slice".to_string(),
            ));
        }
        validate_canonical_edge_vertex_footer_bounds(&layout, &manifest)?;
        validate_canonical_v12_manifest(&manifest)?;
    }
    let source_content_sha256 = validate_source_metadata(&source, multi_graph_slice.as_ref())?;
    let tables = read_table_metadata(&layout, &manifest)?;
    validate_manifest_counts(&manifest, &tables)?;
    validate_simple_directed_edge_topology_footer(&layout)?;

    let file_count = tables.len();
    let total_rows = tables.iter().map(|table| table.row_count).sum();
    let total_bytes = tables.iter().map(|table| table.byte_size).sum();
    Ok(DblpDatasetMetadata {
        root: layout.parquet_root,
        dataset_root: layout.dataset_root,
        scale_factor,
        provenance,
        generation_command,
        table_count: tables.len(),
        file_count,
        total_rows,
        total_bytes,
        manifest_sha256: sha256_file(&layout.manifest_path)?,
        source_metadata_sha256: sha256_file(&layout.source_path)?,
        source_content_sha256,
        multi_graph_slice,
        tables,
    })
}

fn read_json_document(path: &Path, label: &str) -> GraphDatasetResult<Value> {
    let bytes = fs::read(path)?;
    let document: Value = serde_json::from_slice(&bytes)
        .map_err(|_| GraphDatasetError::InvalidDataset(format!("{label} {} is not valid JSON", path.display())))?;
    if !document.is_object() {
        return Err(GraphDatasetError::InvalidDataset(format!("{label} {} must be a JSON object", path.display())));
    }
    Ok(document)
}

fn validate_manifest(manifest: &Value) -> GraphDatasetResult<String> {
    let schema_version = required_u64(manifest, &["schema_version"], "manifest schema_version")?;
    if schema_version != DBLP_FIXTURE_SCHEMA_VERSION {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "unsupported manifest schema_version {schema_version}; expected {DBLP_FIXTURE_SCHEMA_VERSION}"
        )));
    }
    let generator = required_string(manifest, &["generator"], "manifest generator")?;
    if !generator.ends_with("dblp_ingest.py") {
        return Err(GraphDatasetError::InvalidDataset(
            "manifest generator must identify fixture/graph/dblp_ingest.py".to_string(),
        ));
    }
    validate_dataset_id(manifest, "manifest")?;
    let max_papers = required_value(manifest, &["max_papers"], "manifest max_papers")?;
    if !max_papers.is_null() && max_papers.as_u64().filter(|value| *value > 0).is_none() {
        return Err(GraphDatasetError::InvalidDataset(
            "manifest max_papers must be null or a positive unsigned integer".to_string(),
        ));
    }
    required_string(manifest, &["dataset", "source"], "manifest dataset source").map(str::to_string)
}

fn parse_multi_graph_slice(manifest: &Value) -> GraphDatasetResult<Option<DblpMultiGraphSliceMetadata>> {
    let Some(slice) = manifest.get("multi_graph_slice") else {
        return Ok(None);
    };
    parse_multi_graph_slice_value(slice, "manifest multi_graph_slice").map(Some)
}

fn parse_multi_graph_slice_value(slice: &Value, label: &str) -> GraphDatasetResult<DblpMultiGraphSliceMetadata> {
    if !slice.is_object() {
        return Err(GraphDatasetError::InvalidDataset(format!("{label} must be an object")));
    }
    let schema_version = required_u64(slice, &["schema_version"], &format!("{label} schema_version"))?;
    if schema_version != DBLP_MULTI_GRAPH_SLICE_SCHEMA_VERSION {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "unsupported {label} schema_version {schema_version}; expected {DBLP_MULTI_GRAPH_SLICE_SCHEMA_VERSION}"
        )));
    }
    let partition = required_string(slice, &["partition"], &format!("{label} partition"))?;
    if partition != DBLP_MULTI_GRAPH_SLICE_PARTITION {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{label} partition must be {DBLP_MULTI_GRAPH_SLICE_PARTITION}, found {partition}"
        )));
    }
    let slice_count = required_u64(slice, &["slice_count"], &format!("{label} count"))?;
    if slice_count < 2 {
        return Err(GraphDatasetError::InvalidDataset(format!("{label} count must be at least two")));
    }
    let slice_index = required_u64(slice, &["slice_index"], &format!("{label} index"))?;
    if slice_index >= slice_count {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{label} index {slice_index} is outside slice_count {slice_count}"
        )));
    }
    let hash_seed = required_u64(slice, &["hash_seed"], &format!("{label} hash seed"))?;
    let edge_rowset_sha256 = required_string(slice, &["edge_rowset_sha256"], &format!("{label} edge rowset SHA-256"))?;
    if !is_sha256(edge_rowset_sha256) {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{label} edge_rowset_sha256 must be a 64-character hexadecimal digest"
        )));
    }
    Ok(DblpMultiGraphSliceMetadata {
        partition: partition.to_string(),
        slice_count,
        slice_index,
        hash_seed,
        edge_rowset_sha256: edge_rowset_sha256.to_ascii_lowercase(),
    })
}

fn validate_canonical_v12_manifest(manifest: &Value) -> GraphDatasetResult<()> {
    if !required_value(manifest, &["max_papers"], "manifest max_papers")?.is_null() {
        return Err(GraphDatasetError::InvalidDataset(
            "P6 preflight requires the full canonical DBLP V12 fixture, not a max_papers sample".to_string(),
        ));
    }
    for (label, actual, expected) in [
        ("papers", required_u64(manifest, &["counts", "papers"], "manifest paper count")?, DBLP_EXPECTED_PAPERS),
        ("edges", required_u64(manifest, &["counts", "edges"], "manifest edge count")?, DBLP_EXPECTED_EDGES),
        (
            "author_rows",
            required_u64(manifest, &["counts", "author_rows"], "manifest author row count")?,
            DBLP_EXPECTED_AUTHOR_ROWS,
        ),
        (
            "fos_rows",
            required_u64(manifest, &["counts", "fos_rows"], "manifest field-of-study row count")?,
            DBLP_EXPECTED_FOS_ROWS,
        ),
    ] {
        if actual != expected {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "P6 preflight requires canonical DBLP V12 {label}={expected}, found {actual}"
            )));
        }
    }
    Ok(())
}

/// Verifies the graph-cardinality source contract before the runner accepts a
/// full fixture. The footer belongs to each concrete edge file, rather than a
/// detached manifest/config value, so the planner can use it only when the
/// physical scan still denotes that whole rowset.
fn validate_canonical_edge_vertex_footer_bounds(
    layout: &DblpFixtureLayout,
    manifest: &Value,
) -> GraphDatasetResult<()> {
    let expected = required_u64(manifest, &["counts", "papers"], "manifest paper count")?;
    for table in ["edges_by_src", "edges_by_dst"] {
        let path = layout.parquet_root.join(format!("{table}.parquet"));
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let actual =
            parse_graph_vertex_upper_bound_footer(reader.metadata().file_metadata().key_value_metadata(), table)?;
        if actual != expected {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "{table} footer {DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY}={actual} does not match manifest paper count {expected}"
            )));
        }
    }
    Ok(())
}

fn parse_graph_vertex_upper_bound_footer(
    metadata: Option<&Vec<parquet::file::metadata::KeyValue>>,
    table: &str,
) -> GraphDatasetResult<u64> {
    let mut entries =
        metadata.into_iter().flatten().filter(|entry| entry.key == DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY);
    let Some(entry) = entries.next() else {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} is missing required footer key {DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY}"
        )));
    };
    if entries.next().is_some() {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} has duplicate footer key {DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY}"
        )));
    }
    entry.value.as_deref().and_then(|value| value.parse::<u64>().ok()).ok_or_else(|| {
        GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_GRAPH_VERTEX_UPPER_BOUND_FOOTER_KEY} must contain one unsigned integer"
        ))
    })
}

/// Verifies the edge-topology contract shared by canonical DBLP and its
/// source-hash slices. The certificate belongs to the physical edge files,
/// rather than a manifest claim, so source registration and graph lowering see
/// the same rowset-bound fact.
fn validate_simple_directed_edge_topology_footer(layout: &DblpFixtureLayout) -> GraphDatasetResult<()> {
    for table in ["edges_by_src", "edges_by_dst"] {
        let path = layout.parquet_root.join(format!("{table}.parquet"));
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let (source_column_index, destination_column_index) =
            parse_simple_directed_edge_topology_footer(reader.metadata().file_metadata().key_value_metadata(), table)?;
        if (source_column_index, destination_column_index) != (1, 2) {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "{table} footer {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} must bind src/dst physical columns (1, 2), found ({source_column_index}, {destination_column_index})"
            )));
        }
    }
    Ok(())
}

fn parse_simple_directed_edge_topology_footer(
    metadata: Option<&Vec<parquet::file::metadata::KeyValue>>,
    table: &str,
) -> GraphDatasetResult<(u64, u64)> {
    let mut entries =
        metadata.into_iter().flatten().filter(|entry| entry.key == DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY);
    let Some(entry) = entries.next() else {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} is missing required footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY}"
        )));
    };
    if entries.next().is_some() {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} has duplicate footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY}"
        )));
    }
    let Some(value) = entry.value.as_deref() else {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} must contain JSON"
        )));
    };
    let Value::Object(topology) = serde_json::from_str(value).map_err(|_| {
        GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} must contain a JSON object"
        ))
    })?
    else {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} must contain a JSON object"
        )));
    };
    if topology.len() != 2 {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} has an invalid schema"
        )));
    }
    let source_column_index = topology.get("source_column_index").and_then(Value::as_u64).ok_or_else(|| {
        GraphDatasetError::InvalidDataset(format!(
            "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} is missing source_column_index"
        ))
    })?;
    let destination_column_index =
        topology.get("destination_column_index").and_then(Value::as_u64).ok_or_else(|| {
            GraphDatasetError::InvalidDataset(format!(
                "{table} footer key {DBLP_SIMPLE_DIRECTED_EDGE_TOPOLOGY_FOOTER_KEY} is missing destination_column_index"
            ))
        })?;
    Ok((source_column_index, destination_column_index))
}

fn validate_source_metadata(
    source: &Value,
    expected_multi_graph_slice: Option<&DblpMultiGraphSliceMetadata>,
) -> GraphDatasetResult<String> {
    let schema_version = required_u64(source, &["schema_version"], "source schema_version")?;
    if schema_version != DBLP_FIXTURE_SCHEMA_VERSION {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "unsupported source metadata schema_version {schema_version}; expected {DBLP_FIXTURE_SCHEMA_VERSION}"
        )));
    }
    validate_dataset_id(source, "source metadata")?;
    let source_description = required_string(source, &["source"], "source metadata source")?;
    if source_description.is_empty() {
        return Err(GraphDatasetError::InvalidDataset("source metadata source must not be empty".to_string()));
    }
    required_string(source, &["source_json", "path"], "source JSON path")?;
    let source_bytes = required_u64(source, &["source_json", "bytes"], "source JSON byte size")?;
    if source_bytes == 0 {
        return Err(GraphDatasetError::InvalidDataset("source JSON byte size must be greater than zero".to_string()));
    }
    let sha256 = required_string(source, &["source_json", "sha256"], "source JSON SHA-256")?;
    if !is_sha256(sha256) {
        return Err(GraphDatasetError::InvalidDataset(
            "source JSON SHA-256 must be a 64-character hexadecimal digest; rerun generation without --skip-sha256"
                .to_string(),
        ));
    }
    match (expected_multi_graph_slice, source.get("derived_fixture")) {
        (None, None) => {},
        (None, Some(_)) => {
            return Err(GraphDatasetError::InvalidDataset(
                "source metadata declares derived_fixture but manifest is not a multi-graph slice".to_string(),
            ));
        },
        (Some(_), None) => {
            return Err(GraphDatasetError::InvalidDataset(
                "multi-graph slice source metadata is missing derived_fixture".to_string(),
            ));
        },
        (Some(expected), Some(derived)) => {
            let actual = parse_multi_graph_slice_value(derived, "source metadata derived_fixture")?;
            if &actual != expected {
                return Err(GraphDatasetError::InvalidDataset(
                    "source metadata derived_fixture does not match manifest multi_graph_slice".to_string(),
                ));
            }
        },
    }
    Ok(sha256.to_ascii_lowercase())
}

fn validate_dataset_id(document: &Value, label: &str) -> GraphDatasetResult<()> {
    let path = if document.get("dataset_id").is_some() {
        ["dataset_id"].as_slice()
    } else {
        ["dataset", "dataset_id"].as_slice()
    };
    let dataset_id = required_string(document, path, &format!("{label} dataset_id"))?;
    if dataset_id != DBLP_DATASET_ID {
        return Err(GraphDatasetError::InvalidDataset(format!(
            "{label} dataset_id must be {DBLP_DATASET_ID}, found {dataset_id}"
        )));
    }
    Ok(())
}

fn read_table_metadata(layout: &DblpFixtureLayout, manifest: &Value) -> GraphDatasetResult<Vec<DblpTableMetadata>> {
    let mut tables = Vec::with_capacity(DBLP_TABLES.len());
    for table in DBLP_TABLES {
        let manifest_path = match *table {
            "edges_by_src" => &["edge_layouts", "src", "parquet"][..],
            "edges_by_dst" => &["edge_layouts", "dst", "parquet"][..],
            "profiles_by_paper_id" => &["profiles"][..],
            "paper_authors" => &["paper_authors"][..],
            "paper_fos" => &["paper_fos"][..],
            _ => unreachable!("DBLP_TABLES is fixed"),
        };
        let entry = required_value(manifest, manifest_path, &format!("manifest metadata for {table}"))?;
        let path = layout.parquet_root.join(format!("{table}.parquet"));
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() {
            return Err(GraphDatasetError::InvalidDataset(format!("expected {path:?} to be a regular file")));
        }
        let byte_size = metadata.len();
        let expected_bytes = required_u64(entry, &["bytes"], &format!("manifest {table} bytes"))?;
        if byte_size != expected_bytes {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "manifest {table} byte size {expected_bytes} does not match filesystem size {byte_size}"
            )));
        }
        let row_count = required_u64(entry, &["num_rows"], &format!("manifest {table} row count"))?;
        let row_group_count = required_u64(entry, &["num_row_groups"], &format!("manifest {table} row-group count"))?;
        let schema = required_string_array(entry, &["schema"], &format!("manifest {table} schema"))?;
        if schema.is_empty() {
            return Err(GraphDatasetError::InvalidDataset(format!("manifest {table} schema must not be empty")));
        }
        let arrow_schema = ParquetRecordBatchReaderBuilder::try_new(File::open(&path)?)?.schema().clone();
        let actual_schema = arrow_schema.fields().iter().map(|field| field.name().to_string()).collect::<Vec<_>>();
        if actual_schema != schema {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "manifest {table} schema {:?} does not match Parquet schema {:?}",
                schema, actual_schema
            )));
        }
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let parquet_metadata = reader.metadata();
        let actual_row_group_count = u64::try_from(parquet_metadata.num_row_groups()).map_err(|_| {
            GraphDatasetError::InvalidDataset(format!(
                "Parquet row-group count does not fit u64 for {}",
                path.display()
            ))
        })?;
        if actual_row_group_count != row_group_count {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "manifest {table} row-group count {row_group_count} does not match Parquet row-group count {actual_row_group_count}"
            )));
        }
        let actual_row_count = parquet_metadata.row_groups().iter().try_fold(0_u64, |total, row_group| {
            let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
                GraphDatasetError::InvalidDataset(format!(
                    "negative row count in Parquet metadata for {}",
                    path.display()
                ))
            })?;
            total.checked_add(rows).ok_or_else(|| {
                GraphDatasetError::InvalidDataset(format!("Parquet row count overflows u64 for {}", path.display()))
            })
        })?;
        if actual_row_count != row_count {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "manifest {table} row count {row_count} does not match Parquet row count {actual_row_count}"
            )));
        }
        tables.push(DblpTableMetadata {
            table: (*table).to_string(),
            relative_path: format!("{table}.parquet"),
            path,
            byte_size,
            row_count,
            row_group_count,
            schema,
        });
    }
    Ok(tables)
}

fn validate_manifest_counts(manifest: &Value, tables: &[DblpTableMetadata]) -> GraphDatasetResult<()> {
    let table = |name: &str| {
        tables
            .iter()
            .find(|table| table.table == name)
            .ok_or_else(|| GraphDatasetError::InvalidDataset(format!("missing discovered table {name}")))
    };
    let papers = required_u64(manifest, &["counts", "papers"], "manifest paper count")?;
    let edges = required_u64(manifest, &["counts", "edges"], "manifest edge count")?;
    let author_rows = required_u64(manifest, &["counts", "author_rows"], "manifest author row count")?;
    let fos_rows = required_u64(manifest, &["counts", "fos_rows"], "manifest field-of-study row count")?;
    for (label, actual, expected) in [
        ("profiles_by_paper_id", table("profiles_by_paper_id")?.row_count, papers),
        ("edges_by_src", table("edges_by_src")?.row_count, edges),
        ("edges_by_dst", table("edges_by_dst")?.row_count, edges),
        ("paper_authors", table("paper_authors")?.row_count, author_rows),
        ("paper_fos", table("paper_fos")?.row_count, fos_rows),
    ] {
        if actual != expected {
            return Err(GraphDatasetError::InvalidDataset(format!(
                "manifest count mismatch for {label}: table has {actual} rows, manifest records {expected}"
            )));
        }
    }
    Ok(())
}

fn required_value<'a>(value: &'a Value, path: &[&str], label: &str) -> GraphDatasetResult<&'a Value> {
    let mut current = value;
    for key in path {
        current = current
            .as_object()
            .and_then(|object| object.get(*key))
            .ok_or_else(|| GraphDatasetError::InvalidDataset(format!("missing {label}")))?;
    }
    Ok(current)
}

fn required_string<'a>(value: &'a Value, path: &[&str], label: &str) -> GraphDatasetResult<&'a str> {
    required_value(value, path, label)?
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| GraphDatasetError::InvalidDataset(format!("{label} must be a non-empty string")))
}

fn required_u64(value: &Value, path: &[&str], label: &str) -> GraphDatasetResult<u64> {
    required_value(value, path, label)?
        .as_u64()
        .ok_or_else(|| GraphDatasetError::InvalidDataset(format!("{label} must be an unsigned integer")))
}

fn required_string_array(value: &Value, path: &[&str], label: &str) -> GraphDatasetResult<Vec<String>> {
    let values = required_value(value, path, label)?
        .as_array()
        .ok_or_else(|| GraphDatasetError::InvalidDataset(format!("{label} must be an array of strings")))?;
    values
        .iter()
        .map(|value| {
            value.as_str().filter(|value| !value.is_empty()).map(str::to_string).ok_or_else(|| {
                GraphDatasetError::InvalidDataset(format!("{label} must contain only non-empty strings"))
            })
        })
        .collect()
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn format_table_metadata(tables: &[DblpTableMetadata], value: impl Fn(&DblpTableMetadata) -> u64) -> String {
    tables.iter().map(|table| format!("{}={}", table.table, value(table))).collect::<Vec<_>>().join(",")
}

fn sha256_file(path: &Path) -> GraphDatasetResult<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex_digest(hasher.finalize().as_ref()))
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}
