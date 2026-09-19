//! Read-only discovery for the forest-inference workloads.
//!
//! Two workloads score pretrained tree ensembles over Parquet: the EMBER2024
//! malware feature splits with the upstream LightGBM models, both written by
//! `fixture/demo/ember2024/scripts/prepare.py`, and TPC-H `lineitem` with a 500-tree
//! XGBoost JSON regression model over seven numeric columns. Like
//! [`crate::graph`], this module never invokes a generator; it validates the
//! prepared files against their Parquet footers.

use std::{
    fs::{self, File},
    io::Read as _,
    path::{Path, PathBuf},
};

use arrow::datatypes::DataType;
use datafusion::prelude::{ParquetReadOptions, SessionContext};
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    file::reader::{FileReader, SerializedFileReader},
};
use sha2::{Digest as _, Sha256};

pub use crate::error::{WorkloadError, WorkloadResult};
use crate::{
    metadata::WorkloadDatasetMetadata,
    tpch::{TpchTableMetadata, describe_tpch_dataset},
};

/// EMBER2024 fixture root containing `parquet/` and `models/`.
pub const ALGEON_EMBER2024_DIR_ENV: &str = "ALGEON_EMBER2024_DIR";
/// In-repo default EMBER2024 fixture root.
pub const DEFAULT_LOCAL_EMBER2024_DIR: &str = "./fixture/demo/ember2024";
/// TPC-H Parquet root whose `lineitem` table feeds the lineitem forest workload.
pub const ALGEON_FOREST_TPCH_PARQUET_DIR_ENV: &str = "ALGEON_FOREST_TPCH_PARQUET_DIR";
/// XGBoost JSON model scored over [`LINEITEM_FOREST_FEATURES`].
pub const ALGEON_FOREST_LINEITEM_MODEL_ENV: &str = "ALGEON_FOREST_LINEITEM_MODEL";
/// Host-provisioned, gitignored lineitem model; `fixture/README.md` describes
/// how to train it.
pub const DEFAULT_LOCAL_FOREST_LINEITEM_MODEL: &str = "./fixture/tpc-h/models/lineitem_xgboost_500t_d8.json";

/// Split names `prepare.py` may export, in report order.
pub const EMBER2024_SPLITS: &[&str] = &["test", "challenge", "train"];
/// `cuml_forest_predict` `model_format` of every discovered EMBER2024 model.
pub const EMBER2024_MODEL_FORMAT: &str = "lightgbm";
/// `cuml_forest_predict` `model_format` of the lineitem model.
pub const LINEITEM_FOREST_MODEL_FORMAT: &str = "xgboost_json";
/// Lineitem feature columns in the model's feature order. The model was
/// trained on each column cast to Float32.
pub const LINEITEM_FOREST_FEATURES: &[&str] =
    &["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_discount", "l_tax"];

/// A LightGBM text model starts with this line; `thrember.download_models`
/// also ships pickled multi-booster models that no forest loader reads.
const LIGHTGBM_TEXT_MAGIC: &[u8] = b"tree\n";

/// One prepared EMBER2024 Parquet split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ember2024SplitMetadata {
    pub split: String,
    pub path: PathBuf,
    pub byte_size: u64,
    pub row_count: u64,
    pub row_group_count: u64,
}

/// One pretrained model file, identified by the SHA-256 of its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForestModelMetadata {
    /// File stem, e.g. `EMBER2024_all`.
    pub name: String,
    pub path: PathBuf,
    pub model_format: &'static str,
    pub byte_size: u64,
    pub sha256: String,
}

/// Prepared EMBER2024 fixture: every present split shares one schema of
/// `id Int64, f0..f{feature_count-1} Float32, label Int32`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ember2024DatasetMetadata {
    pub root: PathBuf,
    pub parquet_root: PathBuf,
    pub models_root: PathBuf,
    pub feature_count: usize,
    /// Present splits in [`EMBER2024_SPLITS`] order.
    pub splits: Vec<Ember2024SplitMetadata>,
    /// LightGBM text models in file-name order. Pickled models are skipped.
    pub models: Vec<ForestModelMetadata>,
}

impl Ember2024DatasetMetadata {
    pub fn split(&self, split: &str) -> Option<&Ember2024SplitMetadata> {
        self.splits.iter().find(|metadata| metadata.split == split)
    }

    pub fn model(&self, name: &str) -> Option<&ForestModelMetadata> {
        self.models.iter().find(|model| model.name == name)
    }

    /// Feature column names in model feature order.
    pub fn feature_columns(&self) -> Vec<String> {
        (0..self.feature_count).map(|index| format!("f{index}")).collect()
    }

    /// Projects into the shared artifact surface. `metadata_fingerprint` is a
    /// SHA-256 over split sizes, row counts, and the feature count; the
    /// `content_fingerprint` is a SHA-256 over the model digests. Parquet
    /// bytes are not hashed.
    pub fn to_neutral_metadata(&self) -> WorkloadDatasetMetadata {
        let mut layout = Sha256::new();
        for split in &self.splits {
            layout
                .update(format!("{}={}:{}:{}\n", split.split, split.byte_size, split.row_count, split.row_group_count));
        }
        layout.update(format!("features={}\n", self.feature_count));
        let mut models = Sha256::new();
        for model in &self.models {
            models.update(format!("{}={}\n", model.name, model.sha256));
        }
        let format_splits = |value: fn(&Ember2024SplitMetadata) -> u64| {
            self.splits.iter().map(|split| format!("{}={}", split.split, value(split))).collect::<Vec<_>>().join(",")
        };
        WorkloadDatasetMetadata {
            workload: "ember2024",
            root: self.parquet_root.clone(),
            scale_factor: None,
            provenance: "fixture/demo/ember2024/scripts/prepare.py".to_string(),
            generation_command: "fixture/fixture.sh ember2024 prepare".to_string(),
            table_count: self.splits.len(),
            file_count: self.splits.len(),
            total_rows: self.splits.iter().map(|split| split.row_count).sum(),
            total_bytes: self.splits.iter().map(|split| split.byte_size).sum(),
            metadata_fingerprint: format!("layout-sha256:{}", hex_digest(layout.finalize().as_ref())),
            content_fingerprint: Some(format!("models-sha256:{}", hex_digest(models.finalize().as_ref()))),
            table_file_counts: format_splits(|_| 1),
            table_row_counts: format_splits(|split| split.row_count),
            table_byte_sizes: format_splits(|split| split.byte_size),
            table_row_group_counts: format_splits(|split| split.row_group_count),
        }
    }
}

/// Table name [`register_ember2024_tables`] registers a split under.
pub fn ember2024_table_name(split: &str) -> String {
    format!("ember2024_{split}")
}

/// Describes a prepared EMBER2024 fixture rooted at `root` (the directory
/// holding `parquet/` and `models/`).
///
/// At least one split and one LightGBM text model must be present. The
/// feature count is read from the Parquet schema, not assumed.
pub fn describe_ember2024_dataset(root: &Path) -> WorkloadResult<Ember2024DatasetMetadata> {
    let parquet_root = root.join("parquet");
    let models_root = root.join("models");
    for (label, path) in [("Parquet", &parquet_root), ("models", &models_root)] {
        if !path.is_dir() {
            return Err(WorkloadError::InvalidDataset(format!(
                "EMBER2024 {label} directory {} is missing; run fixture/fixture.sh ember2024 prepare",
                path.display()
            )));
        }
    }

    let mut feature_count = None;
    let mut splits = Vec::new();
    for split in EMBER2024_SPLITS {
        let path = parquet_root.join(format!("{split}.parquet"));
        if !path.is_file() {
            continue;
        }
        let split_features = ember2024_feature_count(&path)?;
        match feature_count {
            None => feature_count = Some(split_features),
            Some(expected) if expected != split_features => {
                return Err(WorkloadError::InvalidDataset(format!(
                    "EMBER2024 split {split} has {split_features} features; earlier splits have {expected}"
                )));
            },
            Some(_) => {},
        }
        let reader = SerializedFileReader::new(File::open(&path)?)?;
        let metadata = reader.metadata();
        let row_count = u64::try_from(metadata.file_metadata().num_rows()).map_err(|_| {
            WorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        splits.push(Ember2024SplitMetadata {
            split: (*split).to_string(),
            byte_size: fs::metadata(&path)?.len(),
            row_count,
            row_group_count: metadata.num_row_groups() as u64,
            path,
        });
    }
    let Some(feature_count) = feature_count else {
        return Err(WorkloadError::InvalidDataset(format!(
            "no EMBER2024 split ({}) found under {}",
            EMBER2024_SPLITS.join(", "),
            parquet_root.display()
        )));
    };

    let models = discover_lightgbm_models(&models_root)?;
    if models.is_empty() {
        return Err(WorkloadError::InvalidDataset(format!(
            "no LightGBM text model found under {}",
            models_root.display()
        )));
    }
    Ok(Ember2024DatasetMetadata { root: root.to_path_buf(), parquet_root, models_root, feature_count, splits, models })
}

/// Registers every present split as [`ember2024_table_name`].
pub async fn register_ember2024_tables(
    ctx: &SessionContext,
    metadata: &Ember2024DatasetMetadata,
) -> WorkloadResult<()> {
    for split in &metadata.splits {
        let path = split.path.to_str().ok_or_else(|| WorkloadError::NonUtf8Path(split.path.clone()))?;
        ctx.register_parquet(ember2024_table_name(&split.split), path, ParquetReadOptions::default()).await?;
    }
    Ok(())
}

/// TPC-H `lineitem` scored by the pretrained XGBoost regression model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineitemForestWorkload {
    pub tpch_root: PathBuf,
    pub lineitem: TpchTableMetadata,
    pub model: ForestModelMetadata,
}

impl LineitemForestWorkload {
    /// `cuml_forest_predict` input over `table`: an Int64 `id` unique per
    /// lineitem row (`l_linenumber` is at most 7) and the model features cast
    /// to Float32 in model order.
    ///
    /// TPC-H Parquet writers declare every column nullable although TPC-H
    /// data has no nulls, and cuML inputs require non-nullable fields. The
    /// `COALESCE` fallbacks only change the declared nullability: a null
    /// feature becomes NaN, which the finite-input check rejects at
    /// execution, and a null `id` implies a null `l_orderkey` or
    /// `l_linenumber` feature on the same row.
    pub fn input_sql(&self, table: &str) -> String {
        let features = LINEITEM_FOREST_FEATURES
            .iter()
            .map(|column| format!("COALESCE(CAST({column} AS FLOAT), CAST('NaN' AS FLOAT)) AS {column}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("SELECT COALESCE(l_orderkey * 8 + l_linenumber, -1) AS id, {features} FROM {table}")
    }
}

/// Describes the lineitem forest workload from a TPC-H Parquet root and the
/// XGBoost JSON model path.
pub fn describe_lineitem_forest_workload(
    tpch_root: &Path,
    model_path: &Path,
) -> WorkloadResult<LineitemForestWorkload> {
    let dataset = describe_tpch_dataset(tpch_root, "unknown", "external_parquet_root", "external")?;
    let lineitem = dataset
        .tables
        .into_iter()
        .find(|table| table.table == "lineitem")
        .ok_or_else(|| WorkloadError::InvalidDataset(format!("no lineitem table under {}", tpch_root.display())))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&lineitem.files[0].path)?)?;
    let schema = reader.schema();
    for column in LINEITEM_FOREST_FEATURES {
        let field = schema
            .field_with_name(column)
            .map_err(|_| WorkloadError::InvalidDataset(format!("lineitem has no forest feature column {column}")))?;
        if !field.data_type().is_numeric() {
            return Err(WorkloadError::InvalidDataset(format!(
                "lineitem forest feature {column} is {}, not numeric",
                field.data_type()
            )));
        }
    }
    if !model_path.is_file() {
        return Err(WorkloadError::InvalidDataset(format!(
            "lineitem forest model {} is missing; scripts/dev/bench_forest_stage0.py trains it",
            model_path.display()
        )));
    }
    let model = forest_model_metadata(model_path, LINEITEM_FOREST_MODEL_FORMAT)?;
    Ok(LineitemForestWorkload { tpch_root: tpch_root.to_path_buf(), lineitem, model })
}

fn ember2024_feature_count(path: &Path) -> WorkloadResult<usize> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = reader.schema();
    let fields = schema.fields();
    let invalid = |detail: String| {
        WorkloadError::InvalidDataset(format!(
            "EMBER2024 split {} must have columns id Int64, f0..f<d-1> Float32, label Int32: {detail}",
            path.display()
        ))
    };
    if fields.len() < 3 {
        return Err(invalid(format!("found {} columns", fields.len())));
    }
    let last = fields.len() - 1;
    for (index, field) in fields.iter().enumerate() {
        let (name, data_type) = match index {
            0 => ("id".to_string(), DataType::Int64),
            index if index == last => ("label".to_string(), DataType::Int32),
            index => (format!("f{}", index - 1), DataType::Float32),
        };
        if field.name() != &name || field.data_type() != &data_type {
            return Err(invalid(format!(
                "column {index} is {} {}, expected {name} {data_type}",
                field.name(),
                field.data_type()
            )));
        }
    }
    Ok(fields.len() - 2)
}

fn discover_lightgbm_models(models_root: &Path) -> WorkloadResult<Vec<ForestModelMetadata>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(models_root)? {
        let path = entry?.path();
        if path.is_file() && path.extension().is_some_and(|extension| extension == "model") {
            paths.push(path);
        }
    }
    paths.sort();
    let mut models = Vec::new();
    for path in paths {
        let mut magic = [0_u8; LIGHTGBM_TEXT_MAGIC.len()];
        let is_text = File::open(&path)?.read_exact(&mut magic).is_ok() && magic == LIGHTGBM_TEXT_MAGIC;
        if is_text {
            models.push(forest_model_metadata(&path, EMBER2024_MODEL_FORMAT)?);
        }
    }
    Ok(models)
}

fn forest_model_metadata(path: &Path, model_format: &'static str) -> WorkloadResult<ForestModelMetadata> {
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| WorkloadError::NonUtf8Path(path.to_path_buf()))?
        .to_string();
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut byte_size = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        byte_size += count as u64;
        hasher.update(&buffer[..count]);
    }
    Ok(ForestModelMetadata {
        name,
        path: path.to_path_buf(),
        model_format,
        byte_size,
        sha256: hex_digest(hasher.finalize().as_ref()),
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
