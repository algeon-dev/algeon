use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    path::PathBuf,
};

use arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use parquet::errors::ParquetError;

pub mod catalog;
pub mod dataset;
pub mod registration;

pub use catalog::{CLICKBENCH_QUERIES, CLICKBENCH_QUERY_COUNT, ClickbenchQuery, clickbench_query_ids};
pub use dataset::{
    CLICKBENCH_RAW_TABLE, CLICKBENCH_TABLE, ClickbenchDatasetMetadata, ClickbenchLayout,
    DEFAULT_LOCAL_CLICKBENCH_PARTITIONED_DIR, DEFAULT_LOCAL_CLICKBENCH_SINGLE_DIR,
    DEFAULT_LOCAL_CLICKBENCH_SINGLE_PARQUET, NEXUS_CLICKBENCH_CONTENT_FINGERPRINT_ENV,
    NEXUS_CLICKBENCH_GENERATION_COMMAND_ENV, describe_clickbench_dataset,
    describe_clickbench_dataset_with_content_fingerprint, discover_clickbench_layout,
    external_clickbench_dataset_metadata, prepare_clickbench_report_dataset,
};
pub use registration::register_clickbench_table;

pub use crate::NEXUS_PARQUET_DIR_ENV;

pub type ClickbenchWorkloadResult<T> = std::result::Result<T, ClickbenchWorkloadError>;

#[derive(Debug)]
pub enum ClickbenchWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for ClickbenchWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid ClickBench dataset: {message}"),
        }
    }
}

impl StdError for ClickbenchWorkloadError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::DataFusion(error) => Some(error),
            Self::NonUtf8Path(_) | Self::InvalidDataset(_) => None,
        }
    }
}

impl From<std::io::Error> for ClickbenchWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<ArrowError> for ClickbenchWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}
impl From<ParquetError> for ClickbenchWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}
impl From<DataFusionError> for ClickbenchWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}
