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
pub mod fixture;
pub mod registration;

pub use catalog::{TPCH_QUERIES, TPCH_QUERY_COUNT, TpchQuery, tpch_query_ids};
pub use dataset::{
    CUDF_TPCH_CONTENT_FINGERPRINT_ENV, CUDF_TPCH_GENERATION_COMMAND_ENV, CUDF_TPCH_SCALE_FACTOR_ENV,
    DEFAULT_LOCAL_TPCH_SF1_PARQUET_DIR, DEFAULT_LOCAL_TPCH_SF10_PARQUET_DIR,
    DEFAULT_LOCAL_TPCH_SF100_OPTIMIZED_PARQUET_DIR, TPCH_TABLES, TpchDatasetMetadata, TpchDefaultScale,
    TpchTableFileMetadata, TpchTableLayout, TpchTableMetadata, default_tpch_dataset,
    default_tpch_dataset_with_fingerprint, default_tpch_parquet_root, describe_tpch_dataset,
    describe_tpch_dataset_with_content_fingerprint, discover_tpch_table_layout, external_tpch_dataset_metadata,
    prepare_tpch_report_dataset, tiny_tpch_dataset_metadata,
};
pub use fixture::write_tiny_tpch_parquet_dataset;
pub use registration::{register_tpch_table_layout, register_tpch_tables};

pub use crate::NEXUS_PARQUET_DIR_ENV;

pub type TpchWorkloadResult<T> = std::result::Result<T, TpchWorkloadError>;

#[derive(Debug)]
pub enum TpchWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for TpchWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid TPC-H dataset: {message}"),
        }
    }
}

impl StdError for TpchWorkloadError {
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

impl From<std::io::Error> for TpchWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ArrowError> for TpchWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}

impl From<ParquetError> for TpchWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}

impl From<DataFusionError> for TpchWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}
