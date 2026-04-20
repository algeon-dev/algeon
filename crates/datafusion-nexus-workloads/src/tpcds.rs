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

pub use catalog::{TPCDS_QUERIES, TPCDS_QUERY_COUNT, TpcdsQuery, tpcds_query_ids};
pub use dataset::{
    CUDF_TPCDS_CONTENT_FINGERPRINT_ENV, CUDF_TPCDS_GENERATION_COMMAND_ENV, CUDF_TPCDS_PARQUET_DIR_ENV,
    CUDF_TPCDS_SCALE_FACTOR_ENV, DEFAULT_LOCAL_TPCDS_SF1_PARQUET_DIR, DEFAULT_NVME_TPCDS_SF1_PARQUET_DIR, TPCDS_TABLES,
    TpcdsDatasetMetadata, TpcdsTableFileMetadata, TpcdsTableLayout, TpcdsTableMetadata, default_tpcds_sf1_parquet_root,
    describe_tpcds_dataset, describe_tpcds_dataset_with_content_fingerprint, discover_tpcds_table_layout,
    external_tpcds_dataset_metadata, prepare_tpcds_report_dataset, tiny_tpcds_dataset_metadata,
};
pub use fixture::write_tiny_tpcds_parquet_dataset;
pub use registration::register_tpcds_tables;

pub type TpcdsWorkloadResult<T> = std::result::Result<T, TpcdsWorkloadError>;

#[derive(Debug)]
pub enum TpcdsWorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for TpcdsWorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid TPC-DS dataset: {message}"),
        }
    }
}

impl StdError for TpcdsWorkloadError {
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

impl From<std::io::Error> for TpcdsWorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<ArrowError> for TpcdsWorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}
impl From<ParquetError> for TpcdsWorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}
impl From<DataFusionError> for TpcdsWorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}
