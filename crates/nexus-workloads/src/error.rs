use std::{
    error::Error as StdError,
    fmt::{Display, Formatter},
    path::PathBuf,
};

use arrow::error::ArrowError;
use datafusion::error::DataFusionError;
use parquet::errors::ParquetError;

/// Single shared workload error for all dataset discovery and registration
/// paths. Hand-rolled Display/Error/From per house style.
#[derive(Debug)]
pub enum WorkloadError {
    Io(std::io::Error),
    Arrow(ArrowError),
    Parquet(ParquetError),
    DataFusion(DataFusionError),
    NonUtf8Path(PathBuf),
    InvalidDataset(String),
}

impl Display for WorkloadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Arrow(error) => write!(f, "Arrow error: {error}"),
            Self::Parquet(error) => write!(f, "Parquet error: {error}"),
            Self::DataFusion(error) => write!(f, "DataFusion error: {error}"),
            Self::NonUtf8Path(path) => write!(f, "path is not valid UTF-8: {}", path.display()),
            Self::InvalidDataset(message) => write!(f, "invalid dataset: {message}"),
        }
    }
}

impl StdError for WorkloadError {
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

impl From<std::io::Error> for WorkloadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<ArrowError> for WorkloadError {
    fn from(value: ArrowError) -> Self {
        Self::Arrow(value)
    }
}
impl From<ParquetError> for WorkloadError {
    fn from(value: ParquetError) -> Self {
        Self::Parquet(value)
    }
}
impl From<DataFusionError> for WorkloadError {
    fn from(value: DataFusionError) -> Self {
        Self::DataFusion(value)
    }
}

pub type WorkloadResult<T> = std::result::Result<T, WorkloadError>;
