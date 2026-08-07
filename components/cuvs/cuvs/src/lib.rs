//! Thin safe Rust wrapper for a scoped subset of the cuVS C API.
//!
//! This crate intentionally binds only the wave-1 surface needed by the
//! future datafusion-nexus integration: resources, brute-force kNN, KMeans,
//! PCA fit-transform, and borrowed DLPack tensor adapters.

mod brute_force;
mod cuda;
mod device;
mod dlpack;
mod error;
mod kmeans;
mod pca;
mod resources;

pub use brute_force::{BruteForceIndex, DistanceMetric, KnnSearchResult};
pub use cuda::{CudaStream, device_count, synchronize_device};
pub use device::{DeviceBuffer, DeviceMatrixF32, DeviceMatrixViewF32, MatrixLayout};
pub use dlpack::DlpackTensor;
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use kmeans::{KMeansFit, KMeansInit, KMeansModel, KMeansParams, KMeansPrediction};
pub use pca::{PcaFitTransform, PcaParams};
pub use resources::Resources;
