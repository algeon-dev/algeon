//! Runtime-bound safe Rust integration for the scoped cuVS surface used by
//! DataFusion Nexus.

mod brute_force;
mod device;
mod dlpack;
mod error;
mod kmeans;
mod pca;
mod prepared;
mod resources;

pub use brute_force::{BruteForceIndex, DistanceMetric, KnnSearchOutput};
pub use device::{
    DeviceBuffer, DeviceElement, DeviceTensor, DeviceTensorMut, MatrixLayout, OwnedFloat32Matrix, ValidatedDeviceTensor,
};
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use kmeans::{KMeansFit, KMeansInit, KMeansMetric, KMeansModel, KMeansParams, KMeansPrediction};
pub use pca::{PcaFitTransform, PcaOutputBuffers, PcaParams, PcaSolver, fit_transform_owned};
pub use prepared::{PreparedReadCompletion, PreparedTensorLease, PreparedValidatedTensorLease};
pub use resources::CuvsResources;
