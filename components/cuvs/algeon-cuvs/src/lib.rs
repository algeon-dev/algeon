//! Runtime-bound safe Rust integration for the scoped cuVS surface used by
//! Algeon.

mod brute_force;
mod device;
mod dlpack;
mod error;
mod ivf_flat;
mod kmeans;
mod pairwise;
mod pca;
mod prepared;
mod resources;
mod select_k;

pub use brute_force::{BruteForceIndex, DistanceMetric, KnnSearchOutput};
pub use device::{
    DeviceBuffer, DeviceElement, DeviceTensor, DeviceTensorMut, MatrixLayout, OwnedFloat32Matrix, ValidatedDeviceTensor,
};
pub use error::{Error, ErrorKind, ErrorStatus, Result};
pub use ivf_flat::{IvfFlatIndex, IvfFlatIndexParams, IvfFlatSearchParams};
pub use kmeans::{KMeansFit, KMeansInit, KMeansMetric, KMeansModel, KMeansParams, KMeansPrediction};
pub use pairwise::pairwise_distance_into;
pub use pca::{PcaFitTransform, PcaOutputBuffers, PcaParams, PcaSolver, fit_transform_owned};
pub use prepared::{PreparedReadCompletion, PreparedTensorLease, PreparedValidatedTensorLease};
pub use resources::CuvsResources;
pub use select_k::select_k_into;
