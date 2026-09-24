//! Safe Rust integration with NVIDIA cuML.

mod decomposition;
mod device;
mod error;
mod hdbscan;
mod labels;
mod logistic;
mod prepared;
mod random_forest;
mod resources;
mod umap;

pub use decomposition::{PcaModel, PcaParams, PcaSolver, TsvdModel, TsvdParams};
pub use device::{DeviceBuffer, DeviceElement, DeviceMatrix, MatrixLayout, OwnedFloat32Matrix};
pub use error::{
    CumlErrorDetail, CumlErrorFailureSource, Error, ErrorCode, ErrorKind, ErrorStatus, FailurePhase, Result,
    allocation_attribution,
};
pub use hdbscan::{ClusterSelectionMethod, HdbscanFit, HdbscanOutputBuffers, HdbscanParams};
pub use logistic::{LogisticRegressionFit, LogisticRegressionModel, LogisticRegressionParams};
pub use prepared::{PreparedMatrixLease, PreparedReadCompletion};
pub use random_forest::{RandomForestClassifier, RandomForestParams, RandomForestRegressor, SplitCriterion};
pub use resources::{CumlResources, TrainingStreams};
pub use umap::{UmapInit, UmapModel, UmapParams};

#[cfg(test)]
mod surface_tests {
    use static_assertions::assert_not_impl_any;

    use super::{CumlResources, ErrorCode, ErrorKind, ErrorStatus, OwnedFloat32Matrix, TrainingStreams};

    assert_not_impl_any!(CumlResources: Send, Sync);
    assert_not_impl_any!(OwnedFloat32Matrix: Send, Sync);
    assert_not_impl_any!(TrainingStreams: Send, Sync);

    #[test]
    fn error_taxonomy_projects_leaf_codes() {
        let oom = super::error::error(ErrorCode::OutOfMemory, super::FailurePhase::NativeCall, "test", "oom");
        assert_eq!(oom.kind(), ErrorKind::ResourceExhausted);
        assert_eq!(oom.status(), ErrorStatus::Temporary);
        let fatal = super::error::error(ErrorCode::FatalCuda, super::FailurePhase::NativeCall, "test", "cuda");
        assert_eq!(fatal.kind(), ErrorKind::DependencyFailure);
        assert_eq!(fatal.status(), ErrorStatus::Persistent);
    }
}
