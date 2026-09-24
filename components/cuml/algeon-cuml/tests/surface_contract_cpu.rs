use cuml::{
    CumlResources, DeviceBuffer, DeviceMatrix, LogisticRegressionModel, OwnedFloat32Matrix, PcaModel,
    PreparedMatrixLease, RandomForestClassifier, RandomForestRegressor, TrainingStreams, TsvdModel, UmapModel,
};
use static_assertions::assert_not_impl_any;

assert_not_impl_any!(CumlResources: Send, Sync);
assert_not_impl_any!(TrainingStreams: Send, Sync);
assert_not_impl_any!(DeviceBuffer<f32>: Send, Sync);
assert_not_impl_any!(DeviceMatrix<'static>: Send, Sync);
assert_not_impl_any!(OwnedFloat32Matrix: Send, Sync);
assert_not_impl_any!(PreparedMatrixLease<'static, 'static, 'static>: Send, Sync);
assert_not_impl_any!(PcaModel: Send, Sync);
assert_not_impl_any!(TsvdModel: Send, Sync);
assert_not_impl_any!(LogisticRegressionModel: Send, Sync);
assert_not_impl_any!(RandomForestClassifier: Send, Sync);
assert_not_impl_any!(RandomForestRegressor: Send, Sync);
assert_not_impl_any!(UmapModel: Send, Sync);

#[test]
fn public_error_surface_exposes_structured_classification_and_attribution() {
    let _: fn(&CumlResources, cudf::PackedMatrix) -> cuml::Result<OwnedFloat32Matrix> = OwnedFloat32Matrix::from_packed;
    let _: fn(&cuml::Error) -> cudf::CudfAllocationAttribution = cuml::allocation_attribution;
    let _: fn(&cuml::Error) -> cuml::ErrorCode = cuml::Error::code;
    let _: fn(&cuml::Error) -> cuml::ErrorKind = cuml::Error::kind;
    let _: fn(&cuml::Error) -> cuml::ErrorStatus = cuml::Error::status;
    let _: fn(&cuml::Error) -> cuml::FailurePhase = cuml::Error::phase;
}
