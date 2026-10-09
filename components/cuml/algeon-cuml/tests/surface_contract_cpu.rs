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
