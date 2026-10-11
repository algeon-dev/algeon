use static_assertions::{assert_impl_all, assert_not_impl_any};

assert_not_impl_any!(cuvs::CuvsResources: Send, Sync);
assert_not_impl_any!(cuvs::BruteForceIndex<'static>: Send, Sync);
assert_not_impl_any!(cuvs::KMeansModel<f32>: Send, Sync);
assert_not_impl_any!(cuvs::IvfFlatIndex<f32>: Send, Sync);
assert_not_impl_any!(cuvs::IvfPqIndex<f32>: Send, Sync);
assert_not_impl_any!(cuvs::PreparedTensorLease<'static, 'static, 'static, f32>: Send, Sync);
assert_not_impl_any!(cuvs::PreparedValidatedTensorLease<'static, 'static, 'static, f32>: Send, Sync);
assert_impl_all!(f32: cuvs::CagraElement, cuvs::IvfElement, cuvs::PairwiseElement, cuvs::KMeansElement);
assert_impl_all!(f64: cuvs::PairwiseElement, cuvs::KMeansElement);
assert_impl_all!(i8: cuvs::CagraElement, cuvs::IvfElement);
assert_impl_all!(u8: cuvs::CagraElement, cuvs::IvfElement);
assert_not_impl_any!(f64: cuvs::CagraElement, cuvs::IvfElement);
assert_not_impl_any!(i8: cuvs::PairwiseElement, cuvs::KMeansElement);
assert_not_impl_any!(u8: cuvs::PairwiseElement, cuvs::KMeansElement);
