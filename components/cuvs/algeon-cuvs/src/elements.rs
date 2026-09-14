use crate::device::DeviceElement;

mod sealed {
    pub trait Cagra {}
    pub trait Ivf {}
    pub trait Pairwise {}
    pub trait KMeans {}

    impl Cagra for f32 {}
    impl Cagra for i8 {}
    impl Cagra for u8 {}
    impl Ivf for f32 {}
    impl Ivf for i8 {}
    impl Ivf for u8 {}
    impl Pairwise for f32 {}
    impl Pairwise for f64 {}
    impl KMeans for f32 {}
    impl KMeans for f64 {}
}

pub trait CagraElement: DeviceElement + sealed::Cagra {}
impl<T: DeviceElement + sealed::Cagra> CagraElement for T {}

pub trait IvfElement: DeviceElement + sealed::Ivf {}
impl<T: DeviceElement + sealed::Ivf> IvfElement for T {}

pub trait PairwiseElement: DeviceElement + sealed::Pairwise {
    type Distance: DeviceElement;
}
impl PairwiseElement for f32 {
    type Distance = f32;
}
impl PairwiseElement for f64 {
    type Distance = f64;
}

pub trait KMeansElement: DeviceElement + sealed::KMeans {}
impl<T: DeviceElement + sealed::KMeans> KMeansElement for T {}
