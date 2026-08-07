#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::{c_char, c_float, c_int, c_uint, c_void};

pub type cuvsResources_t = usize;
pub type cudaStream_t = *mut c_void;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvsError_t {
    CUVS_ERROR = 0,
    CUVS_SUCCESS = 1,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(self_: *mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvsDistanceType {
    L2Expanded = 0,
    L2SqrtExpanded = 1,
    CosineExpanded = 2,
    L1 = 3,
    L2Unexpanded = 4,
    L2SqrtUnexpanded = 5,
    InnerProduct = 6,
    Linf = 7,
    Canberra = 8,
    LpUnexpanded = 9,
    CorrelationExpanded = 10,
    JaccardExpanded = 11,
    HellingerExpanded = 12,
    Haversine = 13,
    BrayCurtis = 14,
    JensenShannon = 15,
    HammingUnexpanded = 16,
    KLDivergence = 17,
    RusselRaoExpanded = 18,
    DiceExpanded = 19,
    BitwiseHamming = 20,
    Precomputed = 100,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvsFilterType {
    NO_FILTER = 0,
    BITSET = 1,
    BITMAP = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cuvsFilter {
    pub addr: usize,
    pub type_: cuvsFilterType,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cuvsBruteForceIndex {
    pub addr: usize,
    pub dtype: DLDataType,
}

pub type cuvsBruteForceIndex_t = *mut cuvsBruteForceIndex;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvsKMeansInitMethod {
    KMeansPlusPlus = 0,
    Random = 1,
    Array = 2,
}

#[repr(C)]
#[derive(Debug)]
pub struct cuvsKMeansParams {
    pub metric: cuvsDistanceType,
    pub n_clusters: c_int,
    pub init: cuvsKMeansInitMethod,
    pub max_iter: c_int,
    pub tol: f64,
    pub n_init: c_int,
    pub oversampling_factor: f64,
    pub batch_samples: c_int,
    pub batch_centroids: c_int,
    pub inertia_check: bool,
    pub hierarchical: bool,
    pub hierarchical_n_iters: c_int,
    pub streaming_batch_size: i64,
    pub init_size: i64,
}

pub type cuvsKMeansParams_t = *mut cuvsKMeansParams;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cuvsPcaSolver {
    CUVS_PCA_COV_EIG_DQ = 0,
    CUVS_PCA_COV_EIG_JACOBI = 1,
}

#[repr(C)]
#[derive(Debug)]
pub struct cuvsPcaParams {
    pub n_components: c_int,
    pub copy: bool,
    pub whiten: bool,
    pub algorithm: cuvsPcaSolver,
    pub tol: c_float,
    pub n_iterations: c_int,
}

pub type cuvsPcaParams_t = *mut cuvsPcaParams;

unsafe extern "C" {
    pub fn cuvsGetLastErrorText() -> *const c_char;
    pub fn cuvsSetLastErrorText(error: *const c_char);

    pub fn cuvsResourcesCreate(res: *mut cuvsResources_t) -> cuvsError_t;
    pub fn cuvsResourcesDestroy(res: cuvsResources_t) -> cuvsError_t;
    pub fn cuvsStreamSet(res: cuvsResources_t, stream: cudaStream_t) -> cuvsError_t;

    pub fn cuvsBruteForceIndexCreate(index: *mut cuvsBruteForceIndex_t) -> cuvsError_t;
    pub fn cuvsBruteForceIndexDestroy(index: cuvsBruteForceIndex_t) -> cuvsError_t;
    pub fn cuvsBruteForceBuild(
        res: cuvsResources_t,
        dataset: *mut DLManagedTensor,
        metric: cuvsDistanceType,
        metric_arg: c_float,
        index: cuvsBruteForceIndex_t,
    ) -> cuvsError_t;
    pub fn cuvsBruteForceSearch(
        res: cuvsResources_t,
        index: cuvsBruteForceIndex_t,
        queries: *mut DLManagedTensor,
        neighbors: *mut DLManagedTensor,
        distances: *mut DLManagedTensor,
        prefilter: cuvsFilter,
    ) -> cuvsError_t;

    pub fn cuvsKMeansParamsCreate(params: *mut cuvsKMeansParams_t) -> cuvsError_t;
    pub fn cuvsKMeansParamsDestroy(params: cuvsKMeansParams_t) -> cuvsError_t;
    pub fn cuvsKMeansFit(
        res: cuvsResources_t,
        params: cuvsKMeansParams_t,
        X: *mut DLManagedTensor,
        sample_weight: *mut DLManagedTensor,
        centroids: *mut DLManagedTensor,
        inertia: *mut f64,
        n_iter: *mut c_int,
    ) -> cuvsError_t;
    pub fn cuvsKMeansPredict(
        res: cuvsResources_t,
        params: cuvsKMeansParams_t,
        X: *mut DLManagedTensor,
        sample_weight: *mut DLManagedTensor,
        centroids: *mut DLManagedTensor,
        labels: *mut DLManagedTensor,
        normalize_weight: bool,
        inertia: *mut f64,
    ) -> cuvsError_t;

    pub fn cuvsPcaParamsCreate(params: *mut cuvsPcaParams_t) -> cuvsError_t;
    pub fn cuvsPcaParamsDestroy(params: cuvsPcaParams_t) -> cuvsError_t;
    pub fn cuvsPcaFitTransform(
        res: cuvsResources_t,
        params: cuvsPcaParams_t,
        input: *mut DLManagedTensor,
        trans_input: *mut DLManagedTensor,
        components: *mut DLManagedTensor,
        explained_var: *mut DLManagedTensor,
        explained_var_ratio: *mut DLManagedTensor,
        singular_vals: *mut DLManagedTensor,
        mu: *mut DLManagedTensor,
        noise_vars: *mut DLManagedTensor,
        flip_signs_based_on_U: bool,
    ) -> cuvsError_t;
}

pub type cudaError_t = c_int;
pub type cudaMemcpyKind = c_uint;

pub const cudaSuccess: cudaError_t = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;

unsafe extern "C" {
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: cudaMemcpyKind) -> cudaError_t;
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}
