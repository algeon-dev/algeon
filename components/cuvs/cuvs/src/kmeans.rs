use std::{ptr, ptr::NonNull};

use crate::{
    brute_force::DistanceMetric,
    device::{DeviceBuffer, DeviceMatrixViewF32, MatrixLayout},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, invalid_state, require_nonzero},
    resources::Resources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInit {
    KMeansPlusPlus,
    Random,
    Array,
}

impl KMeansInit {
    fn raw(self) -> cuvs_sys::cuvsKMeansInitMethod {
        match self {
            KMeansInit::KMeansPlusPlus => cuvs_sys::cuvsKMeansInitMethod::KMeansPlusPlus,
            KMeansInit::Random => cuvs_sys::cuvsKMeansInitMethod::Random,
            KMeansInit::Array => cuvs_sys::cuvsKMeansInitMethod::Array,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansParams {
    pub n_clusters: usize,
    pub max_iter: i32,
    pub tol: f64,
    pub n_init: i32,
    pub init: KMeansInit,
    pub metric: DistanceMetric,
}

impl KMeansParams {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 100,
            tol: 1e-4,
            n_init: 1,
            init: KMeansInit::KMeansPlusPlus,
            metric: DistanceMetric::L2Expanded,
        }
    }
}

#[derive(Debug)]
pub struct KMeansFit {
    pub inertia: f64,
    pub n_iter: i32,
}

#[derive(Debug, Clone)]
pub struct KMeansPrediction {
    pub labels: Vec<i32>,
    pub inertia: f64,
}

#[derive(Debug)]
pub struct KMeansModel {
    centroids: DeviceBuffer<f32>,
    n_clusters: usize,
    n_features: usize,
    fit: KMeansFit,
}

struct ParamsHandle {
    raw: NonNull<cuvs_sys::cuvsKMeansParams>,
}

impl ParamsHandle {
    fn new(params: &KMeansParams, operation: &'static str) -> Result<Self> {
        require_nonzero(params.n_clusters, operation, "n_clusters")?;
        if params.max_iter <= 0 {
            return Err(invalid_input(operation, "max_iter must be greater than zero"));
        }
        if params.n_init <= 0 {
            return Err(invalid_input(operation, "n_init must be greater than zero"));
        }

        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(cuvs_sys::cuvsKMeansParamsCreate(&mut raw), operation)? };
        let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned null KMeans params"))?;
        unsafe {
            let p = raw.as_ptr();
            (*p).metric = params.metric.raw();
            (*p).n_clusters = params.n_clusters as i32;
            (*p).init = params.init.raw();
            (*p).max_iter = params.max_iter;
            (*p).tol = params.tol;
            (*p).n_init = params.n_init;
            (*p).hierarchical = false;
            (*p).streaming_batch_size = 0;
        }
        Ok(Self { raw })
    }

    fn as_ptr(&self) -> cuvs_sys::cuvsKMeansParams_t {
        self.raw.as_ptr()
    }
}

impl Drop for ParamsHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = cuvs_sys::cuvsKMeansParamsDestroy(self.raw.as_ptr());
        }
    }
}

impl KMeansModel {
    pub fn fit(
        resources: &Resources,
        data: &DeviceMatrixViewF32<'_>,
        params: &KMeansParams,
        initial_centroids: Option<&[f32]>,
    ) -> Result<Self> {
        let operation = "KMeansModel::fit";
        if data.layout() != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "KMeans input must be row-major"));
        }
        if matches!(params.init, KMeansInit::Array) && initial_centroids.is_none() {
            return Err(invalid_input(operation, "initial_centroids must be provided when init is KMeansInit::Array"));
        }
        require_nonzero(data.rows(), operation, "rows")?;
        require_nonzero(data.cols(), operation, "cols")?;
        require_nonzero(params.n_clusters, operation, "n_clusters")?;
        if params.n_clusters > data.rows() {
            return Err(invalid_input(
                operation,
                format!("n_clusters ({}) must be <= rows ({})", params.n_clusters, data.rows()),
            ));
        }

        let params_handle = ParamsHandle::new(params, operation)?;
        let centroids = if let Some(initial) = initial_centroids {
            let expected = params.n_clusters * data.cols();
            if initial.len() != expected {
                return Err(invalid_input(
                    operation,
                    format!(
                        "initial centroid length {} does not match n_clusters * n_features ({expected})",
                        initial.len()
                    ),
                ));
            }
            DeviceBuffer::from_slice(initial)?
        } else {
            DeviceBuffer::new(params.n_clusters * data.cols())?
        };

        let mut data_tensor = data.to_dlpack()?;
        let mut centroids_tensor = DlpackTensor::device_matrix_f32(
            centroids.as_mut_ptr(),
            params.n_clusters,
            data.cols(),
            MatrixLayout::RowMajor,
        )?;
        let mut inertia = 0.0;
        let mut n_iter = 0;
        unsafe {
            check_cuvs(
                cuvs_sys::cuvsKMeansFit(
                    resources.raw(),
                    params_handle.as_ptr(),
                    data_tensor.as_mut_ptr(),
                    ptr::null_mut(),
                    centroids_tensor.as_mut_ptr(),
                    &mut inertia,
                    &mut n_iter,
                ),
                operation,
            )?;
        }

        Ok(Self {
            centroids,
            n_clusters: params.n_clusters,
            n_features: data.cols(),
            fit: KMeansFit { inertia, n_iter },
        })
    }

    pub fn predict(
        &self,
        resources: &Resources,
        data: &DeviceMatrixViewF32<'_>,
        params: &KMeansParams,
    ) -> Result<KMeansPrediction> {
        let operation = "KMeansModel::predict";
        if data.layout() != MatrixLayout::RowMajor {
            return Err(invalid_input(operation, "KMeans input must be row-major"));
        }
        if data.cols() != self.n_features {
            return Err(invalid_input(
                operation,
                format!("input cols {} must match model features {}", data.cols(), self.n_features),
            ));
        }
        if params.n_clusters != self.n_clusters {
            return Err(invalid_input(
                operation,
                format!("params.n_clusters ({}) must match model clusters ({})", params.n_clusters, self.n_clusters),
            ));
        }

        let params_handle = ParamsHandle::new(params, operation)?;
        let labels = DeviceBuffer::<i32>::new(data.rows())?;
        let mut data_tensor = data.to_dlpack()?;
        let mut centroids_tensor = DlpackTensor::device_matrix_f32(
            self.centroids.as_mut_ptr(),
            self.n_clusters,
            self.n_features,
            MatrixLayout::RowMajor,
        )?;
        let mut labels_tensor = DlpackTensor::device_vector_i32(labels.as_mut_ptr(), data.rows());
        let mut inertia = 0.0;

        unsafe {
            check_cuvs(
                cuvs_sys::cuvsKMeansPredict(
                    resources.raw(),
                    params_handle.as_ptr(),
                    data_tensor.as_mut_ptr(),
                    ptr::null_mut(),
                    centroids_tensor.as_mut_ptr(),
                    labels_tensor.as_mut_ptr(),
                    false,
                    &mut inertia,
                ),
                operation,
            )?;
        }

        Ok(KMeansPrediction { labels: labels.to_vec()?, inertia })
    }

    pub fn fit_result(&self) -> &KMeansFit {
        &self.fit
    }

    pub fn centroids_to_host(&self) -> Result<Vec<f32>> {
        self.centroids.to_vec()
    }
}
