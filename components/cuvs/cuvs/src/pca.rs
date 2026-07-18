use std::{ptr, ptr::NonNull};

use crate::{
    device::{DeviceBuffer, DeviceMatrixViewF32, MatrixLayout},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, invalid_state, require_nonzero},
    resources::Resources,
};

#[derive(Debug, Clone)]
pub struct PcaParams {
    pub n_components: usize,
    pub copy: bool,
    pub whiten: bool,
}

impl PcaParams {
    pub fn new(n_components: usize) -> Self {
        Self { n_components, copy: true, whiten: false }
    }
}

#[derive(Debug)]
pub struct PcaFitTransform {
    transformed: DeviceBuffer<f32>,
    rows: usize,
    n_components: usize,
    components: DeviceBuffer<f32>,
    explained_variance: Vec<f32>,
    explained_variance_ratio: Vec<f32>,
    singular_values: Vec<f32>,
    mean: Vec<f32>,
    noise_variance: f32,
}

struct ParamsHandle {
    raw: NonNull<cuvs_sys::cuvsPcaParams>,
}

impl ParamsHandle {
    fn new(params: &PcaParams, operation: &'static str) -> Result<Self> {
        require_nonzero(params.n_components, operation, "n_components")?;
        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(cuvs_sys::cuvsPcaParamsCreate(&mut raw), operation)? };
        let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned null PCA params"))?;
        unsafe {
            let p = raw.as_ptr();
            (*p).n_components = params.n_components as i32;
            (*p).copy = params.copy;
            (*p).whiten = params.whiten;
        }
        Ok(Self { raw })
    }

    fn as_ptr(&self) -> cuvs_sys::cuvsPcaParams_t {
        self.raw.as_ptr()
    }
}

impl Drop for ParamsHandle {
    fn drop(&mut self) {
        unsafe {
            let _ = cuvs_sys::cuvsPcaParamsDestroy(self.raw.as_ptr());
        }
    }
}

impl PcaFitTransform {
    pub fn fit_transform(resources: &Resources, input: &DeviceMatrixViewF32<'_>, params: &PcaParams) -> Result<Self> {
        let operation = "PcaFitTransform::fit_transform";
        if input.layout() != MatrixLayout::ColumnMajor {
            return Err(invalid_input(operation, "PCA input must be column-major"));
        }
        require_nonzero(input.rows(), operation, "rows")?;
        require_nonzero(input.cols(), operation, "cols")?;
        require_nonzero(params.n_components, operation, "n_components")?;
        if params.n_components > input.cols() {
            return Err(invalid_input(
                operation,
                format!("n_components ({}) must be <= input cols ({})", params.n_components, input.cols()),
            ));
        }

        let params_handle = ParamsHandle::new(params, operation)?;
        let transformed = DeviceBuffer::<f32>::new(input.rows() * params.n_components)?;
        let components = DeviceBuffer::<f32>::new(params.n_components * input.cols())?;
        let explained_variance = DeviceBuffer::<f32>::new(params.n_components)?;
        let explained_variance_ratio = DeviceBuffer::<f32>::new(params.n_components)?;
        let singular_values = DeviceBuffer::<f32>::new(params.n_components)?;
        let mean = DeviceBuffer::<f32>::new(input.cols())?;
        let noise_variance = DeviceBuffer::<f32>::new(1)?;

        let mut input_tensor = input.to_dlpack()?;
        let mut transformed_tensor = DlpackTensor::device_matrix_f32(
            transformed.as_mut_ptr(),
            input.rows(),
            params.n_components,
            MatrixLayout::ColumnMajor,
        )?;
        let mut components_tensor = DlpackTensor::device_matrix_f32(
            components.as_mut_ptr(),
            params.n_components,
            input.cols(),
            MatrixLayout::ColumnMajor,
        )?;
        let mut explained_variance_tensor =
            DlpackTensor::device_vector_f32(explained_variance.as_mut_ptr(), params.n_components);
        let mut explained_variance_ratio_tensor =
            DlpackTensor::device_vector_f32(explained_variance_ratio.as_mut_ptr(), params.n_components);
        let mut singular_values_tensor =
            DlpackTensor::device_vector_f32(singular_values.as_mut_ptr(), params.n_components);
        let mut mean_tensor = DlpackTensor::device_vector_f32(mean.as_mut_ptr(), input.cols());
        let mut noise_variance_tensor = DlpackTensor::device_vector_f32(noise_variance.as_mut_ptr(), 1);

        unsafe {
            check_cuvs(
                cuvs_sys::cuvsPcaFitTransform(
                    resources.raw(),
                    params_handle.as_ptr(),
                    input_tensor.as_mut_ptr(),
                    transformed_tensor.as_mut_ptr(),
                    components_tensor.as_mut_ptr(),
                    explained_variance_tensor.as_mut_ptr(),
                    explained_variance_ratio_tensor.as_mut_ptr(),
                    singular_values_tensor.as_mut_ptr(),
                    mean_tensor.as_mut_ptr(),
                    noise_variance_tensor.as_mut_ptr(),
                    false,
                ),
                operation,
            )?;
        }

        let explained_variance_host = explained_variance.to_vec()?;
        let explained_variance_ratio_host = explained_variance_ratio.to_vec()?;
        let singular_values_host = singular_values.to_vec()?;
        let mean_host = mean.to_vec()?;
        let noise_variance_host = noise_variance.to_vec()?[0];

        Ok(Self {
            transformed,
            rows: input.rows(),
            n_components: params.n_components,
            components,
            explained_variance: explained_variance_host,
            explained_variance_ratio: explained_variance_ratio_host,
            singular_values: singular_values_host,
            mean: mean_host,
            noise_variance: noise_variance_host,
        })
    }

    pub fn transformed_to_host(&self) -> Result<Vec<f32>> {
        self.transformed.to_vec()
    }

    pub fn components_to_host(&self) -> Result<Vec<f32>> {
        self.components.to_vec()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn n_components(&self) -> usize {
        self.n_components
    }

    pub fn explained_variance(&self) -> &[f32] {
        &self.explained_variance
    }

    pub fn explained_variance_ratio(&self) -> &[f32] {
        &self.explained_variance_ratio
    }

    pub fn singular_values(&self) -> &[f32] {
        &self.singular_values
    }

    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    pub fn noise_variance(&self) -> f32 {
        self.noise_variance
    }
}
