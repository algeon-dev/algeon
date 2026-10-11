use cudf::runtime::ExecutionBindingId;

use super::{column_major, dimensions, native, validate_model_input};
use crate::{
    DeviceBuffer, DeviceMatrix, ErrorCode, FailurePhase, OwnedFloat32Matrix, Result,
    error::{error, invalid_input},
    resources::CumlResources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PcaSolver {
    CovEigDq = 0,
    CovEigJacobi = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct PcaParams {
    pub n_components: usize,
    pub solver: PcaSolver,
    pub tolerance: f32,
    pub iterations: u32,
    pub copy: bool,
    pub whiten: bool,
}
impl PcaParams {
    pub fn new(n_components: usize) -> Self {
        Self { n_components, solver: PcaSolver::CovEigDq, tolerance: 0.0, iterations: 15, copy: true, whiten: false }
    }
    fn validate(self, op: &'static str) -> Result<()> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 || self.iterations == 0 {
            return Err(invalid_input(op, "tolerance must be finite and nonnegative and iterations nonzero"));
        }
        Ok(())
    }
}

pub struct PcaModel {
    components: DeviceBuffer<f32>,
    singular_values: DeviceBuffer<f32>,
    mean: DeviceBuffer<f32>,
    explained_variance: DeviceBuffer<f32>,
    explained_variance_ratio: DeviceBuffer<f32>,
    noise_variance: DeviceBuffer<f32>,
    n_features: usize,
    n_components: usize,
    binding_id: ExecutionBindingId,
    whiten: bool,
}
impl PcaModel {
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn n_components(&self) -> usize {
        self.n_components
    }
    pub fn components(&self) -> &DeviceBuffer<f32> {
        &self.components
    }
    pub fn singular_values(&self) -> &DeviceBuffer<f32> {
        &self.singular_values
    }
    pub fn explained_variance(&self) -> &DeviceBuffer<f32> {
        &self.explained_variance
    }
    pub fn explained_variance_ratio(&self) -> &DeviceBuffer<f32> {
        &self.explained_variance_ratio
    }
    pub fn mean(&self) -> &DeviceBuffer<f32> {
        &self.mean
    }
    pub fn noise_variance(&self) -> &DeviceBuffer<f32> {
        &self.noise_variance
    }

    pub fn fit(resources: &CumlResources, input: &mut OwnedFloat32Matrix, params: PcaParams) -> Result<Self> {
        Self::fit_impl(resources, input, params, None).map(|(model, _)| model)
    }
    pub fn fit_transform(
        resources: &CumlResources,
        input: &mut OwnedFloat32Matrix,
        params: PcaParams,
    ) -> Result<(Self, DeviceBuffer<f32>)> {
        let (model, output) = Self::fit_impl(resources, input, params, Some(()))?;
        Ok((model, output.expect("requested PCA transform output")))
    }
    fn fit_impl(
        resources: &CumlResources,
        input: &mut OwnedFloat32Matrix,
        params: PcaParams,
        transform: Option<()>,
    ) -> Result<(Self, Option<DeviceBuffer<f32>>)> {
        const OP: &str = "PcaModel::fit";
        params.validate(OP)?;
        dimensions(input.rows(), input.columns(), params.n_components, OP)?;
        column_major(input.layout(), OP)?;
        if input.binding_id() != resources.binding_id() {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                OP,
                "input and resources use different execution bindings",
            ));
        }
        resources.validate_exact_binding(OP)?;
        let mut components = DeviceBuffer::new(resources, params.n_components * input.columns())?;
        let mut explained = DeviceBuffer::new(resources, params.n_components)?;
        let mut ratio = DeviceBuffer::new(resources, params.n_components)?;
        let mut singular = DeviceBuffer::new(resources, params.n_components)?;
        let mut mean = DeviceBuffer::new(resources, input.columns())?;
        let mut noise = DeviceBuffer::new(resources, 1)?;
        let mut transformed =
            transform.map(|_| DeviceBuffer::new(resources, input.rows() * params.n_components)).transpose()?;
        let input_ptr = input.as_mut_ptr()?.cast();
        let components_ptr = components.as_mut_ptr()?.cast();
        let explained_ptr = explained.as_mut_ptr()?.cast();
        let ratio_ptr = ratio.as_mut_ptr()?.cast();
        let singular_ptr = singular.as_mut_ptr()?.cast();
        let mean_ptr = mean.as_mut_ptr()?.cast();
        let noise_ptr = noise.as_mut_ptr()?.cast();
        let rows = input.rows() as i64;
        let columns = input.columns() as i64;
        let count = params.n_components as i64;
        let algorithm = params.solver as i32;
        let tolerance = params.tolerance;
        let iterations = params.iterations;
        let copy = i32::from(params.copy);
        let whiten = i32::from(params.whiten);
        native(resources, OP, |handle| unsafe {
            if let Some(output) = transformed.as_mut() {
                let output_ptr = output.as_mut_ptr().expect("new output buffer").cast();
                super::cuml_pca_fit_transform(
                    handle,
                    input_ptr,
                    rows,
                    columns,
                    count,
                    algorithm,
                    tolerance,
                    iterations,
                    copy,
                    whiten,
                    output_ptr,
                    components_ptr,
                    explained_ptr,
                    ratio_ptr,
                    singular_ptr,
                    mean_ptr,
                    noise_ptr,
                )
            } else {
                super::cuml_pca_fit(
                    handle,
                    input_ptr,
                    rows,
                    columns,
                    count,
                    algorithm,
                    tolerance,
                    iterations,
                    copy,
                    whiten,
                    components_ptr,
                    explained_ptr,
                    ratio_ptr,
                    singular_ptr,
                    mean_ptr,
                    noise_ptr,
                )
            }
        })?;
        components.validate_finite(OP)?;
        explained.validate_finite(OP)?;
        ratio.validate_finite(OP)?;
        singular.validate_finite(OP)?;
        mean.validate_finite(OP)?;
        noise.validate_finite(OP)?;
        if let Some(output) = transformed.as_mut() {
            output.validate_finite(OP)?;
        }
        let model = Self {
            components,
            singular_values: singular,
            mean,
            explained_variance: explained,
            explained_variance_ratio: ratio,
            noise_variance: noise,
            n_features: input.columns(),
            n_components: params.n_components,
            binding_id: resources.binding_id(),
            whiten: params.whiten,
        };
        Ok((model, transformed))
    }
    /// Preserves the borrowed input by copying it before native mean-centering.
    pub fn transform(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<DeviceBuffer<f32>> {
        const OP: &str = "PcaModel::transform";
        validate_model_input(
            resources,
            input,
            self.binding_id,
            self.n_features,
            self.n_components,
            self.n_features,
            OP,
        )?;
        let mut output = DeviceBuffer::new(
            resources,
            input
                .rows()
                .checked_mul(self.n_components)
                .ok_or_else(|| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, OP, "output shape overflow"))?,
        )?;
        native(resources, OP, |handle| unsafe {
            super::cuml_pca_transform(
                handle,
                input.as_ptr().cast(),
                input.rows() as i64,
                input.columns() as i64,
                self.n_components as i64,
                i32::from(self.whiten),
                self.components.as_ptr().expect("model components").cast(),
                self.singular_values.as_ptr().expect("model singular values").cast(),
                self.mean.as_ptr().expect("model mean").cast(),
                output.as_mut_ptr().expect("output").cast(),
            )
        })?;
        output.validate_finite(OP)?;
        Ok(output)
    }
    pub fn inverse_transform(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<DeviceBuffer<f32>> {
        const OP: &str = "PcaModel::inverse_transform";
        validate_model_input(
            resources,
            input,
            self.binding_id,
            self.n_features,
            self.n_components,
            self.n_components,
            OP,
        )?;
        let mut output = DeviceBuffer::new(
            resources,
            input
                .rows()
                .checked_mul(self.n_features)
                .ok_or_else(|| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, OP, "output shape overflow"))?,
        )?;
        native(resources, OP, |handle| unsafe {
            super::cuml_pca_inverse_transform(
                handle,
                input.as_ptr().cast(),
                input.rows() as i64,
                self.n_features as i64,
                self.n_components as i64,
                i32::from(self.whiten),
                self.components.as_ptr().expect("model components").cast(),
                self.singular_values.as_ptr().expect("model singular values").cast(),
                self.mean.as_ptr().expect("model mean").cast(),
                output.as_mut_ptr().expect("output").cast(),
            )
        })?;
        output.validate_finite(OP)?;
        Ok(output)
    }
}
