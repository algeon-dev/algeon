use cudf::runtime::ExecutionBindingId;

use super::{column_major, dimensions, native, pca::PcaSolver, validate_model_input};
use crate::{
    DeviceBuffer, DeviceMatrix, ErrorCode, FailurePhase, OwnedFloat32Matrix, Result,
    error::{error, invalid_input},
    resources::CumlResources,
};

#[derive(Debug, Clone, Copy)]
pub struct TsvdParams {
    pub n_components: usize,
    pub solver: PcaSolver,
    pub tolerance: f32,
    pub iterations: u32,
}

impl TsvdParams {
    pub fn new(n_components: usize) -> Self {
        Self { n_components, solver: PcaSolver::CovEigDq, tolerance: 0.0, iterations: 15 }
    }
    fn validate(self, op: &'static str) -> Result<()> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 || self.iterations == 0 {
            return Err(invalid_input(op, "tolerance must be finite and nonnegative and iterations nonzero"));
        }
        Ok(())
    }
}

pub struct TsvdModel {
    components: DeviceBuffer<f32>,
    singular_values: DeviceBuffer<f32>,
    explained_variance: DeviceBuffer<f32>,
    explained_variance_ratio: DeviceBuffer<f32>,
    n_features: usize,
    n_components: usize,
    binding_id: ExecutionBindingId,
}
impl TsvdModel {
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

    /// Fits the model and computes the transformed scores needed for explained variance.
    /// The temporary score matrix has `rows * n_components` elements even when it is not returned.
    pub fn fit(resources: &CumlResources, input: &mut OwnedFloat32Matrix, params: TsvdParams) -> Result<Self> {
        Self::fit_impl(resources, input, params).map(|(model, _)| model)
    }
    pub fn fit_transform(
        resources: &CumlResources,
        input: &mut OwnedFloat32Matrix,
        params: TsvdParams,
    ) -> Result<(Self, DeviceBuffer<f32>)> {
        Self::fit_impl(resources, input, params)
    }
    fn fit_impl(
        resources: &CumlResources,
        input: &mut OwnedFloat32Matrix,
        params: TsvdParams,
    ) -> Result<(Self, DeviceBuffer<f32>)> {
        const OP: &str = "TsvdModel::fit";
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
        let mut singular = DeviceBuffer::new(resources, params.n_components)?;
        let mut explained = DeviceBuffer::new(resources, params.n_components)?;
        let mut ratio = DeviceBuffer::new(resources, params.n_components)?;
        let mut transformed = DeviceBuffer::new(resources, input.rows() * params.n_components)?;
        let input_ptr = input.as_mut_ptr()?.cast();
        let components_ptr = components.as_mut_ptr()?.cast();
        let singular_ptr = singular.as_mut_ptr()?.cast();
        let explained_ptr = explained.as_mut_ptr()?.cast();
        let ratio_ptr = ratio.as_mut_ptr()?.cast();
        let rows = input.rows() as i64;
        let columns = input.columns() as i64;
        let count = params.n_components as i64;
        let algorithm = params.solver as i32;
        let tolerance = params.tolerance;
        let iterations = params.iterations;
        native(resources, OP, |handle| unsafe {
            super::cuml_tsvd_fit_transform(
                handle,
                input_ptr,
                rows,
                columns,
                count,
                algorithm,
                tolerance,
                iterations,
                transformed.as_mut_ptr().expect("new output buffer").cast(),
                components_ptr,
                explained_ptr,
                ratio_ptr,
                singular_ptr,
            )
        })?;
        components.validate_finite(OP)?;
        singular.validate_finite(OP)?;
        explained.validate_finite(OP)?;
        ratio.validate_finite(OP)?;
        transformed.validate_finite(OP)?;
        let model = Self {
            components,
            singular_values: singular,
            explained_variance: explained,
            explained_variance_ratio: ratio,
            n_features: input.columns(),
            n_components: params.n_components,
            binding_id: resources.binding_id(),
        };
        Ok((model, transformed))
    }
    pub fn transform(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<DeviceBuffer<f32>> {
        const OP: &str = "TsvdModel::transform";
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
            super::cuml_tsvd_transform(
                handle,
                input.as_ptr().cast(),
                input.rows() as i64,
                self.n_features as i64,
                self.n_components as i64,
                self.components.as_ptr().expect("model components").cast(),
                output.as_mut_ptr().expect("output").cast(),
            )
        })?;
        output.validate_finite(OP)?;
        Ok(output)
    }
    pub fn inverse_transform(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<DeviceBuffer<f32>> {
        const OP: &str = "TsvdModel::inverse_transform";
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
            super::cuml_tsvd_inverse_transform(
                handle,
                input.as_ptr().cast(),
                input.rows() as i64,
                self.n_features as i64,
                self.n_components as i64,
                self.components.as_ptr().expect("model components").cast(),
                output.as_mut_ptr().expect("output").cast(),
            )
        })?;
        output.validate_finite(OP)?;
        Ok(output)
    }
}
