use std::{ptr, ptr::NonNull};

use crate::{
    device::{DeviceBuffer, DeviceTensorMut, MatrixLayout, OwnedFloat32Matrix},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, from_cudf, invalid_input, invalid_state, require_nonzero},
    resources::CuvsResources,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaSolver {
    CovarianceDivideAndConquer,
    CovarianceJacobi,
}

impl PcaSolver {
    fn raw(self) -> cuvs_sys::cuvsPcaSolver {
        match self {
            Self::CovarianceDivideAndConquer => cuvs_sys::cuvsPcaSolver::CUVS_PCA_COV_EIG_DQ,
            Self::CovarianceJacobi => cuvs_sys::cuvsPcaSolver::CUVS_PCA_COV_EIG_JACOBI,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PcaParams {
    pub n_components: usize,
    pub copy: bool,
    pub whiten: bool,
    pub solver: PcaSolver,
    pub tolerance: f32,
    pub iterations: i32,
    pub flip_signs_based_on_u: bool,
}

impl PcaParams {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            copy: true,
            whiten: false,
            solver: PcaSolver::CovarianceDivideAndConquer,
            tolerance: 1e-7,
            iterations: 15,
            flip_signs_based_on_u: false,
        }
    }
}

/// Caller-owned device outputs for one PCA fit-transform operation.
#[derive(Debug)]
pub struct PcaOutputBuffers {
    pub transformed: DeviceBuffer<f32>,
    pub components: DeviceBuffer<f32>,
    pub explained_variance: DeviceBuffer<f32>,
    pub explained_variance_ratio: DeviceBuffer<f32>,
    pub singular_values: DeviceBuffer<f32>,
    pub mean: DeviceBuffer<f32>,
    pub noise_variance: DeviceBuffer<f32>,
}

impl PcaOutputBuffers {
    pub fn new(resources: &CuvsResources, rows: usize, columns: usize, n_components: usize) -> Result<Self> {
        let operation = "PcaOutputBuffers::new";
        require_nonzero(rows, operation, "rows")?;
        require_nonzero(columns, operation, "columns")?;
        require_nonzero(n_components, operation, "n_components")?;
        let transformed_len = rows
            .checked_mul(n_components)
            .ok_or_else(|| invalid_input(operation, "transformed output length overflowed usize"))?;
        let components_len = n_components
            .checked_mul(columns)
            .ok_or_else(|| invalid_input(operation, "components output length overflowed usize"))?;
        Ok(Self {
            transformed: DeviceBuffer::new(resources, transformed_len)?,
            components: DeviceBuffer::new(resources, components_len)?,
            explained_variance: DeviceBuffer::new(resources, n_components)?,
            explained_variance_ratio: DeviceBuffer::new(resources, n_components)?,
            singular_values: DeviceBuffer::new(resources, n_components)?,
            mean: DeviceBuffer::new(resources, columns)?,
            noise_variance: DeviceBuffer::new(resources, 1)?,
        })
    }

    /// Consume the output owner and split the column-major transformed matrix
    /// into independent cudf-owned device columns.
    pub fn into_transformed_columns(self, fit: PcaFitTransform) -> Result<Vec<cudf::Series>> {
        let operation = "PcaOutputBuffers::into_transformed_columns";
        validate_output_lengths(&self, fit.rows, fit.columns, fit.n_components, operation)?;
        if !self.transformed.native_input_validated() {
            return Err(invalid_state(
                operation,
                "transformed PCA output has not passed post-native finite validation",
            ));
        }
        let expected = fit
            .rows
            .checked_mul(fit.n_components)
            .ok_or_else(|| invalid_input(operation, "transformed output length overflowed usize"))?;
        if self.transformed.len() != expected {
            return Err(invalid_input(
                operation,
                format!("transformed output contains {} elements, expected {expected}", self.transformed.len()),
            ));
        }

        let flat =
            cudf::Series::from_column(self.transformed.into_column()?).map_err(|error| from_cudf(operation, error))?;
        (0..fit.n_components)
            .map(|component| flat.slice(component * fit.rows, fit.rows).map_err(|error| from_cudf(operation, error)))
            .collect()
    }

    /// Consume the output owner and return the transformed PCA relation as a
    /// cudf DataFrame with `component_0..N` columns.
    pub fn into_transformed_dataframe(self, fit: PcaFitTransform) -> Result<cudf::DataFrame> {
        let operation = "PcaOutputBuffers::into_transformed_dataframe";
        let columns = self.into_transformed_columns(fit)?;
        cudf::DataFrame::from_series(
            columns.into_iter().enumerate().map(|(component, column)| (format!("component_{component}"), column)),
        )
        .map_err(|error| from_cudf(operation, error))
    }

    fn validate_native_outputs(&mut self, operation: &'static str) -> Result<()> {
        self.transformed.validate_native_input(operation)?;
        self.components.validate_native_input(operation)?;
        self.explained_variance.validate_native_input(operation)?;
        self.explained_variance_ratio.validate_native_input(operation)?;
        self.singular_values.validate_native_input(operation)?;
        self.mean.validate_native_input(operation)?;
        self.noise_variance.validate_native_input(operation)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcaFitTransform {
    rows: usize,
    columns: usize,
    n_components: usize,
}

impl PcaFitTransform {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn n_components(&self) -> usize {
        self.n_components
    }
}

struct ParamsHandle {
    raw: Option<NonNull<cuvs_sys::cuvsPcaParams>>,
}

impl ParamsHandle {
    fn new(params: &PcaParams, operation: &'static str) -> Result<Self> {
        validate_params(params, operation)?;
        let mut raw = ptr::null_mut();
        unsafe { check_cuvs(cuvs_sys::cuvsPcaParamsCreate(&mut raw), operation)? };
        let raw = NonNull::new(raw).ok_or_else(|| invalid_state(operation, "cuVS returned null PCA params"))?;
        unsafe {
            let native = raw.as_ptr();
            (*native).n_components = params.n_components as i32;
            (*native).copy = params.copy;
            (*native).whiten = params.whiten;
            (*native).algorithm = params.solver.raw();
            (*native).tol = params.tolerance;
            (*native).n_iterations = params.iterations;
        }
        Ok(Self { raw: Some(raw) })
    }

    fn as_ptr(&self, operation: &'static str) -> Result<cuvs_sys::cuvsPcaParams_t> {
        self.raw.map(NonNull::as_ptr).ok_or_else(|| invalid_state(operation, "PCA params have been closed"))
    }

    fn close(&mut self, operation: &'static str) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        let result = unsafe { check_cuvs(cuvs_sys::cuvsPcaParamsDestroy(raw.as_ptr()), operation) };
        if result.is_err() {
            self.raw = Some(raw);
        }
        result
    }
}

impl Drop for ParamsHandle {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            unsafe {
                let _ = cuvs_sys::cuvsPcaParamsDestroy(raw.as_ptr());
            }
        }
    }
}

pub(crate) fn fit_transform_into(
    resources: &CuvsResources,
    input: &mut DeviceTensorMut<'_, f32>,
    params: &PcaParams,
    outputs: &mut PcaOutputBuffers,
) -> Result<PcaFitTransform> {
    let operation = "pca::fit_transform_into";
    resources.validate_exact_binding(operation)?;
    if input.binding_id() != resources.binding_id() {
        return Err(invalid_input(operation, "PCA input and cuVS resources must use the same exact runtime binding")
            .with_context("tensor_binding_id", input.binding_id())
            .with_context("resource_binding_id", resources.binding_id()));
    }
    if input.device() != resources.device().ordinal() as i32 {
        return Err(invalid_input(operation, "PCA input and cuVS resources must use the same CUDA device")
            .with_context("tensor_device", input.device())
            .with_context("resource_device", resources.device().ordinal()));
    }
    if matrix_layout_mut(input, operation)? != MatrixLayout::ColumnMajor {
        return Err(invalid_input(operation, "PCA input must be a contiguous mutable column-major matrix"));
    }
    validate_params(params, operation)?;
    let rows = input.rows()?;
    let columns = input.columns()?;
    validate_input_dimensions(rows, columns, params.n_components, operation)?;
    validate_output_lengths(outputs, rows, columns, params.n_components, operation)?;

    let transformed_strides = [1, rows];
    let components_strides = [1, params.n_components];
    let PcaOutputBuffers {
        transformed,
        components,
        explained_variance,
        explained_variance_ratio,
        singular_values,
        mean,
        noise_variance,
    } = outputs;
    let mut transformed_view =
        transformed.as_tensor_mut(resources, &[rows, params.n_components], Some(&transformed_strides), 0)?;
    let mut components_view =
        components.as_tensor_mut(resources, &[params.n_components, columns], Some(&components_strides), 0)?;
    let mut explained_variance_view = explained_variance.as_tensor_mut(resources, &[params.n_components], None, 0)?;
    let mut explained_variance_ratio_view =
        explained_variance_ratio.as_tensor_mut(resources, &[params.n_components], None, 0)?;
    let mut singular_values_view = singular_values.as_tensor_mut(resources, &[params.n_components], None, 0)?;
    let mut mean_view = mean.as_tensor_mut(resources, &[columns], None, 0)?;
    let mut noise_variance_view = noise_variance.as_tensor_mut(resources, &[1], None, 0)?;

    resources.with_native(operation, |native| {
        let mut params_handle = ParamsHandle::new(params, operation)?;
        let mut input_tensor = DlpackTensor::from_mut(input);
        let mut transformed_tensor = DlpackTensor::from_mut(&mut transformed_view);
        let mut components_tensor = DlpackTensor::from_mut(&mut components_view);
        let mut explained_variance_tensor = DlpackTensor::from_mut(&mut explained_variance_view);
        let mut explained_variance_ratio_tensor = DlpackTensor::from_mut(&mut explained_variance_ratio_view);
        let mut singular_values_tensor = DlpackTensor::from_mut(&mut singular_values_view);
        let mut mean_tensor = DlpackTensor::from_mut(&mut mean_view);
        let mut noise_variance_tensor = DlpackTensor::from_mut(&mut noise_variance_view);
        let call_result = unsafe {
            check_cuvs(
                cuvs_sys::cuvsPcaFitTransform(
                    native,
                    params_handle.as_ptr(operation)?,
                    input_tensor.as_mut_ptr(),
                    transformed_tensor.as_mut_ptr(),
                    components_tensor.as_mut_ptr(),
                    explained_variance_tensor.as_mut_ptr(),
                    explained_variance_ratio_tensor.as_mut_ptr(),
                    singular_values_tensor.as_mut_ptr(),
                    mean_tensor.as_mut_ptr(),
                    noise_variance_tensor.as_mut_ptr(),
                    params.flip_signs_based_on_u,
                ),
                operation,
            )
        };
        match (call_result, params_handle.close(operation)) {
            (Err(error), _) => Err(error),
            (Ok(()), close) => close,
        }
    })?;
    drop(transformed_view);
    drop(components_view);
    drop(explained_variance_view);
    drop(explained_variance_ratio_view);
    drop(singular_values_view);
    drop(mean_view);
    drop(noise_variance_view);
    outputs.validate_native_outputs(operation)?;

    Ok(PcaFitTransform { rows, columns, n_components: params.n_components })
}

fn validate_input_dimensions(rows: usize, columns: usize, n_components: usize, operation: &'static str) -> Result<()> {
    if rows < 2 {
        return Err(invalid_input(operation, format!("PCA input must contain at least two rows, found {rows}"))
            .with_context("rows", rows)
            .with_context("minimum_rows", 2));
    }
    if columns < 2 {
        return Err(invalid_input(
            operation,
            format!("PCA input must contain at least two feature columns, found {columns}"),
        )
        .with_context("feature_columns", columns)
        .with_context("minimum_feature_columns", 2));
    }
    if n_components > columns {
        return Err(invalid_input(
            operation,
            format!("n_components ({n_components}) must be <= input columns ({columns})"),
        ));
    }
    Ok(())
}

/// Run PCA against an exclusively owned packed matrix.
///
/// This is the executor-facing ingress for cudf-packed inputs: no caller needs
/// to cast a shared const device pointer into a mutable pointer.
pub fn fit_transform_owned(
    resources: &CuvsResources,
    input: &mut OwnedFloat32Matrix,
    params: &PcaParams,
    outputs: &mut PcaOutputBuffers,
) -> Result<PcaFitTransform> {
    let operation = "pca::fit_transform_owned";
    if !input.is_finite_validated() {
        return Err(invalid_input(operation, "PCA input must carry finite-input validation"));
    }
    if input.layout() != MatrixLayout::ColumnMajor {
        return Err(invalid_input(operation, "PCA owned input must be a contiguous column-major matrix"));
    }
    validate_params(params, operation)?;
    validate_input_dimensions(input.rows(), input.columns(), params.n_components, operation)?;
    validate_output_lengths(outputs, input.rows(), input.columns(), params.n_components, operation)?;
    let mut tensor = input.as_tensor_mut(resources)?;
    fit_transform_into(resources, &mut tensor, params, outputs).map_err(|error| error.with_operation(operation))
}

fn validate_params(params: &PcaParams, operation: &'static str) -> Result<()> {
    require_nonzero(params.n_components, operation, "n_components")?;
    if params.n_components > i32::MAX as usize {
        return Err(invalid_input(operation, "n_components exceeds the cuVS i32 parameter range"));
    }
    if !params.tolerance.is_finite() || params.tolerance < 0.0 {
        return Err(invalid_input(operation, "PCA tolerance must be finite and non-negative"));
    }
    if params.iterations <= 0 {
        return Err(invalid_input(operation, "PCA iterations must be greater than zero"));
    }
    Ok(())
}

fn validate_output_lengths(
    outputs: &PcaOutputBuffers,
    rows: usize,
    columns: usize,
    n_components: usize,
    operation: &'static str,
) -> Result<()> {
    let transformed = rows
        .checked_mul(n_components)
        .ok_or_else(|| invalid_input(operation, "transformed output length overflowed usize"))?;
    let components = n_components
        .checked_mul(columns)
        .ok_or_else(|| invalid_input(operation, "components output length overflowed usize"))?;
    let actual = [
        ("transformed", outputs.transformed.len(), transformed),
        ("components", outputs.components.len(), components),
        ("explained_variance", outputs.explained_variance.len(), n_components),
        ("explained_variance_ratio", outputs.explained_variance_ratio.len(), n_components),
        ("singular_values", outputs.singular_values.len(), n_components),
        ("mean", outputs.mean.len(), columns),
        ("noise_variance", outputs.noise_variance.len(), 1),
    ];
    for (name, actual, expected) in actual {
        if actual != expected {
            return Err(invalid_input(operation, format!("{name} contains {actual} elements, expected {expected}")));
        }
    }
    Ok(())
}

fn matrix_layout_mut(matrix: &DeviceTensorMut<'_, f32>, operation: &'static str) -> Result<MatrixLayout> {
    let rows = matrix.rows()? as i64;
    let columns = matrix.columns()? as i64;
    match matrix.strides() {
        None => Ok(MatrixLayout::RowMajor),
        Some([row_stride, column_stride]) if *row_stride == columns && *column_stride == 1 => {
            Ok(MatrixLayout::RowMajor)
        },
        Some([row_stride, column_stride]) if *row_stride == 1 && *column_stride == rows => {
            Ok(MatrixLayout::ColumnMajor)
        },
        _ => Err(invalid_input(operation, "PCA matrix must be contiguous row-major or column-major")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_dimensions_reject_a_single_row_before_native_pca() {
        let error = validate_input_dimensions(1, 2, 1, "test").expect_err("PCA requires at least two rows");

        assert_eq!(error.kind(), crate::ErrorKind::InvalidInput);
        assert!(error.message().contains("at least two rows"));
        assert_eq!(error.context_value("rows"), Some("1"));
    }

    #[test]
    fn input_dimensions_accept_the_native_boundary() {
        validate_input_dimensions(2, 2, 2, "test").expect("two rows and two columns satisfy the native contract");
    }
}
