use cudf::{DataType, Exec, Series, TypeId, runtime::ExecutionBindingId};

use crate::{
    DeviceBuffer, DeviceMatrix, Error, ErrorCode, FailurePhase, MatrixLayout, Result,
    error::{check_status, error, from_cudf, invalid_input},
    labels::{adopt_labels, validate_class_labels},
    resources::CumlResources,
};

unsafe extern "C" {
    fn cuml_glm_qn_fit(
        handle: cuml_sys::cuml_handle_t,
        input: *mut f32,
        rows: i32,
        columns: i32,
        classes: i32,
        column_major: i32,
        labels: *const f32,
        coefficients: *mut f32,
        loss: i32,
        penalty_l1: f64,
        penalty_l2: f64,
        max_iter: i32,
        linesearch_max_iter: i32,
        lbfgs_memory: i32,
        fit_intercept: i32,
        penalty_normalized: i32,
        grad_tol: f64,
        change_tol: f64,
        objective: *mut f64,
        iterations: *mut i32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_glm_qn_predict(
        handle: cuml_sys::cuml_handle_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        classes: i32,
        column_major: i32,
        coefficients: *const f32,
        loss: i32,
        fit_intercept: i32,
        predictions: *mut f32,
    ) -> cuml_sys::cuml_status_t;
}

#[derive(Debug, Clone, Copy)]
pub struct LogisticRegressionParams {
    pub penalty_l1: f64,
    pub penalty_l2: f64,
    pub max_iter: usize,
    pub linesearch_max_iter: usize,
    pub lbfgs_memory: usize,
    pub fit_intercept: bool,
    pub penalty_normalized: bool,
    pub grad_tol: f64,
    pub change_tol: f64,
}

impl Default for LogisticRegressionParams {
    fn default() -> Self {
        Self {
            penalty_l1: 0.0,
            penalty_l2: 0.0,
            max_iter: 1000,
            linesearch_max_iter: 50,
            lbfgs_memory: 5,
            fit_intercept: true,
            penalty_normalized: true,
            grad_tol: 1e-4,
            change_tol: 1e-5,
        }
    }
}

impl LogisticRegressionParams {
    fn validate(self, op: &'static str) -> Result<()> {
        if !self.penalty_l1.is_finite()
            || self.penalty_l1 < 0.0
            || !self.penalty_l2.is_finite()
            || self.penalty_l2 < 0.0
            || self.max_iter == 0
            || self.linesearch_max_iter == 0
            || self.lbfgs_memory == 0
            || !self.grad_tol.is_finite()
            || self.grad_tol < 0.0
            || !self.change_tol.is_finite()
            || self.change_tol < 0.0
        {
            return Err(invalid_input(op, "QN logistic regression parameters are invalid"));
        }
        for value in [self.max_iter, self.linesearch_max_iter, self.lbfgs_memory] {
            i32::try_from(value).map_err(|_| {
                error(ErrorCode::SizeOverflow, FailurePhase::Preflight, op, "QN integer parameter exceeds i32")
            })?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticRegressionFit {
    pub objective: f64,
    pub iterations: usize,
    pub converged: bool,
}

pub struct LogisticRegressionModel {
    coefficients: DeviceBuffer<f32>,
    n_features: usize,
    n_classes: usize,
    fit_intercept: bool,
    loss: i32,
    binding_id: ExecutionBindingId,
}

impl LogisticRegressionModel {
    pub fn coefficients(&self) -> &DeviceBuffer<f32> {
        &self.coefficients
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn fit(
        resources: &CumlResources,
        input: &DeviceMatrix<'_>,
        labels: &Series,
        params: LogisticRegressionParams,
    ) -> Result<(Self, LogisticRegressionFit)> {
        const OP: &str = "LogisticRegressionModel::fit";
        params.validate(OP)?;
        let rows = i32::try_from(input.rows()).map_err(|_| size_error(OP, "row count exceeds i32"))?;
        let columns = i32::try_from(input.columns()).map_err(|_| size_error(OP, "feature count exceeds i32"))?;
        if rows < 2 || columns < 1 {
            return Err(invalid_input(OP, "expected at least two rows and one feature"));
        }
        if input.binding_id() != resources.binding_id() {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                OP,
                "input and resources use different execution bindings",
            ));
        }
        if labels.len() != input.rows() {
            return Err(invalid_input(OP, "label count must match training rows"));
        }
        let labels = adopt_labels(resources, labels, OP)?;
        let classes = validate_class_labels(&labels, OP)?;
        let class_count = i32::try_from(classes).map_err(|_| size_error(OP, "class count exceeds i32"))?;
        if class_count < 2 {
            return Err(invalid_input(OP, "n_classes must be at least two"));
        }
        // Pinned QN expects C=2 for binary logistic loss but stores one
        // coefficient vector; softmax stores a vector for each class.
        let n_outputs = if classes == 2 { 1 } else { class_count };
        let loss = if classes == 2 { 0 } else { 2 };
        let coefficients_len = input
            .columns()
            .checked_add(usize::from(params.fit_intercept))
            .and_then(|width| width.checked_mul(n_outputs as usize))
            .ok_or_else(|| size_error(OP, "coefficient extent overflow"))?;
        resources.validate_exact_binding(OP)?;
        let mut coefficients = DeviceBuffer::new(resources, coefficients_len)?;
        let labels_f32 =
            labels.on(Exec::RETAINED).astype(DataType::new(TypeId::Float32)).map_err(|e| from_cudf(OP, e))?;
        let labels_ptr = labels_f32.as_column().device_data_ptr().map_err(|e| from_cudf(OP, e))?.cast();
        let mut objective = 0.0_f64;
        let mut iterations = 0_i32;
        let params_copy = params;
        let input_ptr = input.as_ptr().cast_mut().cast();
        let coefficient_ptr = coefficients.as_mut_ptr()?.cast();
        resources.with_native(OP, |handle| {
            // SAFETY: all pointers are live, device-resident buffers bound to the current handle.
            check_status(
                unsafe {
                    cuml_glm_qn_fit(
                        handle,
                        input_ptr,
                        rows,
                        columns,
                        class_count,
                        i32::from(input.layout() == MatrixLayout::ColumnMajor),
                        labels_ptr,
                        coefficient_ptr,
                        loss,
                        params_copy.penalty_l1,
                        params_copy.penalty_l2,
                        params_copy.max_iter as i32,
                        params_copy.linesearch_max_iter as i32,
                        params_copy.lbfgs_memory as i32,
                        i32::from(params_copy.fit_intercept),
                        i32::from(params_copy.penalty_normalized),
                        params_copy.grad_tol,
                        params_copy.change_tol,
                        &mut objective,
                        &mut iterations,
                    )
                },
                OP,
            )
        })?;
        coefficients.validate_finite(OP)?;
        if !objective.is_finite() || iterations < 0 || iterations as usize > params.max_iter {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::OutputValidation,
                OP,
                "native QN fit returned invalid objective or iteration count",
            ));
        }
        let fit = LogisticRegressionFit {
            objective,
            iterations: iterations as usize,
            converged: (iterations as usize) < params.max_iter,
        };
        Ok((
            Self {
                coefficients,
                n_features: input.columns(),
                n_classes: classes,
                fit_intercept: params.fit_intercept,
                loss,
                binding_id: resources.binding_id(),
            },
            fit,
        ))
    }

    pub fn predict(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<Series> {
        const OP: &str = "LogisticRegressionModel::predict";
        let rows = i32::try_from(input.rows()).map_err(|_| size_error(OP, "row count exceeds i32"))?;
        let columns = i32::try_from(input.columns()).map_err(|_| size_error(OP, "feature count exceeds i32"))?;
        if rows < 1
            || columns != self.n_features as i32
            || input.binding_id() != self.binding_id
            || resources.binding_id() != self.binding_id
        {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                OP,
                "model, input, and resources must share a binding and feature shape",
            ));
        }
        resources.validate_exact_binding(OP)?;
        let mut predictions = DeviceBuffer::new(resources, input.rows())?;
        let input_ptr = input.as_ptr();
        let coefficients_ptr = self.coefficients.as_ptr()?.cast();
        let predictions_ptr = predictions.as_mut_ptr()?.cast();
        resources.with_native(OP, |handle| {
            // SAFETY: model and input buffers remain alive for the native call.
            check_status(
                unsafe {
                    cuml_glm_qn_predict(
                        handle,
                        input_ptr.cast(),
                        rows,
                        columns,
                        self.n_classes as i32,
                        i32::from(input.layout() == MatrixLayout::ColumnMajor),
                        coefficients_ptr,
                        self.loss,
                        i32::from(self.fit_intercept),
                        predictions_ptr,
                    )
                },
                OP,
            )
        })?;
        predictions.validate_finite(OP)?;
        let predicted = Series::from_column(predictions.into_column()?).map_err(|e| from_cudf(OP, e))?;
        let min = predicted.on(Exec::RETAINED).min_value::<f32>().map_err(|e| from_cudf(OP, e))?;
        let indices =
            predicted.on(Exec::RETAINED).astype(DataType::new(TypeId::Int32)).map_err(|e| from_cudf(OP, e))?;
        let max_index = indices.on(Exec::RETAINED).max_value::<i32>().map_err(|e| from_cudf(OP, e))?;
        if min.is_none_or(|value| value < 0.0) || max_index.is_none_or(|value| value >= self.n_classes as i32) {
            return Err(error(
                ErrorCode::OutputOutOfRange,
                FailurePhase::OutputValidation,
                OP,
                "QN predictions must be integral class indices in 0..n_classes",
            ));
        }
        let roundtrip =
            indices.on(Exec::RETAINED).astype(DataType::new(TypeId::Float32)).map_err(|e| from_cudf(OP, e))?;
        // Equality reduces the device comparison mask to one host boolean.
        if !predicted
            .on(Exec::RETAINED)
            .eq(&roundtrip)
            .map_err(|e| from_cudf(OP, e))?
            .on(Exec::RETAINED)
            .all_value()
            .map_err(|e| from_cudf(OP, e))?
        {
            return Err(error(
                ErrorCode::OutputOutOfRange,
                FailurePhase::OutputValidation,
                OP,
                "QN predictions must be integral class indices in 0..n_classes",
            ));
        }
        Ok(indices)
    }
}

fn size_error(op: &'static str, message: &'static str) -> Error {
    error(ErrorCode::SizeOverflow, FailurePhase::Preflight, op, message)
}
