use std::{marker::PhantomData, sync::Arc};

use cudf::{DataType, Runtime, Series, TypeId, runtime::ExecutionBindingId};

use crate::{
    DeviceBuffer, DeviceMatrix, MatrixLayout, Result,
    error::{ErrorCode, FailurePhase, check_status, error, from_cudf, invalid_input},
    labels::{adopt_labels, validate_class_labels, validate_finite_targets},
    resources::{CumlResources, TrainingStreams},
};

unsafe extern "C" {
    fn cuml_rf_classifier_fit(
        handle: cuml_sys::cuml_handle_t,
        streams: cuml_sys::cuml_rf_training_streams_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        labels: *const i32,
        n_classes: i32,
        n_trees: i32,
        bootstrap: i32,
        max_samples: f32,
        seed: u64,
        max_depth: i32,
        max_leaves: i32,
        max_features: f32,
        max_n_bins: i32,
        min_samples_leaf: i32,
        min_samples_split: i32,
        min_impurity_decrease: f32,
        max_batch_size: i32,
        split_criterion: i32,
        input_row_major: i32,
        out_forest: *mut cuml_sys::cuml_rf_forest_t,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_rf_regressor_fit(
        handle: cuml_sys::cuml_handle_t,
        streams: cuml_sys::cuml_rf_training_streams_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        labels: *const f32,
        n_trees: i32,
        bootstrap: i32,
        max_samples: f32,
        seed: u64,
        max_depth: i32,
        max_leaves: i32,
        max_features: f32,
        max_n_bins: i32,
        min_samples_leaf: i32,
        min_samples_split: i32,
        min_impurity_decrease: f32,
        max_batch_size: i32,
        split_criterion: i32,
        input_row_major: i32,
        out_forest: *mut cuml_sys::cuml_rf_forest_t,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_rf_classifier_predict(
        handle: cuml_sys::cuml_handle_t,
        forest: cuml_sys::cuml_rf_forest_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        predictions: *mut i64,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_rf_regressor_predict(
        handle: cuml_sys::cuml_handle_t,
        forest: cuml_sys::cuml_rf_forest_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        predictions: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_rf_forest_destroy(forest: cuml_sys::cuml_rf_forest_t) -> cuml_sys::cuml_status_t;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitCriterion {
    Gini,
    Entropy,
    Mse,
}

impl SplitCriterion {
    fn native(self) -> i32 {
        match self {
            Self::Gini => 0,
            Self::Entropy => 1,
            Self::Mse => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RandomForestParams {
    pub n_trees: usize,
    pub bootstrap: bool,
    pub max_samples: f32,
    pub seed: u64,
    pub max_depth: i32,
    pub max_leaves: i32,
    pub max_features: f32,
    pub max_n_bins: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub min_impurity_decrease: f32,
    pub max_batch_size: usize,
    pub split_criterion: SplitCriterion,
}

impl Default for RandomForestParams {
    fn default() -> Self {
        Self {
            n_trees: 100,
            bootstrap: true,
            max_samples: 1.0,
            seed: 0,
            max_depth: 16,
            max_leaves: -1,
            max_features: 1.0,
            max_n_bins: 128,
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
            max_batch_size: 4096,
            split_criterion: SplitCriterion::Gini,
        }
    }
}

impl RandomForestParams {
    fn validate(self, classifier: bool, op: &'static str) -> Result<[i32; 7]> {
        let criterion_ok = if classifier {
            matches!(self.split_criterion, SplitCriterion::Gini | SplitCriterion::Entropy)
        } else {
            self.split_criterion == SplitCriterion::Mse
        };
        if !criterion_ok {
            return Err(error(
                ErrorCode::UnsupportedParameter,
                FailurePhase::Preflight,
                op,
                "split criterion is unsupported for this forest family",
            ));
        }
        if self.n_trees == 0
            || !self.max_samples.is_finite()
            || !(0.0..=1.0).contains(&self.max_samples)
            || self.max_samples == 0.0
            || !self.max_features.is_finite()
            || !(0.0..=1.0).contains(&self.max_features)
            || self.max_features == 0.0
            || !self.min_impurity_decrease.is_finite()
            || self.min_impurity_decrease < 0.0
            || self.max_depth < 0
            || self.max_leaves < -1
            || self.max_n_bins < 2
            || self.min_samples_leaf == 0
            || self.min_samples_split < 2
            || self.max_batch_size == 0
        {
            return Err(invalid_input(op, "random forest parameters are out of range"));
        }
        Ok([
            as_i32(self.n_trees, op, "n_trees")?,
            as_i32(self.max_n_bins, op, "max_n_bins")?,
            as_i32(self.min_samples_leaf, op, "min_samples_leaf")?,
            as_i32(self.min_samples_split, op, "min_samples_split")?,
            as_i32(self.max_batch_size, op, "max_batch_size")?,
            self.max_depth,
            self.max_leaves,
        ])
    }
}

struct Forest {
    raw: Option<cuml_sys::cuml_rf_forest_t>,
    binding_id: ExecutionBindingId,
    n_features: usize,
    n_classes: Option<usize>,
    _runtime: Arc<Runtime>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Forest {
    fn close(&mut self) -> Result<()> {
        let Some(raw) = self.raw else { return Ok(()) };
        // SAFETY: this wrapper uniquely owns the opaque native forest.
        let status = unsafe { cuml_rf_forest_destroy(raw) };
        if status == cuml_sys::CUML_SUCCESS {
            self.raw = None;
            return Ok(());
        }
        Err(error(
            ErrorCode::NativeReleaseFailed,
            FailurePhase::NativeRelease,
            "RandomForest::close",
            "native forest destruction failed",
        ))
    }
}

impl Drop for Forest {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub struct RandomForestClassifier(Forest);
pub struct RandomForestRegressor(Forest);

impl RandomForestClassifier {
    pub fn fit(
        resources: &CumlResources,
        streams: &TrainingStreams,
        input: &DeviceMatrix<'_>,
        labels: &Series,
        params: RandomForestParams,
    ) -> Result<Self> {
        const OP: &str = "RandomForestClassifier::fit";
        let dimensions = validate_fit(resources, input, labels.len(), OP)?;
        let params_i32 = params.validate(true, OP)?;
        let labels = adopt_labels(resources, labels, OP)?;
        let n_classes = validate_class_labels(&labels, OP)?;
        let n_classes = i32::try_from(n_classes).map_err(|_| size_error(OP, "class count exceeds i32"))?;
        let labels_ptr = labels.as_column().device_data_ptr().map_err(|e| from_cudf(OP, e))?.cast();
        resources.validate_exact_binding(OP)?;
        let stream_raw = streams.raw_for_fit(resources, OP)?;
        let mut raw = std::ptr::null_mut();
        resources.with_native(OP, |handle| {
            // SAFETY: input and labels stay live until native fit synchronizes the pool;
            // its streams retain either sibling reservations or backend routes.
            check_status(
                unsafe {
                    cuml_rf_classifier_fit(
                        handle,
                        stream_raw,
                        input.as_ptr().cast(),
                        dimensions.0,
                        dimensions.1,
                        labels_ptr,
                        n_classes,
                        params_i32[0],
                        i32::from(params.bootstrap),
                        params.max_samples,
                        params.seed,
                        params_i32[5],
                        params_i32[6],
                        params.max_features,
                        params_i32[1],
                        params_i32[2],
                        params_i32[3],
                        params.min_impurity_decrease,
                        params_i32[4],
                        params.split_criterion.native(),
                        i32::from(input.layout() == MatrixLayout::RowMajor),
                        &mut raw,
                    )
                },
                OP,
            )
        })?;
        Ok(Self(Forest {
            raw: Some(non_null_forest(raw, OP)?),
            binding_id: resources.binding_id(),
            n_features: input.columns(),
            n_classes: Some(n_classes as usize),
            _runtime: Arc::clone(resources.runtime()),
            _not_send_sync: PhantomData,
        }))
    }

    /// Predicts from row-major features. Pinned cuML copies the matrix to host
    /// and synchronizes the CUDA stream before traversing the host-resident trees.
    pub fn predict(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<Series> {
        const OP: &str = "RandomForestClassifier::predict";
        validate_predict(&self.0, resources, input, OP)?;
        if input.layout() != MatrixLayout::RowMajor {
            return Err(error(
                ErrorCode::UnsupportedLayout,
                FailurePhase::Preflight,
                OP,
                "random forest prediction requires row-major input",
            ));
        }
        resources.validate_exact_binding(OP)?;
        let raw =
            self.0.raw.ok_or_else(|| error(ErrorCode::ModelClosed, FailurePhase::Preflight, OP, "forest is closed"))?;
        let mut predictions = DeviceBuffer::<i64>::new(resources, input.rows())?;
        let ptr = predictions.as_mut_ptr()?.cast();
        let rows = as_i32(input.rows(), OP, "row count")?;
        let columns = as_i32(input.columns(), OP, "feature count")?;
        resources.with_native(OP, |handle| {
            // SAFETY: model and row-major input are bound to the current cuML runtime.
            check_status(
                unsafe { cuml_rf_classifier_predict(handle, raw, input.as_ptr().cast(), rows, columns, ptr) },
                OP,
            )
        })?;
        predictions.validate_range(0, self.0.n_classes.expect("classifier class count") as i64, OP)?;
        let output = Series::from_column(predictions.into_column()?).map_err(|e| from_cudf(OP, e))?;
        output.astype(DataType::new(TypeId::Int32)).map_err(|e| from_cudf(OP, e))
    }

    pub fn close(&mut self) -> Result<()> {
        self.0.close()
    }
    pub fn n_features(&self) -> usize {
        self.0.n_features
    }
}

impl RandomForestRegressor {
    pub fn fit(
        resources: &CumlResources,
        streams: &TrainingStreams,
        input: &DeviceMatrix<'_>,
        targets: &Series,
        params: RandomForestParams,
    ) -> Result<Self> {
        const OP: &str = "RandomForestRegressor::fit";
        let dimensions = validate_fit(resources, input, targets.len(), OP)?;
        let params_i32 = params.validate(false, OP)?;
        let targets = adopt_labels(resources, targets, OP)?;
        validate_finite_targets(&targets, OP)?;
        let target_ptr = targets.as_column().device_data_ptr().map_err(|e| from_cudf(OP, e))?.cast();
        resources.validate_exact_binding(OP)?;
        let stream_raw = streams.raw_for_fit(resources, OP)?;
        let mut raw = std::ptr::null_mut();
        resources.with_native(OP, |handle| {
            // SAFETY: input and targets stay live until native fit synchronizes the pool;
            // its streams retain either sibling reservations or backend routes.
            check_status(
                unsafe {
                    cuml_rf_regressor_fit(
                        handle,
                        stream_raw,
                        input.as_ptr().cast(),
                        dimensions.0,
                        dimensions.1,
                        target_ptr,
                        params_i32[0],
                        i32::from(params.bootstrap),
                        params.max_samples,
                        params.seed,
                        params_i32[5],
                        params_i32[6],
                        params.max_features,
                        params_i32[1],
                        params_i32[2],
                        params_i32[3],
                        params.min_impurity_decrease,
                        params_i32[4],
                        params.split_criterion.native(),
                        i32::from(input.layout() == MatrixLayout::RowMajor),
                        &mut raw,
                    )
                },
                OP,
            )
        })?;
        Ok(Self(Forest {
            raw: Some(non_null_forest(raw, OP)?),
            binding_id: resources.binding_id(),
            n_features: input.columns(),
            n_classes: None,
            _runtime: Arc::clone(resources.runtime()),
            _not_send_sync: PhantomData,
        }))
    }

    /// Predicts from row-major features. Pinned cuML copies the matrix to host
    /// and synchronizes the CUDA stream before traversing the host-resident trees.
    pub fn predict(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<Series> {
        const OP: &str = "RandomForestRegressor::predict";
        validate_predict(&self.0, resources, input, OP)?;
        if input.layout() != MatrixLayout::RowMajor {
            return Err(error(
                ErrorCode::UnsupportedLayout,
                FailurePhase::Preflight,
                OP,
                "random forest prediction requires row-major input",
            ));
        }
        resources.validate_exact_binding(OP)?;
        let raw =
            self.0.raw.ok_or_else(|| error(ErrorCode::ModelClosed, FailurePhase::Preflight, OP, "forest is closed"))?;
        let mut predictions = DeviceBuffer::<f32>::new(resources, input.rows())?;
        let ptr = predictions.as_mut_ptr()?.cast();
        let rows = as_i32(input.rows(), OP, "row count")?;
        let columns = as_i32(input.columns(), OP, "feature count")?;
        resources.with_native(OP, |handle| {
            // SAFETY: model and row-major input are bound to the current cuML runtime.
            check_status(
                unsafe { cuml_rf_regressor_predict(handle, raw, input.as_ptr().cast(), rows, columns, ptr) },
                OP,
            )
        })?;
        predictions.validate_finite(OP)?;
        Series::from_column(predictions.into_column()?).map_err(|e| from_cudf(OP, e))
    }

    pub fn close(&mut self) -> Result<()> {
        self.0.close()
    }
    pub fn n_features(&self) -> usize {
        self.0.n_features
    }
}

fn validate_fit(
    resources: &CumlResources,
    input: &DeviceMatrix<'_>,
    label_len: usize,
    op: &'static str,
) -> Result<(i32, i32)> {
    let rows = as_i32(input.rows(), op, "row count")?;
    let columns = as_i32(input.columns(), op, "feature count")?;
    if rows < 2 || columns < 1 || label_len != input.rows() {
        return Err(invalid_input(op, "training matrix and labels have incompatible dimensions"));
    }
    if input.binding_id() != resources.binding_id() {
        return Err(error(
            ErrorCode::RuntimeBindingMismatch,
            FailurePhase::Preflight,
            op,
            "input and resources use different execution bindings",
        ));
    }
    Ok((rows, columns))
}

fn validate_predict(
    forest: &Forest,
    resources: &CumlResources,
    input: &DeviceMatrix<'_>,
    op: &'static str,
) -> Result<()> {
    if input.columns() != forest.n_features
        || input.binding_id() != forest.binding_id
        || resources.binding_id() != forest.binding_id
    {
        return Err(error(
            ErrorCode::RuntimeBindingMismatch,
            FailurePhase::Preflight,
            op,
            "model, input, and resources must share a binding and feature shape",
        ));
    }
    if input.rows() == 0 {
        return Err(invalid_input(op, "prediction input must contain rows"));
    }
    Ok(())
}

fn non_null_forest(raw: cuml_sys::cuml_rf_forest_t, op: &'static str) -> Result<cuml_sys::cuml_rf_forest_t> {
    if raw.is_null() {
        Err(error(
            ErrorCode::InvariantViolation,
            FailurePhase::NativeCall,
            op,
            "cuML returned a null forest after successful fit",
        ))
    } else {
        Ok(raw)
    }
}

fn as_i32(value: usize, op: &'static str, name: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| size_error(op, name))
}

fn size_error(op: &'static str, what: &'static str) -> crate::Error {
    error(ErrorCode::SizeOverflow, FailurePhase::Preflight, op, format!("{what} exceeds i32"))
}
