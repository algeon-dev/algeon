use std::{marker::PhantomData, ptr, sync::Arc};

use cudf::{Runtime, Series, runtime::ExecutionBindingId};

use crate::{
    DeviceMatrix, ForestInferenceModel, ForestOutput, ForestOutputKind, MatrixLayout, Result,
    error::{ErrorCode, FailurePhase, check_status, error, from_cudf, invalid_input},
    forest_inference::empty_metadata,
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
    fn forest_model(
        &self,
        resources: &CumlResources,
        output: ForestOutput,
        expected: ForestOutputKind,
        op: &'static str,
    ) -> Result<ForestInferenceModel> {
        let raw =
            self.raw.ok_or_else(|| error(ErrorCode::ModelClosed, FailurePhase::Preflight, op, "forest is closed"))?;
        if resources.binding_id() != self.binding_id {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                op,
                "model and resources must share an execution binding",
            ));
        }
        resources.validate_exact_binding(op)?;
        let mut model = ptr::null_mut();
        let mut metadata = empty_metadata();
        resources.with_native(op, |handle| {
            // SAFETY: the forest is live and bound to this handle; import copies its
            // trees to the handle's device before returning.
            check_status(
                unsafe { cuml_sys::cuml_rf_forest_import(handle, raw, output.native(), &mut model, &mut metadata) },
                op,
            )
        })?;
        let model = ForestInferenceModel::from_native(resources, model, metadata, output, op)?;
        if model.info().num_features != self.n_features || model.output_kind() != expected {
            return Err(error(
                ErrorCode::InvariantViolation,
                FailurePhase::NativeCall,
                op,
                "nvForest import disagrees with the fitted forest shape",
            )
            .with_detail("features", model.info().num_features)
            .with_detail("output", format!("{:?}", model.output_kind())));
        }
        Ok(model)
    }

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

    /// Imports the fitted forest into nvForest on the resources' device,
    /// allocating through their query-domain route. The model predicts
    /// row-major Float32 features into one averaged class probability per
    /// training label. Reduce a prediction with
    /// [`ForestPrediction::into_class_prediction`](crate::ForestPrediction::into_class_prediction)
    /// for the most probable label, which resolves ties to the lowest label.
    /// Close the model before releasing the execution grant.
    pub fn forest_model(&self, resources: &CumlResources) -> Result<ForestInferenceModel> {
        let classes = self.0.n_classes.expect("classifier class count");
        self.0.forest_model(
            resources,
            ForestOutput::Scores,
            ForestOutputKind::Vector(classes),
            "RandomForestClassifier::forest_model",
        )
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

    /// Imports the fitted forest into nvForest on the resources' device,
    /// allocating through their query-domain route. The model predicts
    /// row-major Float32 features into one Float32 series of tree-averaged
    /// targets. Close the model before releasing the execution grant.
    pub fn forest_model(&self, resources: &CumlResources) -> Result<ForestInferenceModel> {
        self.0.forest_model(
            resources,
            ForestOutput::Scores,
            ForestOutputKind::Score,
            "RandomForestRegressor::forest_model",
        )
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

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, ptr, sync::Arc};

    use cudf::{DenseMatrixLayout, PackedMatrix, Runtime, Series};
    use cuml_sys::{
        CUML_SUCCESS,
        test_support::{cuml_test_forest_gtil_predict, cuml_test_rf_treelite_checkpoint},
    };

    use super::{Forest, RandomForestClassifier, RandomForestParams, RandomForestRegressor, SplitCriterion};
    use crate::{
        ClassPrediction, DeviceMatrix, ForestInferenceModel, ForestOutputKind, ForestPrecision,
        resources::{CumlResources, TrainingStreams},
    };

    fn pack(runtime: &Runtime, rows: &[Vec<f32>]) -> PackedMatrix {
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let column = runtime.column_from_slice(&flat).unwrap();
        runtime.pack_flat_column(&column, rows.len(), rows[0].len(), DenseMatrixLayout::RowMajor).unwrap()
    }

    fn predict(resources: &CumlResources, mut forest: ForestInferenceModel, rows: &[Vec<f32>]) -> Series {
        let packed = pack(resources.runtime(), rows);
        let validated = packed.validate().unwrap();
        let input = DeviceMatrix::from_validated(&validated).unwrap();
        let series = forest.predict(resources, &input).unwrap().into_series().unwrap();
        forest.close().unwrap();
        let [series]: [Series; 1] = series.try_into().unwrap();
        series
    }

    fn classify(resources: &CumlResources, mut forest: ForestInferenceModel, rows: &[Vec<f32>]) -> ClassPrediction {
        let packed = pack(resources.runtime(), rows);
        let validated = packed.validate().unwrap();
        let input = DeviceMatrix::from_validated(&validated).unwrap();
        let prediction = forest.predict(resources, &input).unwrap().into_class_prediction().unwrap();
        forest.close().unwrap();
        prediction
    }

    /// Host GTIL over the Treelite model that nvForest imports.
    fn gtil(forest: &Forest, rows: &[Vec<f32>]) -> Vec<Vec<f64>> {
        let (mut checkpoint, mut len) = (ptr::null_mut(), 0);
        let status = unsafe { cuml_test_rf_treelite_checkpoint(forest.raw.unwrap(), &mut checkpoint, &mut len) };
        assert_eq!(status, CUML_SUCCESS);
        let flat = rows.iter().flatten().map(|v| f64::from(*v)).collect::<Vec<_>>();
        let mut output = vec![0.0; rows.len() * 8];
        let mut written = 0;
        let status = unsafe {
            cuml_test_forest_gtil_predict(
                checkpoint,
                len,
                flat.as_ptr(),
                rows.len() as i32,
                rows[0].len() as i32,
                output.as_mut_ptr(),
                output.len(),
                &mut written,
            )
        };
        unsafe { cuml_sys::cuml_forest_bytes_free(checkpoint) };
        assert_eq!(status, CUML_SUCCESS);
        output[..written].chunks(written / rows.len()).map(<[f64]>::to_vec).collect()
    }

    fn max_score(scores: &[f64]) -> f64 {
        scores.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn lowest_argmax(scores: &[f64]) -> i32 {
        let max = max_score(scores);
        scores.iter().position(|score| *score == max).unwrap() as i32
    }

    fn setup() -> (CumlResources, TrainingStreams) {
        let runtime = Arc::new(Runtime::new().unwrap());
        let resources = CumlResources::for_runtime(runtime).unwrap();
        let streams = resources.training_streams(NonZeroUsize::new(1).unwrap(), None).unwrap();
        (resources, streams)
    }

    #[test]
    fn nvforest_import_matches_treelite_oracle_for_classifiers_and_regressors() {
        let (resources, streams) = setup();
        let runtime = Arc::clone(resources.runtime());
        let training = (0..48).map(|i| vec![(i % 8) as f32 - 3.5, (i / 8) as f32 - 2.5]).collect::<Vec<Vec<f32>>>();
        let held_out =
            (0..25).map(|i| vec![(i % 5) as f32 * 1.7 - 3.9, (i / 5) as f32 * 1.3 - 2.9]).collect::<Vec<Vec<f32>>>();
        let packed = pack(&runtime, &training);
        let validated = packed.validate().unwrap();
        let input = DeviceMatrix::from_validated(&validated).unwrap();
        let params = RandomForestParams { n_trees: 8, max_depth: 8, ..Default::default() };
        let binary = training.iter().map(|x| i32::from(x[0] + x[1] > 0.0)).collect::<Vec<_>>();
        let three = training.iter().map(|x| i32::from(x[0] > 0.0) + i32::from(x[1] > 0.0)).collect::<Vec<_>>();
        for (labels, classes) in [(binary, 2), (three, 3)] {
            let labels = runtime.series(&labels).unwrap();
            let classifier = RandomForestClassifier::fit(&resources, &streams, &input, &labels, params).unwrap();
            let forest = classifier.forest_model(&resources).unwrap();
            let info = forest.info();
            assert_eq!((info.precision, info.num_features, info.num_trees), (ForestPrecision::Single, 2, 8));
            assert_eq!(forest.output_kind(), ForestOutputKind::Vector(classes));
            let predicted = classify(&resources, forest, &held_out);
            let oracle = gtil(&classifier.0, &held_out);
            let expected = oracle.iter().map(|row| lowest_argmax(row)).collect::<Vec<_>>();
            assert_eq!(predicted.class.to_vec_strict::<i32>().unwrap(), expected, "{classes} classes");
            for (actual, row) in predicted.probability.to_vec_strict::<f32>().unwrap().iter().zip(&oracle) {
                let expected = max_score(row);
                assert!((f64::from(*actual) - expected).abs() <= 1e-5, "{classes} classes: {actual} vs {expected}");
            }
        }

        let targets = runtime.series(&training.iter().map(|x| 2.0 * x[0] + x[1]).collect::<Vec<f32>>()).unwrap();
        let params = RandomForestParams { split_criterion: SplitCriterion::Mse, ..params };
        let regressor = RandomForestRegressor::fit(&resources, &streams, &input, &targets, params).unwrap();
        let forest = regressor.forest_model(&resources).unwrap();
        assert_eq!(forest.output_kind(), ForestOutputKind::Score);
        assert_eq!(forest.info().precision, ForestPrecision::Single);
        let predicted = predict(&resources, forest, &held_out).to_vec_strict::<f32>().unwrap();
        for (actual, expected) in predicted.iter().zip(gtil(&regressor.0, &held_out)) {
            let (actual, expected) = (f64::from(*actual), expected[0]);
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.0), "{actual} vs {expected}");
        }
    }

    #[test]
    fn classifier_ties_resolve_to_the_lowest_label() {
        let (resources, streams) = setup();
        let runtime = Arc::clone(resources.runtime());
        // Every tree sees all rows, so the x = 0 leaf holds classes 0 and 1 at
        // probability 0.5 each.
        let training = (0..24).map(|i| vec![if i < 16 { 0.0 } else { 1.0 }]).collect::<Vec<Vec<f32>>>();
        let labels = runtime.series(&(0..24).map(|i| i / 8).collect::<Vec<i32>>()).unwrap();
        let packed = pack(&runtime, &training);
        let validated = packed.validate().unwrap();
        let input = DeviceMatrix::from_validated(&validated).unwrap();
        let params = RandomForestParams {
            n_trees: 3,
            bootstrap: false,
            max_samples: 1.0,
            max_features: 1.0,
            max_depth: 4,
            ..Default::default()
        };
        let classifier = RandomForestClassifier::fit(&resources, &streams, &input, &labels, params).unwrap();
        let rows = vec![vec![0.0], vec![1.0]];
        let scores = gtil(&classifier.0, &rows);
        assert_eq!(scores[0][0], scores[0][1]);
        assert!(scores[0][0] > scores[0][2]);
        let predicted = classify(&resources, classifier.forest_model(&resources).unwrap(), &rows);
        assert_eq!(predicted.class.to_vec_strict::<i32>().unwrap(), [0, 2]);
        // The x = 0 leaf splits its votes between classes 0 and 1; x = 1 is pure.
        assert_eq!(predicted.probability.to_vec_strict::<f32>().unwrap(), [0.5, 1.0]);
    }
}
