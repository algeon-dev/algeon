use std::{marker::PhantomData, sync::Arc};

use cudf::{
    DataType, DenseMatrixLayout, Exec, Runtime, Series, TypeId,
    runtime::{DeviceEntryGuard, ExecutionBindingId},
};

use crate::{
    ClassPrediction, DeviceBuffer, DeviceMatrix, MatrixLayout, Result,
    error::{ErrorCode, FailurePhase, check_status, error, from_cudf, invalid_input},
    resources::CumlResources,
};

const IMPORT: &str = "ForestInferenceModel::import";
const PREDICT: &str = "ForestInferenceModel::predict";
const ASSEMBLE: &str = "ForestPrediction::into_series";

/// Serialized tree-ensemble formats accepted by [`inspect`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForestModelFormat {
    XgboostJson,
    XgboostUbjson,
    XgboostLegacy,
    LightGbm,
    TreeliteCheckpoint,
}

impl ForestModelFormat {
    fn native(self) -> cuml_sys::cuml_forest_format_t {
        match self {
            Self::XgboostJson => cuml_sys::CUML_FOREST_FORMAT_XGBOOST_JSON,
            Self::XgboostUbjson => cuml_sys::CUML_FOREST_FORMAT_XGBOOST_UBJSON,
            Self::XgboostLegacy => cuml_sys::CUML_FOREST_FORMAT_XGBOOST_LEGACY,
            Self::LightGbm => cuml_sys::CUML_FOREST_FORMAT_LIGHTGBM,
            Self::TreeliteCheckpoint => cuml_sys::CUML_FOREST_FORMAT_TREELITE_CHECKPOINT,
        }
    }
}

/// Floating-point type of the imported forest. Double forests keep double
/// thresholds exact and consume Float64 features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForestPrecision {
    Single,
    Double,
}

impl ForestPrecision {
    pub fn element(self) -> TypeId {
        match self {
            Self::Single => TypeId::Float32,
            Self::Double => TypeId::Float64,
        }
    }
}

/// Output requested at import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForestOutput {
    /// The model's postprocessed outputs, one per native output slot.
    Scores,
    /// The index of the largest raw class score; ties take the lowest index.
    ClassIndex,
}

/// Row shape of the values exposed by [`ForestPrediction::into_series`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForestOutputKind {
    /// One Float32 value per row.
    Score,
    /// One Int32 class index per row in `0..classes`.
    ClassIndex { classes: usize },
    /// `n` Float32 values per row.
    Vector(usize),
}

impl ForestOutput {
    pub(crate) fn native(self) -> cuml_sys::cuml_forest_output_t {
        match self {
            Self::Scores => cuml_sys::CUML_FOREST_OUTPUT_SCORES,
            Self::ClassIndex => cuml_sys::CUML_FOREST_OUTPUT_CLASS_INDEX,
        }
    }
}

impl ForestOutputKind {
    /// Number of output series per row.
    pub fn width(self) -> usize {
        match self {
            Self::Score | Self::ClassIndex { .. } => 1,
            Self::Vector(width) => width,
        }
    }
}

/// Model facts proven by nvForest's importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ForestModelInfo {
    pub num_features: usize,
    pub num_trees: usize,
    /// nvForest's per-row output stride.
    pub native_output_stride: usize,
    pub precision: ForestPrecision,
}

impl ForestModelInfo {
    /// Output shape for `output`; class indices require at least two outputs.
    pub fn output_kind(&self, output: ForestOutput) -> Result<ForestOutputKind> {
        match (output, self.native_output_stride) {
            (ForestOutput::Scores, 1) => Ok(ForestOutputKind::Score),
            (ForestOutput::Scores, width) => Ok(ForestOutputKind::Vector(width)),
            (ForestOutput::ClassIndex, classes) if classes >= 2 => Ok(ForestOutputKind::ClassIndex { classes }),
            (ForestOutput::ClassIndex, _) => Err(invalid_input(
                "ForestModelInfo::output_kind",
                "class index output requires a model with at least two outputs",
            )),
        }
    }

    fn from_native(raw: cuml_sys::cuml_forest_metadata_t, op: &'static str) -> Result<Self> {
        let count = |value: i32, name: &str| {
            usize::try_from(value).ok().filter(|value| *value > 0).ok_or_else(|| {
                error(
                    ErrorCode::InvariantViolation,
                    FailurePhase::NativeCall,
                    op,
                    format!("nvForest reported {name} {value}"),
                )
            })
        };
        let precision = match raw.precision {
            cuml_sys::CUML_FOREST_PRECISION_FLOAT32 => ForestPrecision::Single,
            cuml_sys::CUML_FOREST_PRECISION_FLOAT64 => ForestPrecision::Double,
            other => {
                return Err(error(
                    ErrorCode::InvariantViolation,
                    FailurePhase::NativeCall,
                    op,
                    format!("nvForest reported unknown precision {other}"),
                ));
            },
        };
        Ok(Self {
            num_features: count(raw.num_features, "feature count")?,
            num_trees: count(raw.num_trees, "tree count")?,
            native_output_stride: count(raw.num_outputs, "output count")?,
            precision,
        })
    }
}

/// Parses `bytes` on the host and proves nvForest accepts the model, returning
/// its facts and a Treelite checkpoint for [`ForestInferenceModel::import`].
/// Host parsing memory is outside any device grant.
pub fn inspect(bytes: &[u8], format: ForestModelFormat) -> Result<(ForestModelInfo, Vec<u8>)> {
    const OP: &str = "forest_inference::inspect";
    if bytes.is_empty() {
        return Err(error(ErrorCode::EmptyInput, FailurePhase::Preflight, OP, "forest model bytes are empty"));
    }
    let mut raw = empty_metadata();
    let mut checkpoint = std::ptr::null_mut();
    let mut len = 0;
    // SAFETY: bytes is live for the call; outputs are valid for writes.
    let status = unsafe {
        cuml_sys::cuml_forest_inspect(bytes.as_ptr(), bytes.len(), format.native(), &mut raw, &mut checkpoint, &mut len)
    };
    check_status(status, OP)?;
    if checkpoint.is_null() {
        return Err(error(
            ErrorCode::InvariantViolation,
            FailurePhase::NativeCall,
            OP,
            "inspect returned no checkpoint",
        ));
    }
    // SAFETY: the shim allocated `len` bytes at `checkpoint` and transfers ownership here.
    let owned = unsafe { std::slice::from_raw_parts(checkpoint, len) }.to_vec();
    // SAFETY: checkpoint came from cuml_forest_inspect and is freed once.
    unsafe { cuml_sys::cuml_forest_bytes_free(checkpoint) };
    Ok((ForestModelInfo::from_native(raw, OP)?, owned))
}

/// nvForest version compiled into the cuML shim from `NVForest_VERSION_*`.
pub fn forest_inference_version() -> Result<(u16, u16, u16)> {
    const OP: &str = "forest_inference::version";
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: outputs are valid for writes.
    check_status(unsafe { cuml_sys::cuml_forest_inference_version_get(&mut major, &mut minor, &mut patch) }, OP)?;
    Ok((major as u16, minor as u16, patch as u16))
}

struct NativeForest {
    raw: Option<cuml_sys::cuml_forest_model_t>,
    device: cudf::DeviceId,
}

impl NativeForest {
    fn close(&mut self) -> Result<()> {
        const OP: &str = "ForestInferenceModel::close";
        let Some(raw) = self.raw else { return Ok(()) };
        let _device = DeviceEntryGuard::enter(self.device, OP).map_err(|e| from_cudf(OP, e))?;
        // SAFETY: this wrapper uniquely owns the native model.
        let status = unsafe { cuml_sys::cuml_forest_close(raw) };
        self.raw = None;
        check_status(status, OP).map_err(|native| {
            error(ErrorCode::NativeReleaseFailed, FailurePhase::NativeRelease, OP, native.message().to_owned())
        })
    }
}

impl Drop for NativeForest {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// A forest resident on one cuML execution binding's device.
pub struct ForestInferenceModel {
    native: NativeForest,
    info: ForestModelInfo,
    kind: ForestOutputKind,
    binding_id: ExecutionBindingId,
    _runtime: Arc<Runtime>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl ForestInferenceModel {
    /// Uploads a checkpoint from [`inspect`] at its native precision. Device
    /// storage is allocated through the resources' query-domain route.
    pub fn import(resources: &CumlResources, checkpoint: &[u8], output: ForestOutput) -> Result<Self> {
        if checkpoint.is_empty() {
            return Err(error(ErrorCode::EmptyInput, FailurePhase::Preflight, IMPORT, "forest checkpoint is empty"));
        }
        resources.validate_exact_binding(IMPORT)?;
        let mut raw = std::ptr::null_mut();
        let mut metadata = empty_metadata();
        resources.with_native(IMPORT, |handle| {
            // SAFETY: checkpoint is live for the call; import stages its host copies before returning.
            check_status(
                unsafe {
                    cuml_sys::cuml_forest_import(
                        handle,
                        checkpoint.as_ptr(),
                        checkpoint.len(),
                        output.native(),
                        &mut raw,
                        &mut metadata,
                    )
                },
                IMPORT,
            )
        })?;
        Self::from_native(resources, raw, metadata, output, IMPORT)
    }

    /// Takes ownership of a model a native import returned on `resources`.
    pub(crate) fn from_native(
        resources: &CumlResources,
        raw: cuml_sys::cuml_forest_model_t,
        metadata: cuml_sys::cuml_forest_metadata_t,
        output: ForestOutput,
        op: &'static str,
    ) -> Result<Self> {
        if raw.is_null() {
            return Err(error(ErrorCode::InvariantViolation, FailurePhase::NativeCall, op, "import returned no model"));
        }
        // Owning the handle first releases the device forest if a fact check fails.
        let native = NativeForest { raw: Some(raw), device: resources.device() };
        let info = ForestModelInfo::from_native(metadata, op)?;
        let kind = info.output_kind(output)?;
        Ok(Self {
            native,
            info,
            kind,
            binding_id: resources.binding_id(),
            _runtime: Arc::clone(resources.runtime()),
            _not_send_sync: PhantomData,
        })
    }

    pub fn info(&self) -> ForestModelInfo {
        self.info
    }

    pub fn output_kind(&self) -> ForestOutputKind {
        self.kind
    }

    /// Enqueues inference over row-major features whose element type matches
    /// the forest precision. The input must stay alive until the returned
    /// prediction is assembled or its stream completes.
    pub fn predict(&self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<ForestPrediction> {
        let raw = self
            .native
            .raw
            .ok_or_else(|| error(ErrorCode::ModelClosed, FailurePhase::Preflight, PREDICT, "forest is closed"))?;
        if input.binding_id() != self.binding_id || resources.binding_id() != self.binding_id {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                PREDICT,
                "model, input, and resources must share an execution binding",
            ));
        }
        if input.layout() != MatrixLayout::RowMajor {
            return Err(error(
                ErrorCode::UnsupportedLayout,
                FailurePhase::Preflight,
                PREDICT,
                "forest inference requires row-major input",
            ));
        }
        if input.element() != self.info.precision.element() {
            return Err(error(
                ErrorCode::UnsupportedDtype,
                FailurePhase::Preflight,
                PREDICT,
                "input element type must match the forest precision",
            )
            .with_detail("expected", format!("{:?}", self.info.precision.element()))
            .with_detail("actual", format!("{:?}", input.element())));
        }
        if input.columns() != self.info.num_features {
            return Err(error(
                ErrorCode::ShapeMismatch,
                FailurePhase::Preflight,
                PREDICT,
                "input feature count differs from the model",
            )
            .with_detail("expected", self.info.num_features)
            .with_detail("actual", input.columns()));
        }
        if input.rows() == 0 {
            return Err(error(
                ErrorCode::EmptyInput,
                FailurePhase::Preflight,
                PREDICT,
                "prediction input must contain rows",
            ));
        }
        let rows = i32::try_from(input.rows())
            .map_err(|_| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, PREDICT, "row count exceeds i32"))?;
        let columns = i32::try_from(input.columns()).map_err(|_| {
            error(ErrorCode::SizeOverflow, FailurePhase::Preflight, PREDICT, "feature count exceeds i32")
        })?;
        let stride = self.info.native_output_stride;
        let len = input
            .rows()
            .checked_mul(stride)
            .ok_or_else(|| error(ErrorCode::SizeOverflow, FailurePhase::Preflight, PREDICT, "output size overflow"))?;
        resources.validate_exact_binding(PREDICT)?;
        let mut values = match self.info.precision {
            ForestPrecision::Single => PredictionValues::Single(DeviceBuffer::new(resources, len)?),
            ForestPrecision::Double => PredictionValues::Double(DeviceBuffer::new(resources, len)?),
        };
        let output = match &mut values {
            PredictionValues::Single(buffer) => buffer.as_mut_ptr()?,
            PredictionValues::Double(buffer) => buffer.as_mut_ptr()?,
        };
        resources.with_native(PREDICT, |handle| {
            // SAFETY: input and output are bound to the handle's stream and device,
            // sized rows x features and rows x stride of the forest precision.
            check_status(
                unsafe { cuml_sys::cuml_forest_predict(handle, raw, input.as_ptr(), rows, columns, output) },
                PREDICT,
            )
        })?;
        Ok(ForestPrediction {
            values,
            rows: input.rows(),
            stride,
            kind: self.kind,
            runtime: Arc::clone(resources.runtime()),
        })
    }

    /// Releases the device forest; the engine calls this before returning its
    /// execution grant. Drop repeats it as a fallback and ignores failures.
    pub fn close(&mut self) -> Result<()> {
        self.native.close()
    }
}

enum PredictionValues {
    Single(DeviceBuffer<f32>),
    Double(DeviceBuffer<f64>),
}

/// Native forest output awaiting validation and column assembly.
pub struct ForestPrediction {
    values: PredictionValues,
    rows: usize,
    stride: usize,
    kind: ForestOutputKind,
    runtime: Arc<Runtime>,
}

impl ForestPrediction {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn output_kind(&self) -> ForestOutputKind {
        self.kind
    }

    /// Validates and splits the output into `output_kind().width()` series containing
    /// either finite Float32 scores or Int32 class indices in `0..classes`.
    pub fn into_series(self) -> Result<Vec<Series>> {
        match self.kind {
            ForestOutputKind::Score | ForestOutputKind::Vector(_) => self.into_scores(),
            ForestOutputKind::ClassIndex { classes } => self.into_class_index(classes).map(|series| vec![series]),
        }
    }

    /// Reduces per-class probabilities to each row's most probable class and
    /// probability from a single inference pass. Requires at least two Float32
    /// scores per row. Scores outside `[0, 1]` fail validation.
    pub fn into_class_prediction(self) -> Result<ClassPrediction> {
        if !matches!(self.kind, ForestOutputKind::Vector(_)) {
            return Err(invalid_input(
                ASSEMBLE,
                "class probabilities require a model with at least two scores per row",
            ));
        }
        let runtime = Arc::clone(&self.runtime);
        ClassPrediction::from_probabilities(&runtime, self.into_scores()?, ASSEMBLE)
    }

    fn into_scores(self) -> Result<Vec<Series>> {
        let (flat, double) = match self.values {
            PredictionValues::Single(mut buffer) => {
                buffer.validate_finite(ASSEMBLE)?;
                (buffer.into_column()?, false)
            },
            PredictionValues::Double(mut buffer) => {
                buffer.validate_finite_float64(ASSEMBLE)?;
                (buffer.into_column()?, true)
            },
        };
        let series = self
            .runtime
            .unpack_dense_matrix(flat, self.rows, self.stride, DenseMatrixLayout::RowMajor)
            .map_err(|e| from_cudf(ASSEMBLE, e))?;
        if !double {
            return Ok(series);
        }
        series.iter().map(narrow_finite).collect()
    }

    fn into_class_index(self, classes: usize) -> Result<Series> {
        // nvForest's max_index writes each row's index into its first slot and
        // leaves the other stride slots unspecified; only slot 0 is read.
        let flat = match self.values {
            PredictionValues::Single(buffer) => buffer.into_partially_written_column()?,
            PredictionValues::Double(buffer) => buffer.into_partially_written_column()?,
        };
        let stride = i32::try_from(self.stride).map_err(|_| {
            error(ErrorCode::SizeOverflow, FailurePhase::OutputValidation, ASSEMBLE, "output stride exceeds i32")
        })?;
        let slots = self.runtime.sequence(0_i32, stride, self.rows).map_err(|e| from_cudf(ASSEMBLE, e))?;
        let indices = Series::from_column(flat)
            .and_then(|flat| flat.on(Exec::RETAINED).gather(&slots))
            .and_then(|values| values.on(Exec::RETAINED).astype(DataType::new(TypeId::Float64)))
            .map_err(|e| from_cudf(ASSEMBLE, e))?;
        let nan = indices
            .on(Exec::RETAINED)
            .is_nan()
            .and_then(|mask| mask.on(Exec::RETAINED).any_value())
            .map_err(|e| from_cudf(ASSEMBLE, e))?;
        let (min, max) = indices.on(Exec::RETAINED).minmax_values::<f64>().map_err(|e| from_cudf(ASSEMBLE, e))?;
        if nan || min.is_none_or(|v| v < 0.0) || max.is_none_or(|v| v >= classes as f64) {
            return Err(error(
                ErrorCode::OutputOutOfRange,
                FailurePhase::OutputValidation,
                ASSEMBLE,
                "forest class indices are outside the class range",
            )
            .with_detail("classes", classes)
            .with_detail("observed_min", format!("{min:?}"))
            .with_detail("observed_max", format!("{max:?}")));
        }
        indices.on(Exec::RETAINED).astype(DataType::new(TypeId::Int32)).map_err(|e| from_cudf(ASSEMBLE, e))
    }
}

fn narrow_finite(values: &Series) -> Result<Series> {
    let narrowed =
        values.on(Exec::RETAINED).astype(DataType::new(TypeId::Float32)).map_err(|e| from_cudf(ASSEMBLE, e))?;
    // Finite doubles beyond the Float32 range narrow to infinity.
    let (min, max) = narrowed.on(Exec::RETAINED).minmax_values::<f32>().map_err(|e| from_cudf(ASSEMBLE, e))?;
    if min.is_none_or(|v| !v.is_finite()) || max.is_none_or(|v| !v.is_finite()) {
        return Err(error(
            ErrorCode::OutputNotFinite,
            FailurePhase::OutputValidation,
            ASSEMBLE,
            "forest scores exceed the Float32 range",
        ));
    }
    Ok(narrowed)
}

pub(crate) fn empty_metadata() -> cuml_sys::cuml_forest_metadata_t {
    cuml_sys::cuml_forest_metadata_t { num_features: 0, num_trees: 0, num_outputs: 0, precision: 0 }
}

#[cfg(test)]
mod tests {
    use std::{ffi::CStr, fs, path::PathBuf, ptr::NonNull, sync::Arc};

    use cudf::Runtime;

    use super::{ForestInferenceModel, ForestModelFormat, ForestOutput, inspect};
    use crate::resources::CumlResources;

    #[test]
    fn native_predict_rejects_empty_and_32_bit_overflowing_shapes_before_dereferencing() {
        let runtime = Arc::new(Runtime::new().expect("runtime"));
        let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forest/xgb_binary_logistic.json");
        let (_, checkpoint) = inspect(&fs::read(path).unwrap(), ForestModelFormat::XgboostJson).unwrap();
        let model = ForestInferenceModel::import(&resources, &checkpoint, ForestOutput::Scores).unwrap();
        let raw = model.native.raw.unwrap();
        // i32::MAX rows x 4 features overflows nvForest's uint32 element offsets.
        for (rows, reason) in [(0, "rows or feature count"), (i32::MAX, "32-bit indexing")] {
            let dangling = NonNull::<f32>::dangling().as_ptr().cast();
            let (status, message) = resources
                .with_native("forest_inference::tests", |handle| {
                    let status = unsafe { cuml_sys::cuml_forest_predict(handle, raw, dangling, rows, 4, dangling) };
                    let message = unsafe { CStr::from_ptr(cuml_sys::cuml_last_error_message()) };
                    Ok((status, message.to_string_lossy().into_owned()))
                })
                .unwrap();
            assert_eq!(status, cuml_sys::CUML_ERROR_INVALID_ARGUMENT, "rows {rows}");
            assert!(message.contains(reason), "rows {rows}: {message}");
        }
    }
}
