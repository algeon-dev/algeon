use crate::{
    device::{DeviceMatrix, MatrixLayout, OwnedFloat32Matrix},
    error::{ErrorCode, FailurePhase, Result, check_status, error, invalid_input},
    resources::CumlResources,
};

const FIT: &str = "UmapModel::fit";
const TRANSFORM: &str = "UmapModel::transform";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UmapInit {
    Spectral,
    Random,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UmapParams {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub n_epochs: usize,
    pub learning_rate: f32,
    pub min_dist: f32,
    pub spread: f32,
    pub set_op_mix_ratio: f32,
    pub local_connectivity: f32,
    pub repulsion_strength: f32,
    pub negative_sample_rate: usize,
    pub transform_queue_size: f32,
    pub init: UmapInit,
    pub random_state: u64,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: 0,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            set_op_mix_ratio: 1.0,
            local_connectivity: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            transform_queue_size: 4.0,
            init: UmapInit::Spectral,
            random_state: 0,
        }
    }
}

// Keep the field order and integer widths aligned with cuml_rust/umap.h;
// the native ABI consumes this struct through a pointer, without conversion.
#[repr(C)]
struct NativeUmapParams {
    n_neighbors: i32,
    n_components: i32,
    n_epochs: i32,
    learning_rate: f32,
    min_dist: f32,
    spread: f32,
    set_op_mix_ratio: f32,
    local_connectivity: f32,
    repulsion_strength: f32,
    negative_sample_rate: i32,
    transform_queue_size: f32,
    init: i32,
    random_state: u64,
}

unsafe extern "C" {
    fn cuml_umap_fit_params(
        handle: cuml_sys::cuml_handle_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        params: *const NativeUmapParams,
        embedding: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_umap_transform_params(
        handle: cuml_sys::cuml_handle_t,
        input: *const f32,
        rows: i32,
        columns: i32,
        training_input: *const f32,
        training_rows: i32,
        embedding: *const f32,
        embedding_rows: i32,
        params: *const NativeUmapParams,
        output: *mut f32,
    ) -> cuml_sys::cuml_status_t;
}

fn native_parameters(parameters: UmapParams) -> Result<NativeUmapParams> {
    let n_neighbors =
        i32::try_from(parameters.n_neighbors).map_err(|_| invalid_input(FIT, "n_neighbors exceeds cuML limits"))?;
    let n_components =
        i32::try_from(parameters.n_components).map_err(|_| invalid_input(FIT, "n_components exceeds cuML limits"))?;
    let n_epochs =
        i32::try_from(parameters.n_epochs).map_err(|_| invalid_input(FIT, "n_epochs exceeds cuML limits"))?;
    let negative_sample_rate = i32::try_from(parameters.negative_sample_rate)
        .map_err(|_| invalid_input(FIT, "negative_sample_rate exceeds cuML limits"))?;
    Ok(NativeUmapParams {
        n_neighbors,
        n_components,
        n_epochs,
        learning_rate: parameters.learning_rate,
        min_dist: parameters.min_dist,
        spread: parameters.spread,
        set_op_mix_ratio: parameters.set_op_mix_ratio,
        local_connectivity: parameters.local_connectivity,
        repulsion_strength: parameters.repulsion_strength,
        negative_sample_rate,
        transform_queue_size: parameters.transform_queue_size,
        init: match parameters.init {
            UmapInit::Spectral => 1,
            UmapInit::Random => 0,
        },
        random_state: parameters.random_state,
    })
}

/// A fitted UMAP model retaining its training matrix for future transforms.
pub struct UmapModel {
    training: OwnedFloat32Matrix,
    training_rows: usize,
    columns: usize,
    parameters: UmapParams,
    embedding: OwnedFloat32Matrix,
}

impl UmapModel {
    pub fn embedding(&self) -> &OwnedFloat32Matrix {
        &self.embedding
    }

    pub fn transform(&mut self, resources: &CumlResources, input: &DeviceMatrix<'_>) -> Result<OwnedFloat32Matrix> {
        if input.binding_id() != resources.binding_id()
            || self.training.binding_id() != resources.binding_id()
            || self.embedding.binding_id() != resources.binding_id()
        {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                TRANSFORM,
                "UMAP input and model must use the resource binding",
            ));
        }
        validate_matrix(input, TRANSFORM)?;
        if input.columns() != self.columns {
            return Err(error(
                ErrorCode::ShapeMismatch,
                FailurePhase::Preflight,
                TRANSFORM,
                "transform feature count differs from the fitted matrix",
            ));
        }
        let rows =
            i32::try_from(input.rows()).map_err(|_| invalid_input(TRANSFORM, "row count exceeds cuML limits"))?;
        let training_rows = i32::try_from(self.training_rows)
            .map_err(|_| invalid_input(TRANSFORM, "training row count exceeds cuML limits"))?;
        let columns =
            i32::try_from(self.columns).map_err(|_| invalid_input(TRANSFORM, "feature count exceeds cuML limits"))?;
        validate_parameters(self.parameters, self.training_rows)?;
        let parameters = native_parameters(self.parameters)?;
        let mut output =
            OwnedFloat32Matrix::new(resources, input.rows(), self.parameters.n_components, MatrixLayout::RowMajor)?;
        let output_ptr = output.as_mut_ptr()?.cast();
        // Native transform consumes the fitted inputs again. Retain both owners
        // and revalidate their contents before returning from this mutable call.
        let training_ptr = self.training.as_mut_ptr()?.cast::<f32>().cast_const();
        let embedding_ptr = self.embedding.as_mut_ptr()?.cast::<f32>().cast_const();
        resources.with_native(TRANSFORM, |handle| {
            let status = unsafe {
                cuml_umap_transform_params(
                    handle,
                    input.as_ptr().cast(),
                    rows,
                    columns,
                    training_ptr,
                    training_rows,
                    embedding_ptr,
                    training_rows,
                    &parameters,
                    output_ptr,
                )
            };
            check_status(status, TRANSFORM)
        })?;
        output.validate_finite(TRANSFORM)?;
        self.embedding.validate_finite(TRANSFORM)?;
        self.training.validate_finite(TRANSFORM)?;
        Ok(output)
    }

    pub fn fit(resources: &CumlResources, mut training: OwnedFloat32Matrix, parameters: UmapParams) -> Result<Self> {
        validate_owned_matrix(&training, FIT)?;
        validate_parameters(parameters, training.rows())?;
        if training.binding_id() != resources.binding_id() {
            return Err(error(
                ErrorCode::RuntimeBindingMismatch,
                FailurePhase::Preflight,
                FIT,
                "training matrix must use the resource binding",
            ));
        }
        training.validate_finite(FIT)?;
        let rows = i32::try_from(training.rows()).map_err(|_| invalid_input(FIT, "row count exceeds cuML limits"))?;
        let columns =
            i32::try_from(training.columns()).map_err(|_| invalid_input(FIT, "feature count exceeds cuML limits"))?;
        let native_parameters = native_parameters(parameters)?;
        let training_rows = training.rows();
        let feature_count = training.columns();
        let training_ptr = training.as_mut_ptr()?.cast::<f32>().cast_const();
        let mut embedding =
            OwnedFloat32Matrix::new(resources, training_rows, parameters.n_components, MatrixLayout::RowMajor)?;
        let output_ptr = embedding.as_mut_ptr()?.cast();
        resources.with_native(FIT, |handle| {
            let status =
                unsafe { cuml_umap_fit_params(handle, training_ptr, rows, columns, &native_parameters, output_ptr) };
            check_status(status, FIT)
        })?;
        embedding.validate_finite(FIT)?;
        Ok(Self { training, training_rows, columns: feature_count, parameters, embedding })
    }
}

fn validate_matrix(matrix: &DeviceMatrix<'_>, op: &'static str) -> Result<()> {
    if matrix.rows() == 0 || matrix.columns() == 0 {
        return Err(invalid_input(op, "matrix dimensions must be nonzero"));
    }
    if matrix.layout() != MatrixLayout::RowMajor {
        return Err(error(
            ErrorCode::UnsupportedLayout,
            FailurePhase::Preflight,
            op,
            "UMAP requires row-major float32 input",
        ));
    }
    Ok(())
}

fn validate_owned_matrix(matrix: &OwnedFloat32Matrix, op: &'static str) -> Result<()> {
    if matrix.rows() == 0 || matrix.columns() == 0 {
        return Err(invalid_input(op, "matrix dimensions must be nonzero"));
    }
    if matrix.layout() != MatrixLayout::RowMajor {
        return Err(error(
            ErrorCode::UnsupportedLayout,
            FailurePhase::Preflight,
            op,
            "UMAP requires row-major float32 input",
        ));
    }
    Ok(())
}

fn validate_parameters(parameters: UmapParams, rows: usize) -> Result<()> {
    if rows < 2 || parameters.n_neighbors == 0 || parameters.n_neighbors >= rows {
        return Err(invalid_input(FIT, "n_neighbors must be in 1..training rows"));
    }
    if parameters.n_components == 0 {
        return Err(invalid_input(FIT, "n_components must be nonzero"));
    }
    if parameters.negative_sample_rate == 0 {
        return Err(invalid_input(FIT, "negative_sample_rate must be positive"));
    }
    let finite_nonnegative = [
        ("min_dist", parameters.min_dist),
        ("set_op_mix_ratio", parameters.set_op_mix_ratio),
        ("local_connectivity", parameters.local_connectivity),
        ("repulsion_strength", parameters.repulsion_strength),
    ];
    if finite_nonnegative.iter().any(|(_, value)| !value.is_finite() || *value < 0.0) {
        return Err(invalid_input(FIT, "UMAP parameters must be finite and nonnegative"));
    }
    if !parameters.learning_rate.is_finite() || parameters.learning_rate <= 0.0 {
        return Err(invalid_input(FIT, "learning_rate must be finite and positive"));
    }
    if !parameters.spread.is_finite() || parameters.spread <= 0.0 {
        return Err(invalid_input(FIT, "spread must be finite and positive"));
    }
    if parameters.set_op_mix_ratio > 1.0 {
        return Err(invalid_input(FIT, "set_op_mix_ratio must not exceed 1"));
    }
    if !parameters.transform_queue_size.is_finite() || parameters.transform_queue_size <= 0.0 {
        return Err(invalid_input(FIT, "transform_queue_size must be finite and positive"));
    }
    i32::try_from(parameters.n_neighbors).map_err(|_| invalid_input(FIT, "n_neighbors exceeds cuML limits"))?;
    i32::try_from(parameters.n_components).map_err(|_| invalid_input(FIT, "n_components exceeds cuML limits"))?;
    i32::try_from(parameters.n_epochs).map_err(|_| invalid_input(FIT, "n_epochs exceeds cuML limits"))?;
    i32::try_from(parameters.negative_sample_rate)
        .map_err(|_| invalid_input(FIT, "negative_sample_rate exceeds cuML limits"))?;
    Ok(())
}
