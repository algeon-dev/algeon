mod pca;
mod tsvd;

use cudf::runtime::ExecutionBindingId;
pub use pca::{PcaModel, PcaParams, PcaSolver};
pub use tsvd::{TsvdModel, TsvdParams};

use crate::{
    DeviceMatrix, ErrorCode, FailurePhase, MatrixLayout, Result,
    error::{check_status, error, invalid_input},
    resources::CumlResources,
};

unsafe extern "C" {
    fn cuml_pca_fit(
        handle: cuml_sys::cuml_handle_t,
        input: *mut f32,
        rows: i64,
        columns: i64,
        components: i64,
        algorithm: i32,
        tolerance: f32,
        iterations: u32,
        copy: i32,
        whiten: i32,
        output_components: *mut f32,
        explained_variance: *mut f32,
        explained_variance_ratio: *mut f32,
        singular_values: *mut f32,
        mean: *mut f32,
        noise_variance: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_pca_fit_transform(
        handle: cuml_sys::cuml_handle_t,
        input: *mut f32,
        rows: i64,
        columns: i64,
        components: i64,
        algorithm: i32,
        tolerance: f32,
        iterations: u32,
        copy: i32,
        whiten: i32,
        transformed: *mut f32,
        output_components: *mut f32,
        explained_variance: *mut f32,
        explained_variance_ratio: *mut f32,
        singular_values: *mut f32,
        mean: *mut f32,
        noise_variance: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_pca_transform(
        handle: cuml_sys::cuml_handle_t,
        input: *const f32,
        rows: i64,
        columns: i64,
        components: i64,
        whiten: i32,
        model_components: *const f32,
        singular_values: *const f32,
        mean: *const f32,
        transformed: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_pca_inverse_transform(
        handle: cuml_sys::cuml_handle_t,
        transformed: *const f32,
        rows: i64,
        columns: i64,
        components: i64,
        whiten: i32,
        model_components: *const f32,
        singular_values: *const f32,
        mean: *const f32,
        output: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_tsvd_fit_transform(
        handle: cuml_sys::cuml_handle_t,
        input: *mut f32,
        rows: i64,
        columns: i64,
        components: i64,
        algorithm: i32,
        tolerance: f32,
        iterations: u32,
        transformed: *mut f32,
        output_components: *mut f32,
        explained_variance: *mut f32,
        explained_variance_ratio: *mut f32,
        singular_values: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_tsvd_transform(
        handle: cuml_sys::cuml_handle_t,
        input: *const f32,
        rows: i64,
        columns: i64,
        components: i64,
        model_components: *const f32,
        transformed: *mut f32,
    ) -> cuml_sys::cuml_status_t;
    fn cuml_tsvd_inverse_transform(
        handle: cuml_sys::cuml_handle_t,
        transformed: *const f32,
        rows: i64,
        columns: i64,
        components: i64,
        model_components: *const f32,
        output: *mut f32,
    ) -> cuml_sys::cuml_status_t;
}

fn dimensions(rows: usize, columns: usize, components: usize, op: &'static str) -> Result<()> {
    if rows < 2 || columns < 2 || components == 0 || components > columns {
        return Err(invalid_input(op, "expected rows and columns >= 2 and components in 1..=columns"));
    }
    if rows > i64::MAX as usize
        || columns > i64::MAX as usize
        || components > i64::MAX as usize
        || columns.checked_mul(components).is_none()
        || rows.checked_mul(components).is_none()
    {
        return Err(error(
            ErrorCode::SizeOverflow,
            FailurePhase::Preflight,
            op,
            "matrix extent exceeds the supported range",
        ));
    }
    Ok(())
}

fn column_major(layout: MatrixLayout, op: &'static str) -> Result<()> {
    if layout != MatrixLayout::ColumnMajor {
        return Err(error(
            ErrorCode::UnsupportedLayout,
            FailurePhase::Preflight,
            op,
            "PCA and TruncatedSVD require column-major matrices",
        ));
    }
    Ok(())
}

fn validate_model_input(
    resources: &CumlResources,
    input: &DeviceMatrix<'_>,
    binding_id: ExecutionBindingId,
    n_features: usize,
    n_components: usize,
    expected_columns: usize,
    op: &'static str,
) -> Result<()> {
    dimensions(input.rows(), n_features, n_components, op)?;
    column_major(input.layout(), op)?;
    if input.columns() != expected_columns || input.binding_id() != binding_id || resources.binding_id() != binding_id {
        return Err(error(
            ErrorCode::RuntimeBindingMismatch,
            FailurePhase::Preflight,
            op,
            "model, input, and resources must share a binding and expected feature shape",
        ));
    }
    resources.validate_exact_binding(op)
}

fn native(
    resources: &CumlResources,
    op: &'static str,
    call: impl FnOnce(cuml_sys::cuml_handle_t) -> cuml_sys::cuml_status_t,
) -> Result<()> {
    resources.with_native(op, |handle| check_status(call(handle), op))
}
