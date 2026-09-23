use crate::{
    brute_force::{matrix_layout, validate_matrix_binding},
    device::{DeviceBuffer, MatrixLayout, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input, require_nonzero},
    resources::CuvsResources,
};

/// Selects `k` sorted values per row, writing `[batch, k]` row-major values
/// and zero-based positions. Equal-value selection at the k boundary is unspecified.
pub fn select_k_into(
    resources: &CuvsResources,
    values: &ValidatedDeviceTensor<'_, f32>,
    k: usize,
    select_min: bool,
    out_values: &mut DeviceBuffer<f32>,
    out_positions: &mut DeviceBuffer<i64>,
) -> Result<()> {
    let operation = "select_k_into";
    validate_matrix_binding(resources, values, operation)?;
    if matrix_layout(values, operation)? != MatrixLayout::RowMajor {
        return Err(invalid_input(operation, "select_k requires a row-major input"));
    }
    let batch = values.rows()?;
    let len = values.columns()?;
    require_nonzero(batch, operation, "batch")?;
    require_nonzero(k, operation, "k")?;
    if k > len {
        return Err(invalid_input(operation, "k must not exceed the vector dimension"));
    }
    let output_len = batch.checked_mul(k).ok_or_else(|| invalid_input(operation, "output size overflow"))?;
    if out_values.len() != output_len || out_positions.len() != output_len {
        return Err(invalid_input(operation, "output buffers must contain batch * k elements"));
    }
    let mut output_values = out_values.as_tensor_mut(resources, &[batch, k], None, 0)?;
    let mut output_positions = out_positions.as_tensor_mut(resources, &[batch, k], None, 0)?;
    resources.with_native(operation, |native| {
        let mut input = DlpackTensor::from_const(values.as_tensor());
        let mut output_values = DlpackTensor::from_mut(&mut output_values);
        let mut output_positions = DlpackTensor::from_mut(&mut output_positions);
        unsafe {
            check_cuvs(
                cuvs_sys::cuvsSelectKBatched(
                    native,
                    input.as_mut_ptr(),
                    output_values.as_mut_ptr(),
                    output_positions.as_mut_ptr(),
                    select_min,
                    true,
                ),
                operation,
            )
        }
    })?;
    out_positions.validate_native_index_range(
        0,
        i64::try_from(len).map_err(|_| invalid_input(operation, "dimension exceeds Int64"))?,
        operation,
    )?;
    out_values.validate_native_input(operation)
}
