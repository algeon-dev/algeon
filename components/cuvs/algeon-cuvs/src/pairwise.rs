use crate::{
    brute_force::{DistanceMetric, matrix_layout, validate_matrix_binding},
    device::{DeviceBuffer, MatrixLayout, ValidatedDeviceTensor},
    dlpack::DlpackTensor,
    error::{Result, check_cuvs, invalid_input},
    resources::CuvsResources,
};

/// Writes the row-major `[x.rows(), y.rows()]` metric matrix to `out`.
/// Both inputs must share dimensions, layout, binding, and device.
pub fn pairwise_distance_into(
    resources: &CuvsResources,
    x: &ValidatedDeviceTensor<'_>,
    y: &ValidatedDeviceTensor<'_>,
    metric: DistanceMetric,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let operation = "pairwise_distance_into";
    validate_matrix_binding(resources, x, operation)?;
    validate_matrix_binding(resources, y, operation)?;
    if matrix_layout(x, operation)? != MatrixLayout::RowMajor
        || matrix_layout(y, operation)? != MatrixLayout::RowMajor
        || x.columns()? != y.columns()?
    {
        return Err(invalid_input(operation, "pairwise inputs must share layout and vector dimension"));
    }
    let rows = x.rows()?;
    let columns = y.rows()?;
    let expected = rows.checked_mul(columns).ok_or_else(|| invalid_input(operation, "output size overflow"))?;
    if out.len() != expected || rows == 0 || columns == 0 {
        return Err(invalid_input(operation, "output must contain x.rows * y.rows nonempty elements"));
    }
    let mut output = out.as_tensor_mut(resources, &[rows, columns], None, 0)?;
    resources.with_native(operation, |native| {
        let mut x_tensor = DlpackTensor::from_const(x.as_tensor());
        let mut y_tensor = DlpackTensor::from_const(y.as_tensor());
        let mut output_tensor = DlpackTensor::from_mut(&mut output);
        unsafe {
            check_cuvs(
                cuvs_sys::cuvsPairwiseDistance(
                    native,
                    x_tensor.as_mut_ptr(),
                    y_tensor.as_mut_ptr(),
                    output_tensor.as_mut_ptr(),
                    metric.raw(),
                    0.0,
                ),
                operation,
            )
        }
    })?;
    out.validate_native_input(operation)
}
