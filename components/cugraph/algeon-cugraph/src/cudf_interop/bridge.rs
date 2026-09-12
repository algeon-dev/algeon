use std::ffi::c_void;

use cudf::{
    Runtime,
    core::{Column, DeviceSlice},
};

use super::{
    errors::map_cudf_error_with,
    types::{cudf_type_to_cugraph, cugraph_type_to_cudf},
};
use crate::{
    array::RawDeviceArrayView,
    error::{Error, ErrorKind, Result},
    types::CugraphElement,
};

fn validate_runtime_result_device(
    runtime: &Runtime,
    source_device: Option<i32>,
    operation: &'static str,
    message: &'static str,
) -> Result<()> {
    let Some(source_device) = source_device else {
        return Ok(());
    };

    let runtime_device = runtime.device().ordinal() as i32;
    if runtime_device == source_device {
        return Ok(());
    }

    Err(Error::new(ErrorKind::InvalidInput, operation)
        .with_message(message)
        .with_context("runtime_device", runtime_device.to_string())
        .with_context("result_device", source_device.to_string()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Checked DeviceSlice → DeviceArrayView; zero-copy bridge
// ═══════════════════════════════════════════════════════════════════════════

/// Create a typed [`crate::DeviceArrayView`] from a runtime-typed cudf [`DeviceSlice`],
/// verifying that the slice's element type matches `T` at runtime.
///
/// This is the safe zero-copy bridge between cudf and cugraph. The returned
/// view borrows the slice's GPU memory without copying.
///
/// # Errors
///
/// Returns [`ErrorKind::UnsupportedType`] if the slice's cudf type has no
/// cugraph equivalent, or if the mapped cugraph type does not match `T`.
///
/// # Example
///
/// ```no_run
/// use cugraph::cudf_interop::device_array_view_from_slice;
///
/// let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
/// let series = runtime.series(&[0i32, 1, 2]).unwrap();
/// let col = series.as_column();
/// let slice = col.device_slice::<i32>().unwrap();
///
/// let view = device_array_view_from_slice::<i32>(&slice).unwrap();
/// assert_eq!(view.len(), 3);
/// ```
pub fn device_array_view_from_slice<'a, T: CugraphElement>(
    slice: &'a DeviceSlice<'a>,
) -> Result<crate::array::DeviceArrayView<'a, T>> {
    let cudf_tid = slice.type_id();
    let cugraph_dtype = cudf_type_to_cugraph(cudf_tid).ok_or_else(|| {
        Error::new(ErrorKind::UnsupportedType, "device_array_view_from_slice")
            .with_message(format!("cudf TypeId {:?} has no cugraph equivalent", cudf_tid))
    })?;

    let expected = T::data_type();
    if cugraph_dtype != expected {
        return Err(Error::new(ErrorKind::UnsupportedType, "device_array_view_from_slice")
            .with_message(format!(
                "dtype mismatch: slice contains {:?} but requested type maps to {:?}",
                cugraph_dtype, expected
            ))
            .with_context("slice_dtype", format!("{:?}", cugraph_dtype))
            .with_context("requested_dtype", format!("{:?}", expected)));
    }

    // SAFETY: we have verified that the slice's runtime dtype matches T.
    // DeviceSlice guarantees a valid device pointer for its lifetime.
    let raw = unsafe {
        crate::array::RawDeviceArrayView::try_from_raw(
            slice.as_ptr() as *mut c_void,
            slice.len(),
            expected,
            "device_array_view_from_slice",
        )?
    };
    Ok(crate::array::DeviceArrayView::from_raw_unchecked(raw))
}

// ═══════════════════════════════════════════════════════════════════════════
// RawDeviceArrayView → cudf Column
// ═══════════════════════════════════════════════════════════════════════════

impl RawDeviceArrayView<'_> {
    /// Convert this cugraph device array into a cudf [`Column`].
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    ///
    /// [`Column`]: cudf::Column
    pub(super) fn to_cudf_column(&self, runtime: &Runtime) -> Result<Column> {
        let dtype = self.dtype()?;
        let cudf_type = cugraph_type_to_cudf(dtype).ok_or_else(|| {
            Error::new(ErrorKind::UnsupportedType, "RawDeviceArrayView::to_cudf_column")
                .with_message(format!("cugraph DataType {:?} has no cudf TypeId equivalent", dtype))
        })?;

        validate_runtime_result_device(
            runtime,
            self.device(),
            "RawDeviceArrayView::to_cudf_column",
            "runtime device must match the cugraph result device",
        )?;

        // SAFETY: `self.device_ptr()` points to valid GPU memory containing
        // `self.len()` elements of `dtype`, which we have verified maps to
        // `cudf_type`. The explicit runtime owns the destination allocation
        // and execution domain for the device-to-device copy.
        let col = unsafe { runtime.column_from_device_ptr(self.device_ptr() as *const c_void, self.len(), cudf_type) }
            .map_err(|error| {
                map_cudf_error_with(
                    error,
                    "RawDeviceArrayView::to_cudf_column",
                    "failed to materialize cudf Column from cugraph device array",
                    [("len", self.len().to_string()), ("dtype", format!("{:?}", dtype))],
                )
            })?;

        Ok(col)
    }
}
