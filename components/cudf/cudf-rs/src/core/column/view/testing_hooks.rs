use std::{ffi::c_void, sync::Arc};

use super::ColumnView;
use crate::{
    core::{
        error::{self, Result},
        types::TypeId,
    },
    runtime::Runtime,
};

pub(crate) fn synthetic_string_view_without_children(size: usize) -> Result<ColumnView<'static>> {
    let runtime = Runtime::new()?;
    let size = error::usize_to_i32(size, "testing::synthetic_string_view_without_children", "row_count")?;
    let mut handle = std::ptr::null_mut();
    error::check_status_in(
        unsafe { cudf_sys::cudf_test_column_view_create_synthetic_string_without_children(size, &mut handle) },
        "testing::synthetic_string_view_without_children",
    )?;
    unsafe { ColumnView::from_raw_with_domain(handle, Arc::clone(runtime.execution_domain())) }
}

pub(crate) fn column_view_to_raw_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    view.to_raw_bytes()
}

pub(crate) fn column_view_to_fixed_width_host_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    view.to_fixed_width_host_bytes()
}

pub(crate) fn column_view_copy_fixed_width_to_host_for_benchmark(
    view: &ColumnView<'_>,
    host_ptr: *mut u8,
    host_capacity: usize,
) -> Result<usize> {
    let operation = "testing::column_view::copy_fixed_width_to_host_for_benchmark";
    let dtype = view.data_type().map_err(|error| error.reanchor(operation))?;
    let elem_size = usize::try_from(dtype.id().size_in_bytes()?.ok_or_else(|| {
        crate::core::Error::new_with(
            crate::core::ErrorCode::UnsupportedDtype,
            operation,
            "D2H benchmark copies require a fixed-width type",
        )
        .with_fact(crate::core::error::FACT_TYPE_ID, dtype.id())
    })?)
    .map_err(|_| {
        crate::core::Error::new_with(
            crate::core::ErrorCode::ForeignInvariantViolation,
            operation,
            "fixed-width element size must be non-negative",
        )
        .with_fact(crate::core::error::FACT_TYPE_ID, dtype.id())
    })?;
    let rows = view.len().map_err(|error| error.reanchor(operation))?;
    let copy_bytes = rows
        .checked_mul(elem_size)
        .ok_or_else(|| super::fixed_width_host_byte_range_overflow_error(operation, rows, elem_size, dtype.id()))?;
    if copy_bytes > host_capacity {
        return Err(crate::core::Error::new_with(
            crate::core::ErrorCode::IndexOutOfBounds,
            operation,
            "D2H benchmark host buffer is smaller than the active column payload",
        )
        .with_detail("copy_bytes", copy_bytes.to_string())
        .with_detail("host_capacity", host_capacity.to_string()));
    }
    if copy_bytes == 0 {
        return Ok(0);
    }

    // Match the production fixed-width host-materialization contract exactly:
    // establish host visibility first, then use the same FFI copy whose
    // selected stream is synchronized before it returns. Only the destination
    // allocation kind differs between benchmark lanes.
    view.synchronize_host_visibility(operation)?;
    let exec_options = view.domain.exec_options_raw();
    error::check_status_in(
        unsafe {
            cudf_sys::cudf_column_view_to_host_ex(
                view.handle.as_ptr(),
                host_ptr.cast::<c_void>(),
                copy_bytes,
                &exec_options,
            )
        },
        operation,
    )?;
    Ok(copy_bytes)
}

pub(crate) fn trigger_fixed_width_host_byte_range_overflow_for_test() -> Result<()> {
    Err(super::fixed_width_host_byte_range_overflow_error(
        "testing::column_view::to_fixed_width_host_bytes",
        usize::MAX,
        2,
        TypeId::Int16,
    ))
}
