use std::sync::Arc;

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

pub(crate) fn trigger_fixed_width_host_byte_range_overflow_for_test() -> Result<()> {
    Err(super::fixed_width_host_byte_range_overflow_error(
        "testing::column_view::to_fixed_width_host_bytes",
        usize::MAX,
        2,
        TypeId::Int16,
    ))
}
