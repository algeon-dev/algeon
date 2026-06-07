use super::*;

pub(super) fn borrow_from_materialized_view<'a>(
    operation: &'static str,
    data_type: DataType,
    num_children: i32,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    null_mask: NullMask<'a>,
    execution: ExecutionBinding<'a>,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let interop_type = map_data_type(data_type);
    if interop_type.id().fixed_width_size_bytes().is_none() {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "borrowed interop v1 supports fixed-width column types only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
    }
    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "borrowed interop v1 does not support child-bearing column layouts",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
    }

    BorrowedColumnView::try_new(interop_type, len, offset, data_ptr, null_mask, execution)
        .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn null_mask_for_view<'a>(
    operation: &'static str,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    offset: usize,
    len: usize,
    null_count: usize,
) -> Result<NullMask<'a>, InteropError> {
    if !nullable {
        return Ok(NullMask::absent());
    }

    let bit_len = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "null mask bit length overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;

    let ptr = null_mask_ptr.ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "nullable column did not expose a null-mask pointer")
            .with_context("operation", operation)
    })?;

    NullMask::present(ptr, bit_len, null_count).map_err(|error| error.with_context("operation", operation))
}

/// Extract all fields from a `ColumnView` and produce a `BorrowedColumnView`.
///
/// The execution binding is provided externally so that callers can choose
/// whether to derive it from the view itself or from the owning column/table.
/// The returned view's lifetime is tied to `'a` (the execution binding and the
/// device memory it describes), not to the `ColumnView` borrow.
pub(super) fn borrow_from_view_with_execution<'a>(
    view: &ColumnView<'_>,
    execution: ExecutionBinding<'a>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset = usize::try_from(
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
            .with_context("operation", operation)
    })?;
    let data_ptr =
        view.device_data_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask = null_mask_for_view(operation, nullable, null_mask_ptr, offset, len, null_count)?;

    borrow_from_materialized_view(operation, data_type, num_children, len, offset, data_ptr, null_mask, execution)
}

pub(super) fn borrow_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let execution = execution_binding(operation, view.execution_domain())?;
    borrow_from_view_with_execution(view, execution, operation)
}

pub(super) fn borrow_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(operation, column.execution_domain())?;
    borrow_from_view_with_execution(&view, execution, operation)
}
