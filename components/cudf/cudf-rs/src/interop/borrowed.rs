pub use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, BoundBytes, BufferBinding, BufferId, ColumnId, ColumnLayout,
    CudaEventHandle, CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle, ExecutionBinding,
    PhysicalValidity, Placement, SchemaNullability, SemanticColumn, SemanticTable, TableId, TryBorrowColumnView,
    TryBorrowTableView, TypeId as InteropTypeId,
};

use super::*;

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
    let projection = prepared_column_projection_from_view(view, operation, execution.device(), 0, 0)?;
    BorrowedColumnView::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        execution,
    )
    .map_err(|error| error.with_context("operation", operation))
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
