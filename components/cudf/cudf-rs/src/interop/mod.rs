//! External interop boundaries.
//!
//! This module re-exports the shared borrowed GPU-view ABI from
//! [`rapids_interop`] and implements `cudf-rs` producer adapters for:
//!
//! - [`crate::ColumnView`]
//! - [`crate::TableView`]
//! - [`crate::Series`]
//! - [`crate::DataFrame`]
//!
//! # Contract Summary
//!
//! - v1 is a borrowed GPU-view ABI.
//! - Borrowing does not transfer ownership.
//! - Borrowing avoids host copies on the input path and is zero-copy where
//!   possible because it describes existing device buffers.
//! - Consumer algorithms may still allocate or copy for outputs; this module
//!   does not promise zero-copy output.
//! - v1 supports only fixed-width types.
//! - `String`, `List`, `Struct`, and other nested or child-bearing layouts are
//!   unsupported in borrowed interop v1.
//! - [`crate::Series`] and [`crate::DataFrame`] are convenience adapters, not
//!   the ABI itself.
//! - Explicit execution provenance is required for producer projection.
//! - Ambient/default-only objects are rejected at the interop boundary.
//! - Execution binding semantics are part of the contract and are preserved in
//!   the returned borrowed descriptors.

use std::sync::Arc;

pub use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle,
    ExecutionBinding, InteropError, InteropErrorKind, NullMask, NullMaskKind, TryBorrowColumnView, TryBorrowTableView,
    TypeId as InteropTypeId,
};

use crate::{
    core::{
        column::{Column, ColumnView},
        table::TableView,
        types::{DataType, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    runtime::execution_domain::ExecutionDomain,
};

fn map_type_id(type_id: TypeId) -> InteropTypeId {
    match type_id {
        TypeId::Empty => InteropTypeId::Empty,
        TypeId::Int8 => InteropTypeId::Int8,
        TypeId::Int16 => InteropTypeId::Int16,
        TypeId::Int32 => InteropTypeId::Int32,
        TypeId::Int64 => InteropTypeId::Int64,
        TypeId::UInt8 => InteropTypeId::UInt8,
        TypeId::UInt16 => InteropTypeId::UInt16,
        TypeId::UInt32 => InteropTypeId::UInt32,
        TypeId::UInt64 => InteropTypeId::UInt64,
        TypeId::Float32 => InteropTypeId::Float32,
        TypeId::Float64 => InteropTypeId::Float64,
        TypeId::Bool8 => InteropTypeId::Bool8,
        TypeId::TimestampDays => InteropTypeId::TimestampDays,
        TypeId::TimestampSeconds => InteropTypeId::TimestampSeconds,
        TypeId::TimestampMilliseconds => InteropTypeId::TimestampMilliseconds,
        TypeId::TimestampMicroseconds => InteropTypeId::TimestampMicroseconds,
        TypeId::TimestampNanoseconds => InteropTypeId::TimestampNanoseconds,
        TypeId::DurationDays => InteropTypeId::DurationDays,
        TypeId::DurationSeconds => InteropTypeId::DurationSeconds,
        TypeId::DurationMilliseconds => InteropTypeId::DurationMilliseconds,
        TypeId::DurationMicroseconds => InteropTypeId::DurationMicroseconds,
        TypeId::DurationNanoseconds => InteropTypeId::DurationNanoseconds,
        TypeId::Dictionary32 => InteropTypeId::Dictionary32,
        TypeId::String => InteropTypeId::String,
        TypeId::List => InteropTypeId::List,
        TypeId::Decimal32 => InteropTypeId::Decimal32,
        TypeId::Decimal64 => InteropTypeId::Decimal64,
        TypeId::Decimal128 => InteropTypeId::Decimal128,
        TypeId::Struct => InteropTypeId::Struct,
    }
}

fn map_data_type(data_type: DataType) -> InteropDataType {
    if data_type.id().is_decimal() {
        InteropDataType::fixed_point(map_type_id(data_type.id()), data_type.scale())
    } else {
        InteropDataType::new(map_type_id(data_type.id()))
    }
}

fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let mut interop = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        interop = interop.with_context(key, value.clone());
    }
    interop
}

fn execution_binding<'a>(
    operation: &'static str,
    domain: &Arc<ExecutionDomain>,
) -> Result<ExecutionBinding<'a>, InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Err(InteropError::new(
            InteropErrorKind::AmbientInteropUnsupported,
            "borrowed interop requires explicit execution provenance; ambient/default-only objects are not valid producers",
        )
        .with_context("operation", operation)
        .with_context("policy", "ambient_interop_unsupported")
        .with_context("device_snapshot", domain.device_snapshot().to_string())),
        Some(options) => {
            let stream = domain.explicit_stream().expect("explicit domain missing stream");
            let mr = domain.explicit_memory_resource().expect("explicit domain missing memory resource");
            let stream_ptr = stream.interop_raw_handle().map_err(|error| {
                map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error)
            })?;
            let mr_ptr = mr.interop_raw_handle().map_err(|error| {
                map_cudf_error(InteropErrorKind::ExecutionBindingUnavailable, operation, error)
            })?;
            Ok(ExecutionBinding::explicit(
                options.device,
                CudaStreamHandle::from_raw(stream_ptr)?,
                DeviceMemoryResourceHandle::from_raw(mr_ptr)?,
            ))
        },
    }
}

fn borrow_from_materialized_view<'a>(
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

fn null_mask_for_view<'a>(
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
fn borrow_from_view_with_execution<'a>(
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
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask = null_mask_for_view(operation, nullable, null_mask_ptr, offset, len, null_count)?;

    borrow_from_materialized_view(operation, data_type, num_children, len, offset, data_ptr, null_mask, execution)
}

fn borrow_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let execution = execution_binding(operation, view.execution_domain())?;
    borrow_from_view_with_execution(view, execution, operation)
}

fn borrow_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(operation, column.execution_domain())?;
    borrow_from_view_with_execution(&view, execution, operation)
}

impl<'view> ColumnView<'view> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. Borrowed interop v1 accepts only
    /// fixed-width column types and preserves the column's execution binding.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }
}

impl TryBorrowColumnView for ColumnView<'_> {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_column_view(self, "interop::ColumnView::try_borrow_column_view")
    }
}

impl TableView<'_> {
    /// Project this low-level borrowed table view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. Borrowed interop v1 accepts only
    /// fixed-width columns and preserves the table execution binding.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }
}

impl TryBorrowTableView for TableView<'_> {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let operation = "interop::TableView::try_borrow_table_view";
        let num_rows = usize::try_from(
            self.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", operation)
        })?;
        let num_columns = usize::try_from(
            self.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        )
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", operation)
        })?;
        let execution = execution_binding(operation, self.execution_domain())?;
        let mut columns = Vec::with_capacity(num_columns);
        for index in 0..num_columns {
            let view = self
                .column(index as i32)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let borrowed = borrow_from_view_with_execution(&view, execution, operation)?;
            columns.push(borrowed);
        }
        BorrowedTableView::try_new(num_rows, columns, execution)
            .map_err(|error| error.with_context("operation", operation))
    }
}

impl Series {
    /// Convenience producer that lends this high-level `Series` as a shared
    /// borrowed interop column view without host copies on the input path.
    ///
    /// `Series` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to borrowed interop v1's fixed-width type support.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }
}

impl TryBorrowColumnView for Series {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_owned_column(self.as_column(), "interop::Series::try_borrow_column_view")
    }
}

impl DataFrame {
    /// Convenience producer that lends this high-level `DataFrame` as a shared
    /// borrowed interop table view without host copies on the input path.
    ///
    /// `DataFrame` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and does
    /// not promise zero-copy output for downstream consumer algorithms.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }
}

impl TryBorrowTableView for DataFrame {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let execution = execution_binding("interop::DataFrame::try_borrow_table_view", self.execution_domain())?;
        let columns = self
            .columns_map()
            .values()
            .map(|column| borrow_from_owned_column(column.as_ref(), "interop::DataFrame::try_borrow_table_view"))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, execution)
            .map_err(|error| error.with_context("operation", "interop::DataFrame::try_borrow_table_view"))
    }
}
